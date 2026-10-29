"""ds41 fastpath component "quant_producer": the W4 seam produces wq_a's MXFP8 input for single-request decode.

At T = 6 and T = 8 (one request, FULL CUDA graph: the batch-1 DSpark verify widths) the seam that ends each layer
already computes the next attention input in BF16, and vLLM then quantizes it to MXFP8 for the wq_a + wkv GEMM
(FlashInfer's `mxfp8_e4m3_quantize`, a separate launch). This component swaps the seam launch for a qualified variant
of the same kernel (`lib_seam_quant_cs16.so`, cluster 16, one token per cluster, source in
`kernels/w4-boundary/seam-quant/`) that also writes the MXFP8 bytes and the swizzled scales, and makes the GEMM's
quantize call return them instead of launching its own kernel. BF16 rounding, the FP8 scale and the packing are the
stock ones; every other width and the drafter keep the stock quantize.

There is exactly one capture traversal and one production graph per descriptor (no alternate graph set, no runtime
selector). Each fused capture must engage on 39 of the 40 attention inputs (layer 0's input does not come from a
seam) and must not capture the stock quantize kernel for any of them; anything else raises before the server is
ready. The component prints its ACTIVE line after the target graph manager's capture has fused both widths.

Hooks: fastpath_boundary (seam launch + seam publication), kernels.linear.mxfp8.flashinfer (quantize call, plus a
rebind of the closure the hook's low-latency GEMM path captured), v1.worker.gpu.model_runner (load / capture
bracket) and v1.worker.gpu.cudagraph_utils (CudaGraphManager.capture).
"""
import functools
import inspect
import json
import sys
import threading

from fastpath_hook import activate, register

LOCK = threading.RLock()
MANAGERS = []
STATS = {'original_gets': 0, 'flagged_gets': 0, 'flagged_launches': 0}
INVALID_REASON = None
PROFILE_ONLY = False
CAPTURE_RECORD = None
NORM_PTRS = set()
FUSE_FORWARD = False
FUSED_WIDTHS = (6, 8)
ATTN_NORMS = 40           # target layers
SEAM_FED_INPUTS = 39      # every attention input except layer 0's


def log(message):
    sys.stderr.write('DS41_FASTPATH quant_producer: ' + message + '\n')
    sys.stderr.flush()


def invalidate(where, error):
    """Binding callbacks latch the first failure; the capture wrapper raises it before the server is ready."""
    global INVALID_REASON, FUSE_FORWARD
    if INVALID_REASON is None:
        INVALID_REASON = (f'{type(error).__name__}: {error}'
                          if isinstance(error, BaseException) else str(error))
        log('INVALID ' + where + ' ' + INVALID_REASON)
    FUSE_FORWARD = False


def selected(manager, module, desc):
    return (isinstance(manager, module.ModelCudaGraphManager)
            and desc.cg_mode.name == 'FULL' and desc.num_reqs == 1
            and desc.num_tokens in FUSED_WIDTHS and desc.num_ubatches == 1)


@register('quant_producer', 'fastpath_boundary', defer_active=True)
def patch_boundary(module):
    from producer_binding import patch_boundary as bind
    bind(module)


@register('quant_producer', 'vllm.model_executor.kernels.linear.mxfp8.flashinfer', defer_active=True)
def patch_linear(module):
    from producer_binding import patch_linear as bind
    bind(module)


@register('quant_producer', 'vllm.v1.worker.gpu.model_runner', defer_active=True)
def patch_runner(module):
    cls = module.GPUModelRunner
    if getattr(cls, '_quant_producer_replace', False):
        raise RuntimeError('quant_producer runner hook already installed')
    original_load, original_capture = cls.load_model, cls.capture_model

    @functools.wraps(original_load)
    def load_model(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        from producer_binding import rebind_upstream_quant
        binding = rebind_upstream_quant()
        log('low-latency GEMM quantize closure rebound after model load ' + json.dumps(binding, sort_keys=True))
        return result

    @functools.wraps(original_capture)
    def capture_model(self, *args, **kwargs):
        global PROFILE_ONLY
        with LOCK:
            previous = PROFILE_ONLY
            PROFILE_ONLY = bool(kwargs.get('profile_only', False))
            try:
                return original_capture(self, *args, **kwargs)
            finally:
                PROFILE_ONLY = previous

    cls.load_model, cls.capture_model = load_model, capture_model
    cls._quant_producer_replace = True


@register('quant_producer', 'vllm.v1.worker.gpu.cudagraph_utils', defer_active=True)
def patch_graph(module):
    cls = module.CudaGraphManager
    if getattr(cls, '_quant_producer_replace', False):
        raise RuntimeError('quant_producer graph hook already installed')
    original_capture = cls.capture

    @functools.wraps(original_capture)
    def capture(self, create_forward_fn, *args, **kwargs):
        import torch
        from producer_graph import retained_graphs, inventory
        with LOCK, torch.inference_mode():
            if INVALID_REASON:
                raise RuntimeError('quant_producer INVALID: ' + INVALID_REASON)
            # The engine's memory-profiling capture is throwaway; it runs stock.
            if PROFILE_ONLY or self._capture_mem_samples is not None:
                return original_capture(self, create_forward_fn, *args, **kwargs)
            if self.graphs:
                raise RuntimeError('quant_producer requires a fresh graph manager')
            records = {}

            def factory(desc, warmup):
                fn = create_forward_fn(desc, warmup)
                use_fusion = selected(self, module, desc)
                norms = set()
                if use_fusion:
                    model = inspect.getclosurevars(fn).nonlocals.get('model')
                    if model is None:
                        raise RuntimeError('quant_producer: target model absent from the capture factory')
                    norms = {m.weight.data_ptr() for name, m in model.named_modules()
                             if name.endswith('.attn_norm') and hasattr(m, 'weight')}
                    if len(norms) != ATTN_NORMS:
                        raise RuntimeError(f'quant_producer: expected {ATTN_NORMS} attention norms, found {len(norms)}')
                record = dict(eligible_inputs=set(), quant_nodes=set(), graph=None)

                @functools.wraps(fn)
                def forward(mode):
                    global CAPTURE_RECORD, NORM_PTRS, FUSE_FORWARD
                    previous = CAPTURE_RECORD, NORM_PTRS, FUSE_FORWARD
                    before = {key: STATS[key] for key in ('flagged_gets', 'flagged_launches')}
                    CAPTURE_RECORD = record if use_fusion and not warmup else None
                    NORM_PTRS, FUSE_FORWARD = norms, use_fusion
                    try:
                        result = fn(mode)
                        if INVALID_REASON:
                            raise RuntimeError('quant_producer binding INVALID: ' + INVALID_REASON)
                        if use_fusion and not warmup:
                            counts = {key: STATS[key] - before[key] for key in before}
                            want = {'flagged_gets': SEAM_FED_INPUTS, 'flagged_launches': SEAM_FED_INPUTS}
                            if counts != want:
                                raise RuntimeError(f'quant_producer engaged {counts} in {desc}; expected {want}')
                            if record['quant_nodes']:
                                raise RuntimeError(f'quant_producer: {desc} captured a stock quantize kernel '
                                                   'for a seam-fed input')
                            records[desc] = counts
                        return result
                    finally:
                        CAPTURE_RECORD, NORM_PTRS, FUSE_FORWARD = previous

                return forward

            with retained_graphs():
                result = original_capture(self, factory, *args, **kwargs)
            if INVALID_REASON:
                raise RuntimeError('quant_producer INVALID: ' + INVALID_REASON)
            full = {d: g for d, g in self.graphs.items() if d.cg_mode.name == 'FULL'}
            expected_full = set(self._capture_descs.get(module.CUDAGraphMode.FULL, []))
            if set(full) != expected_full or len({id(g) for g in full.values()}) != len(full):
                raise RuntimeError('quant_producer changed the FULL descriptor / graph ownership')
            fused = {d for d in full if selected(self, module, d)}
            if set(records) != fused:
                raise RuntimeError('quant_producer: fused capture records missing or extra')
            row = dict(manager=type(self).__name__, descriptors=len(full),
                       fused_t8_keys=sorted(str(d) for d in fused if d.num_tokens == 8),
                       fused_t6_keys=sorted(str(d) for d in fused if d.num_tokens == 6),
                       original_capture_traversals=1, duplicate_graphs_retained=0, runtime_selector=False,
                       inventory=[{k: inventory(g)[k] for k in ('node_count', 'edge_count', 'sha256')}
                                  for g in full.values()])
            MANAGERS.append(row)
            log('CAPTURE ' + json.dumps(row, sort_keys=True))
            if isinstance(self, module.ModelCudaGraphManager):
                widths = sorted({d.num_tokens for d in fused})
                if widths != list(FUSED_WIDTHS):
                    raise RuntimeError(f'quant_producer fused widths {widths}; the recipe captures {FUSED_WIDTHS} '
                                       'for one request (check the speculative schedule and capture sizes)')
                activate('quant_producer')
            return result

    cls.capture = capture
    cls._quant_producer_replace = True
