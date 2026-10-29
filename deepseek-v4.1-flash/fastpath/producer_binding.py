"""quant_producer bindings: reuse the boundary's seam allocation and consumer dispatch, change one kernel call."""
import ctypes
from pathlib import Path

LIB_NAME = 'lib_seam_quant_cs16.so'
SEAM_CLUSTER, SEAM_GSIZE = 16, 1   # the qualified variant is built for exactly this seam configuration
BOUNDARY = None
PRIVATE = None


def ctl():
    import fastpath_quant_producer
    return fastpath_quant_producer


def private_library(module):
    """The CS16 seam variant; its parameter block must match the boundary's SeamParams mirror."""
    global PRIVATE
    if PRIVATE is None:
        lib = ctypes.CDLL(str(Path(__file__).with_name(LIB_NAME)))
        P, I = ctypes.c_void_p, ctypes.c_int
        lib.w4_seam_launch.argtypes = [ctypes.POINTER(module.SeamParams), I, I, I, P]
        lib.w4_abi.argtypes = [I]
        if (lib.w4_abi(0), lib.w4_abi(1)) != (ctypes.sizeof(module.SeamParams), module.SeamParams.dbg.offset):
            raise RuntimeError(f'{LIB_NAME}: seam parameter layout differs from fastpath_boundary.SeamParams')
        PRIVATE = lib
    return PRIVATE


def patch_boundary(module):
    global BOUNDARY
    import functools
    c = ctl()
    if module.MXFP8_OUT or module.CLUSTER != SEAM_CLUSTER or module.GSIZE != SEAM_GSIZE:
        raise RuntimeError(f'quant_producer needs the boundary seam at cluster {SEAM_CLUSTER}, gsize {SEAM_GSIZE}, '
                           f'MXFP8_OUT off; found cluster {module.CLUSTER}, gsize {module.GSIZE}, '
                           f'MXFP8_OUT {module.MXFP8_OUT}')
    BOUNDARY = module
    original_lib, original_seam = module.lib, module.seam

    @functools.wraps(original_lib)
    def lib():
        value = original_lib()
        if not getattr(value, '_producer_hook', False):
            original = value.w4_seam_launch

            def launch(params, cluster, gsize, pdl, stream):
                try:
                    p = ctypes.cast(params, ctypes.POINTER(module.SeamParams)).contents
                    if c.FUSE_FORWARD and p.fi_q and p.T in c.FUSED_WIDTHS and int(p.norm_w or 0) in c.NORM_PTRS:
                        rc = private_library(module).w4_seam_launch(params, cluster, gsize, pdl, stream)
                        if rc:
                            raise RuntimeError('CS16 seam launch refused: ' + str(rc))
                        c.STATS['flagged_launches'] += 1
                        return rc
                except Exception as exc:
                    c.invalidate('PRODUCER_DISPATCH', exc)
                return original(params, cluster, gsize, pdl, stream)

            value.w4_seam_launch = launch
            value._producer_hook = True
        return value

    @functools.wraps(original_seam)
    def seam(*args, **kwargs):
        record = c.CAPTURE_RECORD
        eligible = False
        try:
            # Only the current seam's input is eligible. Graph-pool addresses may be reused by a later FFN seam;
            # pointer history is not identity.
            if record is not None:
                record['eligible_inputs'].clear()
            norm = kwargs.get('norm_w', args[5] if len(args) > 5 else None)
            eligible = norm is not None and norm.data_ptr() in c.NORM_PTRS and args[0].shape[0] in c.FUSED_WIDTHS
        except Exception as exc:
            c.invalidate('SEAM_BINDING', exc)
        emit = c.FUSE_FORWARD and eligible and not c.INVALID_REASON
        kwargs['mxfp8'] = bool(emit)
        result = original_seam(*args, **kwargs)
        try:
            if record is not None and eligible and result[1] is not None:
                record['eligible_inputs'].add(result[1].data_ptr())
            if emit and result[3] is not None:
                # Some Engram-entry callers discard the fourth seam return. Publish here too; every consumer reads
                # only _PREQ[-1].
                module._PREQ.append(dict(x=result[1], q=result[3][0], sf=result[3][1]))
        except Exception as exc:
            c.invalidate('SEAM_RECORD', exc)
        return result

    module.lib = lib
    module.seam = seam


def patch_linear(module):
    import functools
    original = module.mxfp8_e4m3_quantize

    @functools.wraps(original)
    def quantize(x, is_sf_swizzled_layout=False, alignment=0):
        c = ctl()
        b = BOUNDARY
        try:
            if (c.FUSE_FORWARD and not c.INVALID_REASON and b is not None and b._PREQ and is_sf_swizzled_layout
                    and alignment == 0):
                e = b._PREQ[-1]
                li = e['x']
                if x.data_ptr() == li.data_ptr() and tuple(x.shape) == tuple(li.shape):
                    c.STATS['flagged_gets'] += 1
                    return e['q'], e['sf']
        except Exception as exc:
            c.invalidate('PREQUANT_BINDING', exc)
        result = original(x, is_sf_swizzled_layout, alignment)
        record = c.CAPTURE_RECORD
        try:
            if (record is not None and x.data_ptr() in record['eligible_inputs'] and is_sf_swizzled_layout
                    and alignment == 0):
                import torch
                if torch.cuda.is_current_stream_capturing():
                    from producer_graph import last_kernel
                    graph, node = last_kernel()
                    if record['graph'] is not None and record['graph'] != graph:
                        raise RuntimeError('selected quant calls crossed capture graphs')
                    record['graph'] = graph
                    record['quant_nodes'].add(node)
        except Exception as exc:
            c.invalidate('QUANT_BINDING', exc)
        return result

    module.mxfp8_e4m3_quantize = quantize
    module._producer_quantize_wrapper = quantize


def rebind_upstream_quant():
    """The hook's low-latency GEMM path captured the quantizer in a closure; module assignment cannot reach it."""
    import sys
    name = 'vllm.model_executor.kernels.linear.mxfp8.flashinfer'
    module = sys.modules.get(name)
    wrapper = getattr(module, '_producer_quantize_wrapper', None)
    cls = getattr(module, 'FlashInferCutedslMxfp8LinearKernel', None)
    fn = getattr(cls, 'apply_weights', None)
    if not callable(wrapper) or not getattr(cls, '_ll_gemm', False) or not getattr(fn, '__closure__', None):
        raise RuntimeError('low-latency GEMM quantize closure absent after model load')
    names = fn.__code__.co_freevars
    if names.count('mxfp8_e4m3_quantize') != 1:
        raise RuntimeError('low-latency GEMM quantize closure cell is not unique')
    cell = fn.__closure__[names.index('mxfp8_e4m3_quantize')]
    old = cell.cell_contents
    if old is not wrapper and old is not wrapper.__wrapped__:
        raise RuntimeError('low-latency GEMM quantize differs from the quantizer quant_producer wrapped')
    cell.cell_contents = wrapper
    if cell.cell_contents is not wrapper:
        raise RuntimeError('low-latency GEMM quantize closure rebind failed')
    return dict(class_name=cls.__qualname__, freevar='mxfp8_e4m3_quantize', apply_module=fn.__module__,
                wrapper_module=wrapper.__module__, closure_is_wrapper=True)
