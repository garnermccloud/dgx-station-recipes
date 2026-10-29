"""ds41 fastpath component "a12_retire": MegaMoE scale factors move into compressible memory at load, once.

After each MegaMoE expert module finalizes its weights (the 40 target layers and the 3 DSpark drafter layers), its
two transformed scale-factor banks (L1 and L2, int32, MN-major) are copied into a generic-compression device
allocation (native_scales.NativeScales: the grant and byte equality with the source are checked) and the module's
tensors are pointed at the copies. The originals are then released and their storage expiry is asserted through the
Storage weak-reference API before KV-cache profiling and before any CUDA graph is captured, so the one graph set
captured afterwards reads only the new banks. There is no runtime selector, no second bank and no alternate capture.

Any failure raises during load. The component prints its ACTIVE line once all 43 layers are retired; the engine's
capture of each graph manager is checked to have produced exactly one FULL graph per descriptor. The memory guard
("a12_memory_guard") checks the free memory left after capture.

Hooks: models.deepseek_v4.nvidia.model.DeepseekV4MegaMoEExperts.finalize_weights,
v1.worker.gpu.model_runner.GPUModelRunner.load_model / capture_model, v1.worker.gpu.cudagraph_utils.CudaGraphManager.
"""
import functools
import gc
import json
import re
import sys
import threading

from fastpath_hook import activate, register
from native_scales import NativeScales

EXPERTS = 384           # routed experts per target layer (hot on the GB300 + cold on the RTX)
TARGET_LAYERS = 40
DRAFT_LAYERS = 3        # DSpark drafter MoE layers, numbered 40..42 here
DRAFT_GEOMETRY = (128, 3, 5120, 2304)  # num_experts, top_k, hidden, intermediate
L1_SHAPE, L2_SHAPE = (4608, 40), (5120, 18)  # per expert: (N, packed K scale groups)
ROWS = {}
MANAGERS = []
PROFILE_ONLY = False
LOCK = threading.RLock()


def write_receipt(event, **fields):
    payload = dict(schema='a12-retire-v1', event=event)
    payload.update(fields)
    sys.stderr.write('DS41_A12_RETIRE ' + json.dumps(payload, sort_keys=True) + '\n')
    sys.stderr.flush()
    return payload


def layer_id(experts):
    """(layer id, experts in this module's banks): target layers 0..39 from the hook's split, drafter 40..42."""
    split = getattr(experts, '_mp', None)
    if split is not None:
        if split.n_hot + split.n_cold != EXPERTS or split.n_hot <= 0:
            raise RuntimeError(f'a12_retire: rowmap split {split.n_hot}/{split.n_cold} is not a partition of '
                               f'{EXPERTS} experts')
        result = int(split.idx)
        if not 0 <= result < TARGET_LAYERS:
            raise RuntimeError(f'a12_retire: target layer {result} outside 0..{TARGET_LAYERS - 1}')
        return result, split.n_hot
    geometry = (experts.num_experts, experts.top_k, experts.hidden_size, experts.intermediate_size)
    if geometry != DRAFT_GEOMETRY:
        raise RuntimeError(f'a12_retire: unexpected drafter expert geometry {geometry}')
    match = re.search(r'(?:^|\.)layers\.(\d+)\.', experts.prefix)
    if not match:
        raise RuntimeError('a12_retire: unknown drafter prefix ' + experts.prefix)
    result = int(match.group(1))
    result = result + TARGET_LAYERS if result < DRAFT_LAYERS else result
    if not TARGET_LAYERS <= result < TARGET_LAYERS + DRAFT_LAYERS:
        raise RuntimeError(f'a12_retire: drafter layer {result} outside {TARGET_LAYERS}..'
                           f'{TARGET_LAYERS + DRAFT_LAYERS - 1}')
    return result, DRAFT_GEOMETRY[0]


def source_spec(tensor, e, n, q):
    import torch
    logical = e * n * q * 4
    if (not tensor.is_cuda or tensor.dtype != torch.int32
            or tuple(tensor.shape) != (e, n, q) or tensor.stride() != (n * q, 1, n)):
        raise RuntimeError(f'a12_retire: source scale factors {tuple(tensor.shape)} {tensor.stride()} '
                           f'{tensor.dtype} are not the MN-major ({e}, {n}, {q}) int32 layout')
    storage = tensor.untyped_storage()
    # A shared or larger backing storage cannot be certified released here.
    if tensor.storage_offset() != 0 or storage.nbytes() != logical:
        raise RuntimeError('a12_retire: source scale factors are not a dedicated complete storage')
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                logical_bytes=logical, storage_bytes=storage.nbytes())


def _replace(experts, layer, e):
    """Function scope drops every strong reference to the original scale-factor storage."""
    import torch
    l1, l2 = experts._transformed_l1_weights, experts._transformed_l2_weights
    specs = [source_spec(l1[1], e, *L1_SHAPE), source_spec(l2[1], e, *L2_SHAPE)]
    if l1[1].device != l2[1].device:
        raise RuntimeError('a12_retire: L1 and L2 scale factors are on different devices')
    handles = [pair[1].untyped_storage()._weak_ref() for pair in (l1, l2)]
    owners = []
    try:
        for pair in (l1, l2):
            owners.append(NativeScales(pair[1], True))
        if any(owner.storage.granted != 1 for owner in owners):
            raise RuntimeError('a12_retire: compressible allocation not granted')
        experts._transformed_l1_weights = (l1[0], owners[0].tensor)
        experts._transformed_l2_weights = (l2[0], owners[1].tensor)
        # The only persistent new owner. No reference to the source tensors is kept.
        experts._a12_retire_owners = tuple(owners)
        row = dict(layer=layer, experts=e, original_specs=specs,
                   original_logical_bytes=sum(s['logical_bytes'] for s in specs),
                   candidate_logical_bytes=sum(o.storage.nbytes for o in owners),
                   candidate_allocated_bytes=sum(o.storage.size for o in owners),
                   granted=[o.storage.granted for o in owners],
                   weights_unchanged=(experts._transformed_l1_weights[0] is l1[0]
                                      and experts._transformed_l2_weights[0] is l2[0]),
                   original_graphs_captured=0, fallback=False)
        return handles, row
    except BaseException:
        for handle in handles:
            torch.UntypedStorage._free_weak_ref(handle)
        raise


def assert_retired(handles):
    import torch
    try:
        state = [bool(torch.UntypedStorage._expired(handle)) for handle in handles]
        if not all(state):
            raise RuntimeError('a12_retire: original scale-factor storage is still live: ' + repr(state))
        return state
    finally:
        for handle in handles:
            torch.UntypedStorage._free_weak_ref(handle)


def replace_finalized(experts):
    import torch
    layer, e = layer_id(experts)
    if hasattr(experts, '_a12_retire_owners'):
        if layer not in ROWS:
            raise RuntimeError('a12_retire: owner without a layer record')
        return
    if layer in ROWS:
        raise RuntimeError(f'a12_retire: layer {layer} finalized twice')
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('a12_retire: finalize during graph capture')
    device = experts._transformed_l1_weights[1].device
    before = dict(device_free_bytes=torch.cuda.mem_get_info(device)[0],
                  torch_reserved_bytes=torch.cuda.memory_reserved(device))
    handles, row = _replace(experts, layer, e)
    torch.cuda.synchronize(device)
    gc.collect()
    row['original_storage_gone'] = assert_retired(handles)
    # Return the released allocator segments before KV profiling, capture and the memory guard.
    torch.cuda.empty_cache()
    row.update(status='RETIRED', before=before,
               after=dict(device_free_bytes=torch.cuda.mem_get_info(device)[0],
                          torch_reserved_bytes=torch.cuda.memory_reserved(device)))
    ROWS[layer] = row


def state():
    return dict(install_form='replace', original_banks_retained=TARGET_LAYERS + DRAFT_LAYERS - len(ROWS),
                alternate_graphs_retained=0, runtime_selector=False, diagnostic_banks_bytes=0, fallback=False,
                candidate_allocated_bytes=sum(r['candidate_allocated_bytes'] for r in ROWS.values()),
                original_logical_bytes=sum(r['original_logical_bytes'] for r in ROWS.values()),
                layers=sorted(ROWS), rows=list(ROWS.values()), managers=MANAGERS)


def _summary():
    s = state()
    return {k: v for k, v in s.items() if k != 'rows'}


@register('a12_retire', 'vllm.models.deepseek_v4.nvidia.model', defer_active=True)
def patch_model(module):
    cls = module.DeepseekV4MegaMoEExperts
    if getattr(cls, '_a12_retire_hook', False):
        raise RuntimeError('a12_retire finalizer already installed')
    original = cls.finalize_weights

    @functools.wraps(original)
    def finalize(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        replace_finalized(self)
        return result

    cls.finalize_weights = finalize
    cls._a12_retire_hook = True


@register('a12_retire', 'vllm.v1.worker.gpu.model_runner', defer_active=True)
def patch_runner(module):
    cls = module.GPUModelRunner
    original_load, original_capture = cls.load_model, cls.capture_model

    @functools.wraps(original_load)
    def load(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        expected = list(range(TARGET_LAYERS + DRAFT_LAYERS))
        if sorted(ROWS) != expected:
            raise RuntimeError(f'a12_retire: retired layers {sorted(ROWS)}; expected all of {expected}')
        if any(not all(r['original_storage_gone']) for r in ROWS.values()):
            raise RuntimeError('a12_retire: original scale-factor storage was not released')
        write_receipt('POST_LOAD', status='PASS', **_summary())
        activate('a12_retire')
        return result

    @functools.wraps(original_capture)
    def capture(self, *args, **kwargs):
        global PROFILE_ONLY
        with LOCK:
            previous = PROFILE_ONLY
            PROFILE_ONLY = bool(kwargs.get('profile_only', False))
            try:
                return original_capture(self, *args, **kwargs)
            finally:
                PROFILE_ONLY = previous

    cls.load_model, cls.capture_model = load, capture


@register('a12_retire', 'vllm.v1.worker.gpu.cudagraph_utils', defer_active=True)
def patch_graph(module):
    cls = module.CudaGraphManager
    original = cls.capture

    @functools.wraps(original)
    def capture(self, factory, *args, **kwargs):
        # Exactly one stock capture traversal; this wrapper only checks what it produced.
        profile = PROFILE_ONLY or self._capture_mem_samples is not None
        if not profile and self.graphs:
            raise RuntimeError('a12_retire requires a fresh graph manager')
        result = original(self, factory, *args, **kwargs)
        if profile:
            return result
        full = {d: g for d, g in self.graphs.items() if d.cg_mode.name == 'FULL'}
        expected = set(self._capture_descs.get(module.CUDAGraphMode.FULL, []))
        if set(full) != expected or len({id(g) for g in full.values()}) != len(full):
            raise RuntimeError('a12_retire: FULL descriptors / graph owners differ from the capture plan')
        MANAGERS.append(dict(manager=type(self).__name__, descriptors=len(full),
                             graph_keys=sorted(str(d) for d in full),
                             original_capture_traversals=1, runtime_selector=False,
                             original_graphs_captured=0, duplicate_graphs_retained=0))
        write_receipt('CAPTURE_' + type(self).__name__, status='PASS', descriptors=len(full),
                      candidate_allocated_bytes=state()['candidate_allocated_bytes'])
        return result

    cls.capture = capture
