"""ds41 fastpath component "a12_memory_guard": refuse to serve when capture left too little free GPU memory.

Runs once, after the worker's compile/warm-up (KV cache allocated, every CUDA graph captured). It first re-checks
a12_retire's state against the real graph managers (all 43 scale-factor banks retired with their storage released,
one FULL graph per descriptor, no aliasing, one capture traversal), then requires

    free memory after capture >= profiled runtime peak + DS41_GPU_RESERVE_MIB + 150 MiB

where the profiled runtime peak is the worker's own activation peak minus the CUDA-graph estimate already folded
into it. DS41_GPU_RESERVE_MIB is memory another process on the same GB300 may still allocate after ds41 is up
(0 on a dedicated GPU; serve.sh passes the configured value). A shortfall raises before the server is ready; the
receipt line (A12_INSTALL_MEMORY) carries every term. There is no trimming and no retry.

Hook: v1.worker.gpu_worker.Worker.compile_or_warm_up_model.
"""
import functools
import gc
import json
import os
import sys

from fastpath_hook import activate, register
from a12_memory_policy import MIB, Refused, startup_floor

LAST_RECEIPT = None


def declared_reserve():
    value = os.environ.get('DS41_GPU_RESERVE_MIB')
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise Refused('DS41_GPU_RESERVE_MIB must be set to a decimal MiB count (0 on a dedicated GPU)')
    return int(value) * MIB


def retirement_facts(runner, member):
    """Check a12_retire's state() against the worker's final graph managers."""
    facts = member.state()
    if (facts.get('install_form') != 'replace'
            or facts.get('alternate_graphs_retained') != 0
            or facts.get('runtime_selector') is not False
            or facts.get('diagnostic_banks_bytes') != 0
            or facts.get('fallback') is not False):
        raise Refused('a12_retire must own one graph set with no selector, fallback or extra banks')
    rows = facts.get('rows', [])
    n_layers = member.TARGET_LAYERS + member.DRAFT_LAYERS
    if (facts.get('layers') != list(range(n_layers))
            or sorted(r.get('layer', -1) for r in rows) != list(range(n_layers))
            or facts.get('original_banks_retained') != 0):
        raise Refused(f'a12_retire must retire all {n_layers} original scale-factor banks')
    for row in rows:
        if (row.get('status') != 'RETIRED'
                or row.get('original_storage_gone') != [True, True]
                or row.get('weights_unchanged') is not True
                or row.get('granted') != [1, 1]
                or row.get('original_graphs_captured') != 0
                or row.get('fallback') is not False):
            raise Refused(f'a12_retire layer {row.get("layer")}: retirement proof is incomplete')
        after = row.get('after', {})
        if any(type(after.get(k)) is not int or after[k] < 0 for k in ('device_free_bytes', 'torch_reserved_bytes')):
            raise Refused('a12_retire: retirement lacks post-release byte counters')
    for key in ('candidate_allocated_bytes', 'original_logical_bytes'):
        if facts.get(key) != sum(r[key] for r in rows):
            raise Refused('a12_retire: bank accounting differs from the layer records')
    managers = (runner.cudagraph_manager, runner.speculator.query_cudagraph_manager)
    captures = facts.get('managers', [])
    if len(captures) != len(managers):
        raise Refused('a12_retire needs exactly one target and one drafter capture record')
    full_mode = sys.modules['vllm.v1.worker.gpu.cudagraph_utils'].CUDAGraphMode.FULL
    owners = []
    for manager in managers:
        name = type(manager).__name__
        full = {d: g for d, g in manager.graphs.items() if d.cg_mode.name == 'FULL'}
        planned = set(manager._capture_descs.get(full_mode, []))
        if not full or set(full) != planned:
            raise Refused(f'{name}: FULL graphs differ from the manager\'s capture descriptors')
        identities = [id(g) for g in full.values()]
        if len(set(identities)) != len(identities):
            raise Refused(f'{name}: FULL graph owners alias')
        records = [r for r in captures if r.get('manager') == name]
        if len(records) != 1:
            raise Refused(f'{name}: no unique a12_retire capture record')
        record = records[0]
        if (record.get('descriptors') != len(full)
                or record.get('graph_keys') != sorted(str(d) for d in full)
                or record.get('original_capture_traversals') != 1
                or record.get('duplicate_graphs_retained') != 0
                or record.get('runtime_selector') is not False
                or record.get('original_graphs_captured') != 0):
            raise Refused(f'{name}: capture record differs from the final graph owners')
        owners.append(dict(manager=name, full_count=len(full)))
    summary = {k: v for k, v in facts.items() if k != 'rows'}
    return dict(summary, checked_graph_managers=owners)


def emit(status, *, terms=None, facts=None, reason=None, phase):
    global LAST_RECEIPT
    LAST_RECEIPT = dict(schema='a12-memory-guard-v1', status=status, phase=phase,
                        terms=terms, retirement=facts, reason=reason)
    sys.stderr.write('A12_INSTALL_MEMORY ' + json.dumps(LAST_RECEIPT, sort_keys=True) + '\n')
    sys.stderr.flush()
    return LAST_RECEIPT


@register('a12_memory_guard', 'vllm.v1.worker.gpu_worker', defer_active=True)
def patch_worker(module):
    cls = module.Worker
    original = cls.compile_or_warm_up_model
    if getattr(original, '_a12_memory_guard', False):
        raise RuntimeError('a12_memory_guard already bound')

    @functools.wraps(original)
    def startup(self, *args, **kwargs):
        try:
            reserve = declared_reserve()
        except Exception as exc:
            emit('REFUSED', reason=str(exc), phase='before_warmup')
            raise
        result = original(self, *args, **kwargs)
        import torch
        import vllm.envs as envs
        terms = facts = None
        try:
            member = sys.modules['fastpath_a12_retire']
            facts = retirement_facts(self.model_runner, member)
            torch.cuda.synchronize(self.device)
            gc.collect()
            torch.cuda.empty_cache()
            raw = dict(startup_free_bytes=int(self.init_snapshot.free_memory),
                       requested_bytes=int(self.requested_memory),
                       profiled_peak_activation_bytes=int(self.peak_activation_memory),
                       applied_graph_estimate_bytes=(int(self.cudagraph_memory_estimate)
                                                     if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS else 0),
                       post_capture_free_bytes=int(torch.cuda.mem_get_info(self.device)[0]))
            terms = startup_floor(raw, reserve_bytes=reserve)
            terms['kv_bytes'] = int(self.available_kv_cache_memory_bytes)
            if terms['status'] != 'PASS':
                raise Refused('free memory after capture is below runtime peak + reserve + 150 MiB; '
                              'lower DS41_UTIL')
        except Exception as exc:
            emit('REFUSED', terms=terms, facts=facts, reason=f'{type(exc).__name__}: {exc}',
                 phase='after_capture_warmup_gc')
            raise
        self._a12_memory_receipt = emit('PASS', terms=terms, facts=facts, phase='after_capture_warmup_gc')
        activate('a12_memory_guard')
        return result

    startup._a12_memory_guard = True
    cls.compile_or_warm_up_model = startup
