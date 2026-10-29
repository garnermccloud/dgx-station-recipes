"""a12_memory_guard policy: the free-memory floor after capture. Pure arithmetic, no device work."""
MIB = 1 << 20
MARGIN_BYTES = 150 * MIB


class Refused(RuntimeError):
    pass


def integer(value, name):
    if type(value) is not int or value < 0:
        raise Refused(name + ' must be an explicit nonnegative integer byte count')
    return value


def startup_floor(terms, *, reserve_bytes):
    """PASS when free memory after capture covers the profiled runtime peak + the other-tenant reserve + margin.

    terms: the worker's own numbers (startup_free_bytes, requested_bytes, profiled_peak_activation_bytes,
    applied_graph_estimate_bytes, post_capture_free_bytes). The runtime peak is the profiled activation peak minus
    the CUDA-graph estimate the profiler already folded into it (the graphs are resident after capture).
    reserve_bytes: memory another process on this GPU may still allocate after ds41 is up (0 on a dedicated GPU).
    """
    r = integer(reserve_bytes, 'reserve_bytes')
    values = {k: integer(terms.get(k), k) for k in (
        'startup_free_bytes', 'requested_bytes', 'profiled_peak_activation_bytes',
        'applied_graph_estimate_bytes', 'post_capture_free_bytes')}
    h = values['profiled_peak_activation_bytes'] - values['applied_graph_estimate_bytes']
    if h < 0:
        raise Refused('profiled activation bytes are smaller than the applied graph estimate')
    required = h + r + MARGIN_BYTES
    return dict(values, profiled_runtime_peak_bytes=h, reserve_bytes=r, margin_bytes=MARGIN_BYTES,
                unrequested_slack_bytes=values['startup_free_bytes'] - values['requested_bytes'],
                required_free_bytes=required,
                headroom_bytes=values['post_capture_free_bytes'] - required,
                status='PASS' if values['post_capture_free_bytes'] >= required else 'REFUSED')
