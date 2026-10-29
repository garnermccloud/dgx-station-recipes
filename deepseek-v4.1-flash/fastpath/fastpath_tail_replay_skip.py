"""tail_replay_skip: skip the per-step replay_start upload when the device already holds those values.

DeepseekV41ModelState.prepare_attn uploads the batch's SWA bounded-replay starts every step
(`self._replay_start_staging.copy_to_gpu(replay_start_np, out=self._replay_start[:num_reqs])`: a UVA-to-device copy
kernel on the critical path). Decode rows always carry 0, so in steady decoding the same zeros are re-uploaded each
step (NVIDIA vLLM #61868 item 18). This keeps a host mirror of what `_replay_start` holds on the device (it is the
copy's only writer; it starts as torch.zeros) and skips the copy when the requested values are already there.
The device buffer's contents are identical in both paths at every point a kernel reads them.
"""
from __future__ import annotations

try:
    from fastpath_hook import register
except ImportError:  # imported directly by tests
    def register(component, target):
        return lambda fn: fn

MODEL_STATE_MODULE = "vllm.models.deepseek_v41.nvidia.model_state"


class MirroredStaging:
    """Wraps the UvaBufferPool; copies to `device_buf` slices from offset 0 are skipped when the mirror matches."""

    def __init__(self, pool, device_buf):
        import numpy as np

        self.pool = pool
        self.device_buf = device_buf
        self.mirror = np.zeros(device_buf.numel(), dtype=np.int32)  # device_buf starts as torch.zeros
        self.skipped = 0

    def copy_to_gpu(self, x, out=None):
        import numpy as np

        if (out is not None and isinstance(x, np.ndarray) and out.data_ptr() == self.device_buf.data_ptr()
                and out.dim() == 1 and len(x) == out.numel()):
            n = len(x)
            if np.array_equal(self.mirror[:n], x):
                self.skipped += 1
                return out
            res = self.pool.copy_to_gpu(x, out=out)
            self.mirror[:n] = x
            return res
        return self.pool.copy_to_gpu(x, out=out)

    def __getattr__(self, name):
        return getattr(self.pool, name)


@register("tail_replay_skip", MODEL_STATE_MODULE)
def patch(module) -> None:
    import sys

    cls = module.DeepseekV41ModelState
    if getattr(cls, "_w5_replay_skip", False):
        return
    orig_init = cls.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if hasattr(self, "_replay_start_staging") and hasattr(self, "_replay_start"):
            self._replay_start_staging = MirroredStaging(self._replay_start_staging, self._replay_start)
            sys.stderr.write("DS41_FASTPATH tail_replay_skip: replay_start uploads skipped when unchanged\n")

    cls.__init__ = __init__
    cls._w5_replay_skip = True
