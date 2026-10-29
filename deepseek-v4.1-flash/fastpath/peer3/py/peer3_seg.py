"""The peer tier v3 shared segment: a /dev/shm file mapped and registered (mapped, portable) in each CUDA process.

Both sides open the same path; the GB300 side creates and zeroes it (control words, token records) at startup.
"""
from __future__ import annotations

import mmap
import os
import struct

import torch

SEG_BYTES_DEFAULT = None  # filled from the extension (peer3::kSegBytes)


class Segment:
    """One process's view of the segment: host mmap + device pointer of the registered mapping."""

    def __init__(self, path: str, nbytes: int, device: torch.device, create: bool, zero_header: bool):
        self.path, self.nbytes, self.device = path, nbytes, device
        try:
            fd = os.open(path, os.O_RDWR)
        except FileNotFoundError:
            if not create:
                raise
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            if os.fstat(fd).st_size < nbytes:
                os.ftruncate(fd, nbytes)
            self.buf = mmap.mmap(fd, nbytes)
        finally:
            os.close(fd)
        self.host = torch.frombuffer(self.buf, dtype=torch.uint8)
        if zero_header:
            self.host[: 4096 + 64 * 128].zero_()
        from cuda.bindings import runtime as rt

        flags = rt.cudaHostRegisterMapped | rt.cudaHostRegisterPortable
        with torch.cuda.device(device):
            (err,) = rt.cudaHostRegister(self.host.data_ptr(), nbytes, flags)
            if err != rt.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaHostRegister({path}) failed: {err}")
            err, dptr = rt.cudaHostGetDevicePointer(self.host.data_ptr(), 0)
            if err != rt.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaHostGetDevicePointer failed: {err}")
        self.dptr = int(dptr)

    def u64(self, offset: int) -> int:
        """Host read of a control word (for logging and the sidecar's control loop)."""
        return struct.unpack_from("<Q", self.buf, offset)[0]

    def set_u64(self, offset: int, value: int) -> None:
        struct.pack_into("<Q", self.buf, offset, value)
