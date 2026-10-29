"""JIT build of the W3 CUDA extensions (torch cpp_extension, nvcc 13, sm_103a)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def load(name: str, sources: tuple[str, ...], defines: tuple[str, ...] = ()):
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.3a")
    from torch.utils.cpp_extension import load as _load

    build_dir = ROOT / "build" / name
    build_dir.mkdir(parents=True, exist_ok=True)
    return _load(
        name=name,
        sources=[str(ROOT / s) for s in sources],
        extra_include_paths=[str(ROOT / "include")],
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo",
                           "-gencode=arch=compute_103a,code=sm_103a", *[f"-D{d}" for d in defines]],
        extra_cflags=["-O3", "-std=c++17"],
        build_directory=str(build_dir),
        verbose=False,
    )
