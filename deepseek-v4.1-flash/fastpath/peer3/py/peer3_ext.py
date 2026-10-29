"""Build/load the peer tier v3 extensions (GB300 side and RTX side) for the current GPU.

Sources live in ../csrc. `load(side)` JIT-builds into PEER3_BUILD_DIR (default ../build; ninja makes an unchanged
rebuild a no-op). With PEER3_PREBUILT=<dir>, the module `peer3_<side>_sm<arch>*.so` in <dir> is imported instead,
so a read-only production mount needs no compiler run.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CSRC = HERE.parent / "csrc"
SOURCES = {"gb300": "peer3_gb300.cu", "rtx": "peer3_rtx.cu"}
# The image's toolkit lacks cusparse.h (torch's CUDAContext includes it); the pip CUDA headers have it but must come
# after the toolkit's own (their crt headers are a different minor version).
PIP_CUDA_INC = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"


def _arch() -> str:
    """Compute capability of the current device as "86", "103", "120" (PEER3_ARCH overrides, for compile-only)."""
    if os.environ.get("PEER3_ARCH"):
        return os.environ["PEER3_ARCH"]
    major, minor = torch.cuda.get_device_capability()
    return f"{major}{minor}"


def load(side: str, verbose: bool = False):
    """Import the `side` extension ("gb300" or "rtx") for the current device's compute capability."""
    if side not in SOURCES:
        raise ValueError(f"unknown peer3 extension side {side!r}")
    arch = _arch()
    name = f"peer3_{side}_sm{arch}" + os.environ.get("PEER3_VARIANT", "")
    if name in sys.modules:
        return sys.modules[name]
    prebuilt = os.environ.get("PEER3_PREBUILT") or None
    if prebuilt:
        found = sorted(Path(prebuilt).glob(f"{name}*.so"))
        if not found:
            raise FileNotFoundError(f"PEER3_PREBUILT={prebuilt} has no {name}*.so")
        spec = importlib.util.spec_from_file_location(name, found[0])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[name] = mod
        return mod
    from torch.utils.cpp_extension import load as jit_load

    build_dir = Path(os.environ.get("PEER3_BUILD_DIR", HERE.parent / "build")) / name
    build_dir.mkdir(parents=True, exist_ok=True)
    # The RTX side needs the arch-specific target (sm_120a) for the block-scaled QMMA.
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{arch[:-1]}.{arch[-1]}" + ("a" if side == "rtx" and arch == "120" else "")
    return jit_load(name=name, sources=[str(CSRC / SOURCES[side])], build_directory=str(build_dir),
                    extra_include_paths=[str(CSRC)], verbose=verbose,
                    extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo", "-isystem", PIP_CUDA_INC]
                    + os.environ.get("PEER3_DEFINES", "").split(),
                    extra_cflags=["-O3", "-std=c++17", "-isystem", PIP_CUDA_INC])
