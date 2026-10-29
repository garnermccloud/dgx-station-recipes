# sitecustomize.py - bind-mounted over /usr/lib/python3.12/sitecustomize.py when DS41_FASTPATH is set.
# Installs the MegaMoE hot/cold-tier hook, then the ds41 fastpath kernels named in DS41_FASTPATH
# (comma-separated, e.g. "attn,gemm"); see fastpath/README.md. A hook failure is fatal, never a silent stock boot.
from importlib.util import find_spec as _find_spec

if _find_spec("apport_python_hook") is not None:
    import apport_python_hook
    apport_python_hook.install()

import os
import sys

_MEGA_HOOK = "/w/mega_peer_hook.py"  # the release's fixed mount; there is no env switch
if os.path.exists(_MEGA_HOOK):
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("mega_peer_hook", _MEGA_HOOK)
        _m = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
        _m.install()
    except Exception as _e:
        sys.stderr.write(f"MEGA_PEER hook FAILED to install: {_e!r}\n")
        raise
elif os.environ.get("DS41_FASTPATH"):
    raise RuntimeError(f"MEGA_PEER hook missing at {_MEGA_HOOK}")

if os.environ.get("DS41_FASTPATH"):
    try:
        fastpath_dir = os.environ.get("DS41_FASTPATH_DIR", "/fastpath")
        if not os.path.isdir(fastpath_dir):
            raise RuntimeError(f"DS41_FASTPATH_DIR missing: {fastpath_dir}")
        sys.path.insert(0, fastpath_dir)
        import fastpath_hook
        fastpath_hook.install(os.environ["DS41_FASTPATH"])
    except Exception as _e:
        sys.stderr.write(f"DS41_FASTPATH hook FAILED to install: {_e!r}\n")
        raise
