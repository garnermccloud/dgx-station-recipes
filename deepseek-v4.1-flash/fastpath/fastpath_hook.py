"""ds41 fastpath: install faster decode kernels into vLLM by patching modules as they are imported.

Each component registers (target module, patch function). `install("attn,gemm")` installs only the named components,
so a boot can A/B one kernel at a time; with DS41_FASTPATH unset this module is never imported.
"""
import importlib.abc
import importlib.util
import os
import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager

_COMPONENTS: dict[str, list[tuple[str, Callable]]] = {}
_ACTIVE: set[str] = set()
_DEFERRED: set[str] = set()


def register(component: str, target: str, defer_active: bool = False) -> Callable[[Callable], Callable]:
    """Decorator: patch function `fn(module)` runs when `target` is imported, if `component` is enabled.

    defer_active: the component calls `activate(component)` itself once it has really engaged (megamoe: after the JIT
    include root is live; peer3: when the finish fold is set up)."""
    if defer_active:
        _DEFERRED.add(component)

    def wrap(fn: Callable) -> Callable:
        _COMPONENTS.setdefault(component, []).append((target, fn))
        return fn
    return wrap


def activate(component: str) -> None:
    """Log the one ACTIVE line for `component`, once, after it has really engaged."""
    if component not in _ACTIVE:
        _ACTIVE.add(component)
        _log(f"ACTIVE {component}")


@contextmanager
def bounded(what: str, seconds: float):
    """Abort the boot with one FATAL line if a blocking post-load wait exceeds `seconds`."""
    done = threading.Event()

    def _watchdog():
        if not done.wait(seconds):
            _log(f"FATAL {what} did not finish within {seconds:g}s; aborting boot")
            os._exit(1)

    threading.Thread(target=_watchdog, name=f"ds41-bounded-{what[:32]}", daemon=True).start()
    try:
        yield
    finally:
        done.set()


def _log(msg: str) -> None:
    sys.stderr.write(f"DS41_FASTPATH {msg}\n")


class _Finder(importlib.abc.MetaPathFinder):
    """Runs `patch(module)` right after `target` executes for the first time."""

    def __init__(self, target: str, patch: Callable, component: str) -> None:
        self.target, self.patch, self.component = target, patch, component

    def find_spec(self, name, path, target=None):
        if name != self.target:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig=orig_exec, _patch=self.patch, _component=self.component):
            _orig(module)
            _run_patch(_component, _patch, module)

        spec.loader.exec_module = exec_module
        return spec


def _run_patch(component: str, patch: Callable, module) -> None:
    patch(module)
    if component not in _DEFERRED:
        activate(component)


def install(spec: str) -> None:
    """Install the comma-separated components in `spec`; unknown names fail loudly.

    A component is logged ACTIVE only after its first patch function has actually run (or, for deferred components,
    after the component calls `activate()` when it really engages).
    """
    import components

    if not components.REGISTERED:
        raise RuntimeError("components.py registered no modules")
    wanted = [c.strip() for c in spec.split(",") if c.strip()]
    unknown = [c for c in wanted if c not in _COMPONENTS]
    if unknown:
        raise ValueError(f"unknown DS41_FASTPATH components {unknown}; known: {sorted(_COMPONENTS)}")
    for component in wanted:
        for target, patch in _COMPONENTS[component]:
            if target in sys.modules:
                _run_patch(component, patch, sys.modules[target])
            else:
                sys.meta_path.insert(0, _Finder(target, patch, component))
