"""suffix: copy (suffix-match) drafting on top of DSpark, with a wide verify for batch-1 copy runs.

Agentic outputs re-emit text that is already in the context (edit old/new strings, file bodies, paths, commands).
At batch 1 this component drafts such runs by copying from the request's own history, on the GPU, every step
(a device mirror of the history, brute-force longest-suffix match over every position: ~9 us at 512K tokens,
~22 us GPU per step in total with the chain, one-hot q rows and the flag copy):

  draft chain (K = 31 tokens), built after DSpark's 5 drafts in the drafter's propose():
    if the history's longest suffix match is >= W9_GPU_TH tokens:  copy the K tokens that followed it
    else:  DSpark's 5 drafts, then the copy that follows (history + those 5 drafts), or a filler token
  verify width per step, chosen by the scheduler from the latest GPU match length (one step old):
    1 + K rows if the match length is >= W9_TRIGGER, else 1 + 5 rows (today's batch-1 width)
  batch > 1: untouched (DSpark only, schedule as configured).

Copied tokens are verified against q = one-hot(token), written into the rejection sampler's draft-logits cache, so the
output distribution is exactly the target's at any temperature (block or token-wise rejection), and greedy decoding
accepts a copied token iff it is the target argmax.

Boot: DS41_FASTPATH=suffix. The recipe is fixed: verify width 32, suffix-match trigger 24, pure-copy threshold 16,
normal K 5, DSpark block 7, dynamic K on with cost ratio 1.09, early decision on, K_LO block off. The component
rewrites the boot's --speculative-config to num_speculative_tokens = 31 with a batch-1 schedule entry of the same size
(so the runner sizes its buffers and captures 1 x 32 decode graphs), keeps DSpark's own block at 7, and lets the
scheduler pick 5 or 31 drafts per batch-1 step.

Per-request dynamic draft length (on): a batch-1 normal step verifies K_LO = 5 or K_HI = 7 drafts, whichever gives
more expected tokens per unit step cost; the estimate uses the request's acceptance history, the measured step cost
ratio, and a global prior. See `DynK` below for the estimator.
"""

from __future__ import annotations

import statistics
import sys
import time
from collections import deque

try:
    from fastpath_hook import register
except ImportError:  # imported directly by tests
    def register(component, target):
        return lambda fn: fn

WIDTH = 32
TRIGGER = 24
GPU_TH = 16
NORMAL_K = 5
LOG_EVERY = 20000
MIRROR_SIDE = True
K_WIDE = WIDTH - 1
DSPARK_K = 7  # the 1017 checkpoint's dspark_block_size
DYNK_LO = 5
DYNK = True
DYNK_COST = "1.09"
DYNK_DECAY = 0.9
DYNK_N0 = 4.0
DYNK_PRIOR = (2.6, 0.3, 1.15)
DYNK_EXPLORE = 16
DYNK_OPTIMISM = 0.2
DYNK_EXPLORE_LO = 64
DYNK_EARLY = True  # decide in propose(), short Markov loop on K_LO
DYNK_BLOCK5 = False
BLOCK5_CHECK = False

# Shared between the scheduler and the model runner, which run in the same EngineCore process (TP1, uniproc).
STATE: dict = {"runner": None, "drafter": None, "wide_steps": 0, "normal_steps": 0,
               # W6_DYNK_EARLY: the width decided by the last propose() for the next schedule (req_id, k), and
               # counters: decisions applied, K_LO decisions (short Markov loop), wide steps ruled out, fallbacks
               "early": None, "proposed": None, "early_on": True, "early_used": 0, "early_lo": 0,
               "early_fallback": 0, "short_runs": 0, "block5_chk": None}


def _log(msg: str) -> None:
    sys.stderr.write(f"DS41_FASTPATH suffix: {msg}\n")


class _Req:
    """One request's evidence: per-step decayed sums (weight w, min(a, lo) m, [a >= lo] f), per-sample decayed tail
    sums (weight tw, sum tg of min(a, hi) - lo over hi steps that reached lo, plus optimistic prior pseudo-samples) and
    the run of consecutive lo choices (for exploration)."""

    __slots__ = ("w", "m", "f", "tw", "tg", "lo_run", "hi_run")

    def __init__(self, w: float = 0.0, m: float = 0.0, f: float = 0.0, tw: float = 0.0, tg: float = 0.0):
        self.w, self.m, self.f, self.tw, self.tg, self.lo_run, self.hi_run = w, m, f, tw, tg, 0, 0


class DynK:
    """Per-request choice between lo and hi drafts for batch-1 normal steps (see the module docstring)."""

    def __init__(self, lo: int, hi: int, cost: str, decay: float, n0: float, prior: tuple[float, float, float],
                 explore: int = 16, tail_decay: float = 0.9, tail_n0: float = 4.0, glob_tail_decay: float = 0.98,
                 glob_tail_n0: float = 4.0, optimism: float = 0.2, explore_lo: int = 64, cost_window: int = 64):
        self.lo, self.hi, self.decay, self.n0 = lo, hi, decay, n0
        self.explore, self.tail_decay, self.tail_n0 = explore, tail_decay, tail_n0
        self.explore_lo = explore_lo
        self.glob_tail_decay, self.glob_tail_n0, self.optimism = glob_tail_decay, glob_tail_n0, optimism
        self.auto = cost == "auto"
        self.fixed_ratio = 1.10 if self.auto else float(cost)
        pm, pf, self.g_prior = prior
        self.glob = _Req(8.0, pm * 8.0, pf * 8.0)
        self.reqs: dict[str, _Req] = {}
        self.step_us = {lo: 0.0, hi: 0.0}  # median of the recent samples
        self.step_n = {lo: 0, hi: 0}
        self.step_win = {lo: deque(maxlen=cost_window), hi: deque(maxlen=cost_window)}
        self.rejected = 0
        self.chosen = {lo: 0, hi: 0}
        self.explored = 0
        self.explored_lo = 0
        self.last_t: float | None = None

    def _tail(self, r: _Req, sample: float | None) -> None:
        """A hi step that reached lo is a tail sample (decays the older ones); a full lo accept, where a hi step would
        have produced one, adds an optimistic pseudo-sample of the prior, so a tail estimate that only hi steps could
        correct drifts back toward the prior instead of locking lo in."""
        for e, d, w in ((r, self.tail_decay, 1.0), (self.glob, self.glob_tail_decay, 0.25)):
            if sample is not None:
                e.tw = d * e.tw + 1.0
                e.tg = d * e.tg + sample
            else:
                e.tw += self.optimism * w
                e.tg += self.optimism * w * self.g_prior

    def observe(self, req_id: str, k: int, a: int) -> None:
        """Scheduler accounting for one finished step of one request (k scheduled drafts, a accepted)."""
        if k not in (self.lo, self.hi):
            return
        a = max(0, min(a, k))
        r = self.reqs.get(req_id)
        if r is None:
            r = self.reqs[req_id] = _Req()
        full = 1.0 if a >= self.lo else 0.0
        for e, d in ((r, self.decay), (self.glob, 0.998)):
            e.w = d * e.w + 1.0
            e.m = d * e.m + min(a, self.lo)
            e.f = d * e.f + full
        if full:
            self._tail(r, float(min(a, self.hi) - self.lo) if k == self.hi else None)

    def observe_interval(self, k: int | None, now: float) -> None:
        """Output arrival of a step; k = its drafts if it was a batch-1 decode step (else None)."""
        last, self.last_t = self.last_t, (now if k is not None else None)
        if k not in self.step_us or last is None:
            return
        us = (now - last) * 1e6
        win = self.step_win[k]
        if us > 200_000 or (len(win) >= 16 and us > 1.5 * self.step_us[k]):
            self.rejected += 1
            return  # an idle gap or a host stall, not a step
        win.append(us)
        self.step_n[k] += 1
        self.step_us[k] = statistics.median(win)

    def ratio(self) -> float:
        """Step-time ratio hi : lo."""
        if self.auto and min(self.step_n.values()) >= 16 and self.step_us[self.lo] > 0:
            return min(max(self.step_us[self.hi] / self.step_us[self.lo], 1.0), 1.5)
        return self.fixed_ratio

    def glob_g(self) -> float:
        gl, n = self.glob, self.glob_tail_n0
        return (gl.tg + n * self.g_prior) / (gl.tw + n)

    def estimates(self, req_id: str) -> tuple[float, float, float]:
        """(m, f, g) for a request: its own evidence shrunk toward the global estimates (the global g toward the
        prior)."""
        gl = self.glob
        gm, gf, gg = gl.m / gl.w, gl.f / gl.w, self.glob_g()
        r = self.reqs.get(req_id)
        if r is None:
            return gm, gf, gg
        n0, nt = self.n0, self.tail_n0
        return (r.m + n0 * gm) / (r.w + n0), (r.f + n0 * gf) / (r.w + n0), (r.tg + nt * gg) / (r.tw + nt)

    def choose(self, req_id: str) -> int:
        """Drafts to verify in the request's next batch-1 normal step; every `explore`-th consecutive lo choice of a
        request is turned into hi, so the tail keeps getting fresh samples."""
        m, f, g = self.estimates(req_id)
        k = self.hi if 1.0 + m + f * g > self.ratio() * (1.0 + m) else self.lo
        r = self.reqs.get(req_id)
        if r is not None:
            if k == self.lo:
                r.lo_run += 1
                if self.explore and r.lo_run >= self.explore:
                    k = self.hi
                    self.explored += 1
            else:
                r.hi_run += 1
                if self.explore_lo and r.hi_run >= self.explore_lo:
                    k = self.lo
                    self.explored_lo += 1
            if k == self.hi:
                r.lo_run = 0
            else:
                r.hi_run = 0
        self.chosen[k] += 1
        return k

    def prune(self, live) -> None:
        """Forget finished requests."""
        for rid in [r for r in self.reqs if r not in live]:
            del self.reqs[rid]

    def summary(self) -> str:
        gl = self.glob
        return (f"dynamic K: chose {self.lo} x{self.chosen[self.lo]}, {self.hi} x{self.chosen[self.hi]} (explore "
                f"{self.explored} up, {self.explored_lo} down; cost samples rejected {self.rejected}); global m {gl.m / gl.w:.2f} f {gl.f / gl.w:.3f} g {self.glob_g():.2f}; step us "
                f"{self.lo}: {self.step_us[self.lo]:.0f} (n {self.step_n[self.lo]}), {self.hi}: "
                f"{self.step_us[self.hi]:.0f} (n {self.step_n[self.hi]}); ratio {self.ratio():.3f}"
                f"{' (auto)' if self.auto and min(self.step_n.values()) >= 16 else ' (fixed)'}"
                + (f"; early decisions {STATE['early_used']} (K_LO {STATE['early_lo']}, scheduler fallbacks "
                   f"{STATE['early_fallback']}, on {STATE['early_on']})" if DYNK_EARLY else "")
                + (f"; K_LO blocks drafted {STATE['short_runs']}" if DYNK_BLOCK5 else "")
                + (f"; block-{self.lo} check (steps, draft mismatches, q mismatches) "
                   f"{STATE['block5_chk'].tolist() if STATE['block5_chk'] is not None else None}" if BLOCK5_CHECK
                   else ""))


DYN = DynK(DYNK_LO, DSPARK_K, DYNK_COST, DYNK_DECAY, DYNK_N0, DYNK_PRIOR, DYNK_EXPLORE,
           optimism=DYNK_OPTIMISM, explore_lo=DYNK_EXPLORE_LO) if DYNK else None


def widen_schedule(schedule: list, k_wide: int, normal_k: int) -> list:
    """[[1,4,5],...] -> [[1,1,k_wide],[2,4,5],...]: batch 1 gets its own entry, other batch sizes keep theirs."""
    out = []
    for lo, hi, k in sorted(([int(a), int(b), int(c)] for a, b, c in schedule), key=lambda e: e[0]):
        if lo == 1:
            out.append([1, 1, k_wide])
            if hi > 1:
                out.append([2, hi, k])
        else:
            out.append([lo, hi, k])
    if not out or out[0][0] != 1:
        out.insert(0, [1, 1, k_wide])
    return out


@register("suffix", "vllm.engine.arg_utils")
def patch_engine_args(module) -> None:
    """Size the speculative config for the wide width: K drafts max, batch-1 schedule entry K."""
    cls = module.EngineArgs
    orig = cls.create_speculative_config

    def create_speculative_config(self, target_model_config, target_parallel_config):
        sc = self.speculative_config
        if isinstance(sc, dict) and sc.get("method") == "dspark" and K_WIDE > DSPARK_K:
            sc = dict(sc)
            sched = sc.get("num_speculative_tokens_per_batch_size") or [[1, 1, NORMAL_K]]
            sc["num_speculative_tokens_per_batch_size"] = widen_schedule(sched, K_WIDE, NORMAL_K)
            sc["num_speculative_tokens"] = K_WIDE
            self.speculative_config = sc
            _log(f"speculative config: num_speculative_tokens {K_WIDE}, schedule "
                 f"{sc['num_speculative_tokens_per_batch_size']} (trigger {TRIGGER}, copy chain at {GPU_TH})")
        return orig(self, target_model_config, target_parallel_config)

    cls.create_speculative_config = create_speculative_config


@register("suffix", "vllm.v1.worker.gpu.cudagraph_utils")
def patch_cudagraph(module) -> None:
    """Capture the wide uniform-decode width only for one request (it is never scheduled for more)."""
    cls = module.CudaGraphManager if hasattr(module, "CudaGraphManager") else None
    if cls is None or not hasattr(cls, "_init_candidates"):
        _log("cudagraph manager not found; capturing every width the schedule implies")
        return
    orig = cls._init_candidates

    def _init_candidates(self) -> None:
        orig(self)
        wide = 1 + K_WIDE
        if getattr(self, "decode_query_len", None) != wide:
            return  # the drafter's own manager (query length 5) or a runner without the wide width
        dropped = 0
        for mode, descs in list(self._capture_descs.items()):
            keep = [d for d in descs if not (d.uniform_token_count == wide and (d.num_reqs or 0) > 1)]
            dropped += len(descs) - len(keep)
            self._capture_descs[mode] = keep
        for key, cands in list(self._candidates.items()):
            self._candidates[key] = [d for d in cands
                                     if not (d.uniform_token_count == wide and (d.num_reqs or 0) > 1)]
        if dropped:
            _log(f"skipping {dropped} multi-request {wide}-row decode graphs")
        if DYNK:
            add_single_request_decode_graph(self, module, 1 + DSPARK_K)

    def capture(self, *args, **kwargs):
        out = orig_capture(self, *args, **kwargs)
        if getattr(self, "decode_query_len", None) == 1 + K_WIDE:
            rows = sorted(d.uniform_token_count for d in self.graphs
                          if d.num_reqs == 1 and d.uniform_token_count and d.num_active_loras == 0)
            _log(f"captured FULL 1-request decode graphs: {', '.join(f'1 x {r}' for r in rows) or 'none'}")
        return out

    orig_capture = cls.capture
    cls._init_candidates = _init_candidates
    cls.capture = capture


def add_single_request_decode_graph(mgr, module, rows: int) -> None:
    """Add a FULL decode graph for exactly one request of `rows` tokens to a CudaGraphManager's capture list and to
    the dispatch candidates of key (rows, loras), ahead of any PIECEWISE candidate. Other keys never match it: a
    one-request batch of uniform `rows` tokens dispatches with num_tokens == rows."""
    mode = mgr.cudagraph_mode.decode_mode() if mgr.cudagraph_mode else None
    full = module.CUDAGraphMode.FULL
    if mode != full or not mgr.cudagraph_mode.separate_routine() or mgr.varlen_decode:
        _log(f"no separate FULL uniform-decode routine (mode {mgr.cudagraph_mode}); 1 x {rows} graph not added")
        return
    if rows > mgr.compilation_config.max_cudagraph_capture_size:
        _log(f"1 x {rows} exceeds the capture limit; not added")
        return
    added = 0
    for loras in mgr.lora_capture_cases:
        desc = module.BatchExecutionDescriptor(cg_mode=full, num_tokens=rows, num_reqs=1, uniform_token_count=rows,
                                               num_active_loras=loras)
        descs = mgr._capture_descs.setdefault(full, [])
        if desc in descs:
            continue
        descs.append(desc)
        descs.sort(key=lambda d: d.num_tokens, reverse=True)
        cands = mgr._candidates.setdefault((rows, loras), [])
        first_other = next((i for i, d in enumerate(cands) if d.cg_mode != full), len(cands))
        cands.insert(first_other, desc)
        added += 1
    _log(f"dynamic K: {'added' if added else 'already had'} the 1 x {rows} FULL decode graph")


@register("suffix", "vllm.v1.worker.gpu.model_runner")
def patch_runner(module) -> None:
    """Remember the runner so the drafter can read the request token history (all_token_ids, total_len)."""
    cls = module.GPUModelRunner
    orig = cls.__init__

    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        STATE["runner"] = self

    cls.__init__ = __init__


def _dspark_view(vllm_config):
    """The same config with num_speculative_tokens = DSpark's trained block, for the drafter's own buffers."""
    import copy

    cfg = copy.copy(vllm_config)
    spec = copy.copy(vllm_config.speculative_config)
    object.__setattr__(spec, "num_speculative_tokens", DSPARK_K)
    object.__setattr__(cfg, "speculative_config", spec)
    return cfg


@register("suffix", "vllm.v1.worker.gpu.spec_decode.dspark.speculator")
def patch_dspark(module) -> None:
    """DSpark keeps its 5-token block; its draft rows grow to K; propose() appends the copy chain at batch 1."""
    import torch

    cls = module.DSparkSpeculator
    orig_init = cls.__init__
    orig_propose = cls.propose
    orig_icm = cls.init_cudagraph_manager
    orig_capture = cls.capture
    orig_ctx_rows = cls._num_graph_context_tokens

    def _short_attrs(self) -> dict:
        """The K_LO block's values of the attributes propose() / capture() read (W6_DYNK_BLOCK5)."""
        sa = getattr(self, "_w6_short_attrs", None)
        if sa is None:
            lo, dev = DYNK_LO, self.draft_tokens.device
            nq = lo if self.sample_from_anchor else 1 + lo
            sa = {"num_speculative_steps": lo, "num_query_per_req": nq,
                  "sample_col": torch.arange(lo, dtype=torch.int32, device=dev).repeat(self.max_num_reqs),
                  "_anchor_idx": torch.arange(self.max_num_reqs, dtype=torch.int64, device=dev) * nq,
                  "_step_cols": torch.arange(lo, dtype=torch.int32, device=dev),
                  "query_cudagraph_manager": self._w6_short_mgr}
            loop = getattr(self, "_w6_markov", None)
            if loop is not None:  # the fused Markov loop for a K_LO block (own slots / counters, same weights)
                sa["_w6_markov"] = type(loop)(loop.w1, loop.w2, self.max_num_reqs, lo)
            self._w6_short_attrs = sa
        return sa

    class _Short:
        """Swap the speculator into its K_LO block for one propose() or capture()."""

        def __init__(self, spec):
            self.spec, self.saved = spec, {}

        def __enter__(self):
            for k, v in _short_attrs(self.spec).items():
                self.saved[k] = getattr(self.spec, k)
                setattr(self.spec, k, v)
            return self

        def __exit__(self, *exc):
            for k, v in self.saved.items():
                setattr(self.spec, k, v)
            return False

    def init_cudagraph_manager(self, cudagraph_mode):
        orig_icm(self, cudagraph_mode)
        self._w6_short_mgr = None
        if not DYNK_BLOCK5 or getattr(self, "_w9", None) is None or self.num_speculative_steps <= DYNK_LO:
            return
        main = self.query_cudagraph_manager
        if main is None or not main.cudagraph_mode:
            _log("W6_DYNK_BLOCK5: the drafter runs without CUDA graphs; K_LO blocks off")
            return
        from vllm.config.compilation import CUDAGraphMode
        from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
        from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager

        nq = DYNK_LO if self.sample_from_anchor else 1 + DYNK_LO
        mgr = DFlashCudaGraphManager(self.vllm_config, self.device, main.cudagraph_mode, decode_query_len=nq)
        own = [d for d in mgr._candidates.get((nq, 0), []) if d.cg_mode == CUDAGraphMode.FULL and d.num_reqs == 1]
        desc = own[0] if own else BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=nq, num_reqs=1,
                                                            uniform_token_count=nq)
        mgr._capture_descs = {CUDAGraphMode.FULL: [desc]}  # one graph: 1 request x K_LO query tokens
        mgr._candidates = {(nq, 0): [desc]}
        self._w6_short_mgr = mgr
        self._w6_full_steps = self.num_speculative_steps

    def capture(self):
        orig_capture(self)
        if getattr(self, "_w6_short_mgr", None) is not None:
            with _Short(self):
                orig_capture(self)
            _log(f"W6_DYNK_BLOCK5: captured the 1 x {self._w6_short_attrs['num_query_per_req']} drafter graph for "
                 f"K_LO steps (Markov loop {'fused' if '_w6_markov' in self._w6_short_attrs else 'stock / W5'})")

    def _num_graph_context_tokens(self, num_reqs):
        full = getattr(self, "_w6_full_steps", None)
        if full is None:
            return orig_ctx_rows(self, num_reqs)
        # both block variants store K_HI + 1 context rows (the previous verify can have been K_HI)
        return min(num_reqs * (full + 1), self.max_num_tokens)

    def _propose_short(self, args, kwargs):
        """One K_LO-block propose; with W6_BLOCK5_CHECK the full block runs first and drafts 1..K_LO and their q
        rows are compared on the device."""
        lo = DYNK_LO
        if BLOCK5_CHECK:
            out = orig_propose(self, *args, **kwargs)
            d_full = self.draft_tokens[0, :lo].clone()
            q_full = self.draft_logits[0, :lo].clone() if self.draft_logits is not None else None
        with _Short(self):
            out = orig_propose(self, *args, **kwargs)
        STATE["short_runs"] += 1
        if BLOCK5_CHECK:
            chk = STATE["block5_chk"]
            if chk is None:
                chk = STATE["block5_chk"] = torch.zeros(3, dtype=torch.int64, device=self.draft_tokens.device)
            chk[0] += 1
            chk[1] += (self.draft_tokens[0, :lo] != d_full).any().to(torch.int64)
            if q_full is not None:
                chk[2] += (self.draft_logits[0, :lo].view(torch.int16) != q_full.view(torch.int16)).any().to(torch.int64)
            if STATE["short_runs"] % 512 == 0:
                _log(f"W6_BLOCK5_CHECK: K_LO block vs full block (steps, draft mismatches, q mismatches) "
                     f"{chk.tolist()}")
        return out

    def __init__(self, vllm_config, device):
        k_total = vllm_config.speculative_config.num_speculative_tokens
        self._w9_k = k_total
        if k_total <= DSPARK_K:
            orig_init(self, vllm_config, device)
            self._w9 = None
            _log(f"num_speculative_tokens {k_total} <= {DSPARK_K}: stock DSpark")
            return
        # The drafter (its buffers, CUDA graphs, attention builders and model) sees DSpark's 5-token block.
        orig_init(self, _dspark_view(vllm_config), device)
        # Rows the runner and the rejection sampler index: [max_num_reqs, K]. DSpark writes columns 0..4.
        self.draft_tokens = torch.zeros(self.max_num_reqs, k_total, dtype=torch.int64, device=device)
        self.draft_token_confidence_probs = torch.empty_like(self.draft_tokens, dtype=torch.float32)
        if self.draft_logits is not None:
            old = self.draft_logits
            new = torch.full((old.shape[0], k_total, old.shape[2]), float("-inf"), dtype=old.dtype, device=device)
            new[:, :DSPARK_K].copy_(old)
            self.draft_logits = new
            del old
        from w9_suffix_kernels import SuffixDrafter

        filler = int(getattr(self.draft_model_config.hf_config, "dspark_noise_token_id", 0))
        self._w9 = SuffixDrafter(self.max_num_reqs, k_total, DSPARK_K, filler, self.max_model_len, device)
        STATE["drafter"] = self._w9
        nbytes = 0 if self.draft_logits is None else self.draft_logits.numel() * self.draft_logits.element_size()
        _log(f"DSpark block {DSPARK_K}, draft rows {k_total}, draft-logits cache {nbytes:,} B")

    def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
                num_sampled, num_rejected, last_sampled, next_prefill_tokens, temperature, seeds,
                dp_sync=None, dummy_run=False, skip_attn_for_dummy_run=False, mm_inputs=None, is_profile=False):
        w9 = getattr(self, "_w9", None)
        if w9 is not None and self.draft_logits is not None:
            w9.reset_onehot(self.draft_logits)
        runner = STATE["runner"]
        mirror_done = None
        if (w9 is not None and not dummy_run and not is_profile and runner is not None and input_batch.num_reqs == 1
                and MIRROR_SIDE and not torch.cuda.is_current_stream_capturing()):
            rs = runner.req_states
            slot = int(input_batch.idx_mapping_np[0])
            # last index the history can hold now: decode steps are bounded by the sequence-length upper bound, a
            # (chunked) prefill by the prompt already written to all_token_ids
            ub = max(int(input_batch.seq_lens_cpu_upper_bound[0]), int(rs.prefill_len.np[slot]) - 1)
            # the history window copy runs on a side stream under the drafter graph; joined before draft()
            mirror_done = w9.sync_mirror_async(rs.all_token_ids.gpu, input_batch.req_ids[0], slot, ub)
        loop = getattr(self, "_w6_markov", None)
        if DYN is not None and DYNK_EARLY and loop is not None and hasattr(loop, "set_active") and not dummy_run \
                and not is_profile and not torch.cuda.is_current_stream_capturing():
            early_decide(loop, w9, list(input_batch.req_ids[: input_batch.num_reqs]),
                         int(input_batch.idx_mapping_np[0]) if input_batch.num_reqs == 1 else None)
        short = (DYNK_BLOCK5 and getattr(self, "_w6_short_mgr", None) is not None and not dummy_run
                 and not is_profile and input_batch.num_reqs == 1 and STATE["early"] is not None
                 and STATE["early"][1] == DYNK_LO and not torch.cuda.is_current_stream_capturing())
        try:
            args = (input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states, num_sampled,
                    num_rejected, last_sampled, next_prefill_tokens, temperature, seeds)
            kwargs = dict(dp_sync=dp_sync, dummy_run=dummy_run, skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                          mm_inputs=mm_inputs, is_profile=is_profile)
            out = _propose_short(self, args, kwargs) if short else orig_propose(self, *args, **kwargs)
        finally:
            if mirror_done is not None:
                torch.cuda.current_stream().wait_event(mirror_done)
        if w9 is None or dummy_run or is_profile or runner is None:
            return out
        if input_batch.num_reqs != 1:
            w9.invalidate()
            return out
        rs = runner.req_states
        slot = int(input_batch.idx_mapping_np[0])
        ub = max(int(input_batch.seq_lens_cpu_upper_bound[0]), int(rs.prefill_len.np[slot]) - 1)
        if mirror_done is None:
            w9.sync_mirror(rs.all_token_ids.gpu, input_batch.req_ids[0], slot, ub)
        w9.draft(w9.mirror, w9.mirror_slot0, rs.total_len.gpu, input_batch.idx_mapping, self.draft_tokens[0],
                 self.draft_logits, ub + self._w9_k, GPU_TH)
        if LOG_EVERY and int(w9.stats_host[0]) // LOG_EVERY != STATE.get("logged", -1):
            STATE["logged"] = int(w9.stats_host[0]) // LOG_EVERY
            s = w9.stats_host.tolist()
            _log(f"batch-1 proposals {s[0]}, copy chains {s[1]}, extensions {s[2]}, mean match "
                 f"{s[3] / max(1, s[0]):.2f}; scheduled wide {STATE['wide_steps']}, normal {STATE['normal_steps']}")
        return out

    cls.__init__ = __init__
    cls.propose = propose
    if DYNK_BLOCK5:
        cls.init_cudagraph_manager = init_cudagraph_manager
        cls.capture = capture
        cls._num_graph_context_tokens = _num_graph_context_tokens


def early_decide(loop, w9, req_ids: list, slot: int | None) -> None:
    """W6_DYNK_EARLY, in propose() before the drafter runs: decide the batch-1 request's next width and set the fused
    Markov loop to that many positions (every position when no decision is made: batch > 1, a wide step may follow,
    early decisions off)."""
    n_run = loop.num_steps
    STATE["early"] = None
    STATE["proposed"] = frozenset(req_ids)
    STATE["early_capable"] = True
    if STATE["early_on"] and len(req_ids) == 1 and w9 is not None and slot is not None:
        if not (TRIGGER > 0 and int(w9.lflag_host[slot]) >= TRIGGER):  # a wide step may follow: full loop
            k = DYN.choose(req_ids[0])
            STATE["early"] = (req_ids[0], k)
            n_run = k
    loop.set_active(n_run)


def batch1_width(sched) -> int | None:
    """Drafts for the single running request's next step, or None when the step is not a batch-1 decode."""
    if len(sched.running) != 1 or len(sched.waiting) or len(getattr(sched, "skipped_waiting", ())):
        return None
    req = sched.running[0]
    if not req.spec_token_ids or req.is_prefill_chunk or getattr(req, "use_structured_output", False):
        return None
    early = None
    if DYN is not None and DYNK_EARLY and STATE["early_on"]:
        e = STATE["early"]
        if e is not None and e[0] == req.request_id:
            early = e[1]
            STATE["early_used"] += 1
            if early < DSPARK_K:  # the drafter ran only `early` Markov positions: verify exactly those
                STATE["early_lo"] += 1
                return early
        elif STATE.get("early_capable") and (STATE["proposed"] is None or req.request_id not in STATE["proposed"]):
            # a batch-1 decode whose drafts no propose() since the previous schedule produced: another engine
            # ordering, where a decision could pair with the wrong step
            STATE["early_on"] = False
            _log("W6_DYNK_EARLY: a batch-1 decode was scheduled before the propose() that drafts it; early "
                 "decisions off for this process (the Markov loop runs every position)")
    drafter, runner = STATE["drafter"], STATE["runner"]
    if drafter is not None and runner is not None and TRIGGER > 0:
        slot = runner.req_states.req_id_to_index.get(req.request_id)
        if slot is not None and int(drafter.lflag_host[slot]) >= TRIGGER:
            return K_WIDE
    if early is not None:
        return early
    if DYN is not None and DYNK_EARLY and STATE["early_on"]:
        STATE["early_fallback"] += 1
    return DYN.choose(req.request_id) if DYN is not None else NORMAL_K


def batch1_step_k(scheduler_output) -> int | None:
    """Drafts verified by a finished step if it was a one-request decode step (else None)."""
    nst = scheduler_output.num_scheduled_tokens
    if len(nst) != 1:
        return None
    (rid, n), = nst.items()
    spec = scheduler_output.scheduled_spec_decode_tokens.get(rid)
    if not spec or n != len(spec) + 1:
        return None
    return len(spec)


@register("suffix", "vllm.v1.core.sched.scheduler")
def patch_scheduler(module) -> None:
    """Before each schedule(): the single decoding request gets 5 or K placeholder drafts; others never exceed 5."""
    cls = module.Scheduler
    orig = cls.schedule

    def schedule(self, *args, **kwargs):
        if K_WIDE > DSPARK_K:
            k = batch1_width(self)
            STATE["early"] = STATE["proposed"] = None  # a propose decision applies to the next schedule only
            if k is not None:
                if len(self.running[0].spec_token_ids) != k:
                    self.running[0].spec_token_ids = [-1] * k
                STATE["wide_steps" if k == K_WIDE else "normal_steps"] += 1
            else:
                for req in self.running:
                    if len(req.spec_token_ids) > NORMAL_K:
                        req.spec_token_ids = [-1] * NORMAL_K
        return orig(self, *args, **kwargs)

    cls.schedule = schedule
    if DYN is None:
        return
    orig_stats = cls.make_spec_decoding_stats
    orig_update = cls.update_from_output

    def make_spec_decoding_stats(self, spec_decoding_stats, num_draft_tokens, num_accepted_tokens,
                                 num_invalid_spec_tokens, request_id):
        DYN.observe(request_id, int(num_draft_tokens), int(num_accepted_tokens))
        return orig_stats(self, spec_decoding_stats, num_draft_tokens, num_accepted_tokens, num_invalid_spec_tokens,
                          request_id)

    def update_from_output(self, scheduler_output, model_runner_output):
        DYN.observe_interval(batch1_step_k(scheduler_output), time.perf_counter())
        out = orig_update(self, scheduler_output, model_runner_output)
        STATE["updates"] = STATE.get("updates", 0) + 1
        if STATE["updates"] % 1024 == 0 and len(DYN.reqs) > 64:
            DYN.prune(self.requests)
        n = sum(DYN.chosen.values())
        if LOG_EVERY and n and n % 4096 == 0 and n != STATE.get("dynk_logged"):
            STATE["dynk_logged"] = n
            _log(DYN.summary())
        return out

    cls.make_spec_decoding_stats = make_spec_decoding_stats
    cls.update_from_output = update_from_output
    _log(f"dynamic K v3 on: batch-1 normal steps verify {DYN.lo} or {DYN.hi} drafts (cost "
         f"{'auto, fallback ' if DYN.auto else ''}{DYN.fixed_ratio}, decay {DYN.decay}, n0 {DYN.n0}, prior "
         f"{DYNK_PRIOR}, explore up every {DYN.explore} / down every {DYN.explore_lo}, optimism {DYN.optimism}, "
         f"cost = median of 64){'; W6_DYNK_EARLY: decided in propose(), short Markov loop on K_LO steps' if DYNK_EARLY else ''}"
         f"{'; W6_DYNK_BLOCK5: K_LO steps draft a K_LO block' if DYNK_BLOCK5 else ''}"
         f"{' (W6_BLOCK5_CHECK: full block too, compared)' if BLOCK5_CHECK else ''}")


@register("suffix", "vllm.v1.worker.gpu.states")
def patch_states(module) -> None:
    """A new request in a slot starts in normal mode (no stale copy-mode flag from the slot's previous owner)."""
    cls = module.RequestState
    orig = cls.add_request

    def add_request(self, req_id, *args, **kwargs):
        orig(self, req_id, *args, **kwargs)
        drafter = STATE["drafter"]
        if drafter is not None:
            drafter.lflag_host[self.req_id_to_index[req_id]] = 0

    cls.add_request = add_request
