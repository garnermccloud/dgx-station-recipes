"""tail_drafthead_*: the DSpark drafter's base logits from a quantized full-vocabulary copy of the LM head (W5).

The drafter shares the target's BF16 lm_head (1.32 GB read per step for 5 rows, ~208 us in the prod trace). These
components give the drafter its own quantized copy, built in load_draft_model (before KV-cache profiling), and replace
only DSparkDeepseekV4ForCausalLM.compute_draft_logits; the target head is untouched and every vocabulary row stays a
candidate (unlike tail_fr*, nothing is truncated). GEMM: DeepGEMM's sm100 block-scaled fp8_fp4_gemm_nt (w5_dghead).

  component                 weights        activations     extra HBM        head time (graph, L2 cold, M=5)
  tail_drafthead_mxfp8      MXFP8 (e4m3)   MXFP8 2-term    682,598,400 B    138 us
  (production cuBLAS BF16 head on the same bench: 222 us)

Drift on 320 real drafter rows with production's Markov bias (bench/qhead_drift.py), KL(q_bf16 || q) at T = 1:
mxfp8 2-term 2.15e-3 (= BF16 activations); the BF16 head itself is 3.3e-4 from fp64.

The logits feed production's Markov loop (or tail_markov) unchanged, so q is softmax(base_q + markov bias) and the
q cache the rejection sampler reads holds exactly the logits sampled from: lossless output, DRAFT-ONLY change
(acceptance only).
"""
from __future__ import annotations

import sys

try:
    from fastpath_hook import register
except ImportError:  # imported directly by tests
    def register(component, target):
        return lambda fn: fn

SPECULATOR_MODULE = "vllm.v1.worker.gpu.spec_decode.dspark.speculator"
NAME = "tail_drafthead_mxfp8"
FMT = "mxfp8"
TWO_TERM = True


def install_head(spec, model):
    """Build the quantized head for `model` (a loaded full-vocab DSpark drafter) and route compute_draft_logits to it.
    Raises when the drafter has a reduced vocabulary: the shipped recipe requires the full-vocab head."""
    import types

    from w5_dghead import DGHead

    if getattr(model, "draft_id_to_target_id", None) is not None:
        raise RuntimeError(f"{NAME}: drafter has a reduced vocabulary; the shipped recipe requires full-vocab")
    head = DGHead(model.lm_head.weight, FMT, two_term=TWO_TERM)
    max_m = int(spec.max_num_reqs) * int(spec.num_speculative_steps)
    head.warmup(max_m)  # DeepGEMM JIT for every draft width, here and never inside a graph capture

    def compute_draft_logits(m, hidden_states):
        return head(m.model.norm(hidden_states))

    model._w5_head = head
    model.compute_draft_logits = types.MethodType(compute_draft_logits, model)
    sys.stderr.write(f"DS41_FASTPATH {NAME}: draft head {FMT} x MXFP8 2-term activations "
                     f"(DeepGEMM), extra HBM {head.nbytes:,} B (allocated in load_draft_model), warmed M=1..{max_m}\n")
    return head


def patch(module) -> None:
    cls = module.DSparkSpeculator
    if getattr(cls, "_w5_drafthead", None) is not None:
        raise RuntimeError(f"{NAME}: {cls._w5_drafthead} is already installed; install one tail_drafthead_* component")
    orig_load = cls.load_draft_model

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = orig_load(self, target_model, target_attn_layer_names)
        install_head(self, model)
        return model

    cls.load_draft_model = load_draft_model
    cls._w5_drafthead = NAME


patch = register(NAME, SPECULATOR_MODULE)(patch)
