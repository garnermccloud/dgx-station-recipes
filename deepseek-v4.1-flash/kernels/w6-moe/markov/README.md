# deliver-markov-v9 + deliver-w9-dynk-v4: the Markov loop skips the draft positions a K5 step never verifies (W6, 2026-10-21)

dynk picks K5 on 50-68% of batch-1 steps (ab-rebal-1021), but DSpark drafts its 7-block every step. The fused Markov
loop runs 7 sequential positions (87.6 us in the ab-rebal trace: prologue 5.4 + step 0 ~19 + 6 x ~9.5), and on a K5
step positions 5 and 6 are thrown away. This pair runs only the positions the next verify uses: about -19 us of
drafter time on every K5 step. Drafts 1..5 and their q-cache rows are bit-identical to the full loop's; target outputs
are unaffected.

## Pieces (each armable on its own)
- **Markov v9** (`w6mk.cu` ABI 4; runtime wrapper `../fastpath/w6_markov.py`; `fastpath_tail_markov_fused.py` unchanged from v5):
  - `Args.n_active`: a device int the kernel reads at start; it runs steps `0..n-1`, clamped to 1..S.
  - Steps n..S-1 leave their draft tokens and q-cache rows untouched.
  - `FusedMarkovLoop.set_active(n)` changes it in stream order with one fill kernel, and only when the value changes.
    CUDA-graph replays follow it because the graph holds the pointer.
  - With nobody calling `set_active`, n stays S, which is exactly v5.
- **suffix v4** (`fastpath_w9_suffix.py` = dynk v3 + early decisions, fixed on in the shipped recipe):
  - A batch-1 propose() decides the next step's width (`DYN.choose`) before the drafter graph
    runs, and sets the Markov loop to that many positions.
  - schedule() then applies that decision instead of deciding itself.

## Why the short loop's stale positions are never verified
- **Ordering:** in the uniproc engine, `sample_tokens(N)`, and with it `propose(N)`, runs synchronously before
  `schedule(N+1)` (`step_with_batch_queue`).
- **Lifetime:** a decision applies to the next schedule only; each schedule clears it.
- **No wide step after a short loop:** a K5 decision returns 5 even if the copy-mode flag has crossed W9_TRIGGER by
  then. A wide chain can contain DSpark's positions 5..6.
- **Full loop when a wide step may follow:** if propose sees the flag at or above W9_TRIGGER, it makes no decision and
  runs the full loop.
- **Batch and mode changes only narrow:** batch >= 2, waiting requests, prefill chunks and structured output all trim
  to W9_NORMAL_K. W6_DYNK_EARLY refuses to turn on unless W9_NORMAL_K <= W9_DYNK_LO.
- **Ordering guard:** if a batch-1 decode is ever scheduled before the propose that drafts it (another engine
  ordering), early decisions switch off for the process, with one log line, and every later loop runs every position.
- **Cost:** the decision sees acceptance one step older than schedule() did. update_from_output(N-1) runs after
  sample_tokens(N).

## Gates
- `test_early_exit.py` (GB300):
  - one CUDA graph of the loop, replayed at n = 7, 5, 7, 6, 5, 7;
  - each replay's drafts and q rows 0..n-1 must be bitwise the full run's, and rows n..6 must keep a sentinel;
  - an eager launch at n = 5 is checked too;
  - it also measures GPU time at n = 7 / 6 / 5.
- `test_markov_fused.py ../ref 24`: v5's exactness test, now on v9 with n = S.
- `dynk/v4/test_early.py` (CPU, production image):
  - drives the component's real `early_decide` and the wrapped `Scheduler.schedule` through 20000 random uniproc steps
    (copy-mode flags around W9_TRIGGER, batch 1-2, waiting requests);
  - zero verified DSpark positions the loop skipped;
  - the reversed ordering switches early decisions off;
  - a mutant that lets a wide step follow a short loop fails it (426 violations).
- `dynk/v4/test_dynk.py`: v3's suite, PASS with and without W6_DYNK_EARLY.

## Install
Install `../fastpath/w6_markov.py` + `libw6mk.so` together (ABI 4 refuses the v5 pair), plus
`fastpath_w9_suffix.py` v4. The shipped recipe has early decisions and dynamic K on; there is no env switch.

Engagement lines in ds41.log:
- `dynamic K v3 on: ...; W6_DYNK_EARLY: decided in propose(), short Markov loop on K_LO steps`;
- the periodic dynk summary ends with `early decisions N (K_LO M, scheduler fallbacks F, on True)`;
- the line `early decisions off for this process` must NOT appear.

In the trace, `markov_kernel` becomes bimodal: about 68 us on K5 steps and about 88 us on K7 steps.
