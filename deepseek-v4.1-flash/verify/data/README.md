# verification data

- `gsm8k-200.jsonl` — the first 200 rows of RonanLabs' `gsm8k_test.jsonl` (Apache-2.0, Copyright 2026
  RonanLabs; see `LICENSE.ronanlabs` and `SOURCE.ronanlabs`).  The underlying GSM8K questions are from
  OpenAI's `grade-school-math` (MIT; see `LICENSE.gsm8k`).
- `coding_source.py` — the 1017 frontend overlay builder, used as the ~4.4K-token coding prompt.
- `fixed_prompt.txt` — the single-stream fixed prompt of `bench --full`: the coding task (`CODE_TASK`) from Fraser
  Price's BRRR bench, repeated ten times (~2.6K tokens), so single-stream results line up with that bench (Apache-2.0;
  see `LICENSE.brrr` and `SOURCE.brrr`).
- `LICENSE.gsm8k` — MIT licence for the GSM8K questions.
