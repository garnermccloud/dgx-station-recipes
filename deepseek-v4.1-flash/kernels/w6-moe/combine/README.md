# MegaMoE fork v11: combine-in-seam support (W6 with W4, 2026-10-21)

v11 is v5 plus `megamoe_c11_patch.py`. The fd gate shows v11 is bit-identical to v5 whenever skip_combine is 0 (the
default).

## What changes
A second control block, C11, sits at the fold control + 256 B, i.e. symm.buffer bytes 21009536..21009568:

| Field | Type | Byte offset | Notes |
|---|---|---|---|
| skip_combine | u32 | 21009536 | |
| magic | u32 | 21009540 | 0x31314336 ("6C11") once a call has run |
| slot_base | u64 | 21009544 | |
| slot_stride | u64 | 21009552 | |
| token_stride | u64 | 21009560 | |

Per call:
- Every CTA reads skip_combine before the epilogue grid sync.
- After the sync, SM 0 clears it to 0 (self-clearing) and exports the slot layout and the magic.
- With skip_combine = 1, single rank and the fold not armed, the kernel keeps the L2 epilogues, the grid sync, the
  combine-ready store and the cleanup, but skips the combine loop: y is not written.
- The slots then hold the call's outputs until the next call on this symm buffer.
- The producer stores 1 before the launch; on W4's router path, the router kernel does this.

## Host side
fastpath_megamoe.py adds:
- `c11_skip(symm)`
- `c11_layout(symm)`
- `combine_into(y, symm, T, layout)`, a ctypes wrapper around `libw6combine.so`.

The `w6_combine_launch` symbol, built from w6combine.cu, is the v5 lane-parallel combine as a standalone kernel, with
no fold.

## Gates (v11_checks.log)
- fd_check is 36/36 bit-identical to v5 at both E=24 and E=40.
- test_p3fold --check and the component test pass.
- test_c11 passes 19/19 widths (T 1..16, 24, 32, 64) on one symm buffer. For each:
  - a skipped call leaves y untouched;
  - the word self-clears;
  - the exported layout is correct (slot stride 64 x 10240, token stride 10240);
  - combine_into reproduces the in-kernel combine bitwise;
  - the next call is v5's y.
- The end-to-end gate through the new seam is W4's.
