# peer tier v3, d1 build (W4 COPY_AHEAD)

This tree is production's peer3 (the v4 RTX kernel) plus one addition for the GB300's copy-ahead (W4
deliver-copy-d1, `W4_BOUNDARY_COPY_AHEAD=1`). The protocol version is unchanged, and a GB300 without COPY_AHEAD
ignores the new words.
- `csrc/peer3_proto.cuh`:
  - `kOffReady` (1152): a u32 host ready word. The w4 router arms it to 0 before a call's records.
  - `kOffCap` (1160) = `kCapReady` (`0x52454459`): the capability.
- `csrc/peer3_rtx_kernel.cuh`: after every done-tag store, the RTX also stores ready = 1 (`st.release.sys`).
- `csrc/peer3_rtx.cu`: exports `OFF_READY` / `OFF_CAP` / `CAP_READY`.
- `sidecar/peer_server3.py`: at segment init, writes ready = 0 and the capability.
- `prebuilt/peer3_rtx_sm120.so`: rebuilt from these sources (PEER3_ARCH=120 JIT build in the production image).
  `prebuilt/peer3_gb300_sm103.so` is production's, unchanged.
- This directory is already at `fastpath/peer3/`; restart the sidecar after replacing it.

# peer tier v3: the cold-expert sidecar without host round trips

v3 replaces the decode-sized (T <= 64) path of the MegaMoE hook's peer tier v2. The v2 path is the sidecar's
Python poll loop, a CUDA graph replay per call, memcpy nodes and a sync: about 55 us of fixed cost per layer call.
In v3, one persistent cooperative kernel on the RTX PRO 6000 polls the host-mapped segment itself, reads b12x's
prepared FP4 weights in place, and writes the cold rows straight back. Prefill-sized calls (T > 64) keep using v2
(b12x) through the same sidecar process.

## Files
| path | what |
|---|---|
| `csrc/peer3_proto.cuh` | segment layout, tag format, PTX helpers |
| `csrc/peer3_gb300.cuh`, `.cu` | GB300 device stages and standalone graph-safe kernels |
| `csrc/peer3_rtx.cuh`, `peer3_rtx_kernel.cuh`, `peer3_rtx.cu` | the RTX persistent kernel and its launcher |
| `py/peer3_ext.py`, `py/peer3_seg.py` | extension loader (prebuilt .so or JIT) and the shared segment |
| `py/peer3_stats.py` | per-call statistics: the sidecar's fold of the stats ring, and a read-only `live` / `show` CLI |
| `sidecar/peer_server3.py` | the sidecar: upstream peer_server2's b12x v2 server plus the v3 kernel |
| `../fastpath/fastpath_peer3.py` | fastpath component `peer3` for the GB300 |
| `prebuilt/` | `peer3_gb300_sm103.so`, `peer3_rtx_sm120.so`, built in the production image |
| `tests/` | exactness and protocol tests, benchmarks |

`csrc/peer3_gb300.cuh` contents:
- **For W4 / W8 to call directly:** `split_routes`, `send_token_warp` / `send_token_cta`, `wait_done`, `add_row`, `advance`;
  statistics only: `post_route_ids`, `post_prev_wait`.
- **Standalone kernels:** `send` (grid T) and `finish` (grid T, optionally a PDL dependent of MegaMoE).

## Protocol (segment `/dev/shm/vllm_peer_tier3`, ~1 MB, pinned and mapped in both processes)
**GB300 send, per token:**
- writes the token's MXFP8 row (quantizer bit-identical to `route_send2`) and its route weights;
- then publishes `word0 = tag16 | 6 cold ids x 8 bit` with a system-scope release;
- `tag = counter4 | layer6 | (T - 1)6`.

**RTX:**
- CTA 0 polls slot 0, reads the T words and lists the cold rows, then releases the call to the grid.
- Every CTA builds the same plan from the T words, in a warp-parallel and deterministic pass.
- Tail CTAs stage the MXFP8 rows.
- FC1 items are (expert, 256-channel tile pair, K slice). FC2 items are (expert, 256-column tile, K slice); they run from the far end of the grid, so CTAs without FC1 work stream FC2 at once.
- Weights are whole 16 KB tiles of b12x's prepared layout, loaded with `cp.async.bulk` (CTA-local form) and an mbarrier ring.
- Math is SM120 block-scaled QMMA (`mma.sync kind::mxf8f6f4.block_scale` m16n8k32, e2m1 x e4m3, UE8M0 per 32, fp32 accumulate).
- The last K slice of a tile pair runs MegaMoE's SwiGLU and MXFP8 epilogue.
- The last item of a column tile sums its K slices in order, rounds each (token, slot) to bf16, and adds the token's cold slots in top-k order in fp32 (MegaMoE's combine). It writes bf16 rows to the segment.
- The 20 column finalizers count down; the last one releases `done = tag`.

**GB300 finish:** waits for `done == tag` (always, once per call) and adds the rows. With `PEER3_PDL=1` (default)
it is launched as a programmatic dependent of MegaMoE, so its wait and its host reads come before
`griddepcontrol.wait` (measured: -1.0 us per layer call). A parallel placement (the wait and row copy on a side branch
during a 136-SM MegaMoE, then a device-local add) measured 0-4 us slower and was dropped: the finish's cost is its
launch, not its host reads. The next saving is fusing the finish into the following kernel (`wait_done` + `add_row`).

**v3.1 additions (protocol version 2; both halves must come from the same build):**
- The sidecar stamps `kProtoVersion` at segment offset 64. The GB300 waits up to `PEER3_VERSION_WAIT_S` (120 s) for
  it at its first call and fails the process on a mismatch.
- Stats ring (32768 x 32 B after the output rows; the segment is now 2,056,192 B). After acknowledging a call, CTA 0
  of the RTX kernel writes one record: tag, rows, pairs, distinct cold experts, distinct hot experts (from the global
  ids the send writes at record + 32, tagged), RTX busy time, and the GB300's wait for its previous call (forwarded
  by the next send through record 0 + 48). The sidecar folds the ring every second and writes per-layer and per-T
  histograms to `PEER3_STATS_FILE` (default `peer3_stats.json` beside `peer_stats.json`) every 10 s;
  `python3 py/peer3_stats.py show [file]` prints them, `... live [segment] [seconds]` folds the live segment read-only.
- The send's per-token publish is one `st.release.sys` (the extra `fence.acq_rel.sys` before it was redundant).
- Idle parking: the persistent kernel exits after `PEER3_IDLE_S` (default 2) without a call and the sidecar relaunches
  it when the next call is pending (~0.1 ms once), so an idle sidecar no longer spins the RTX (~180 W).

**v3.2 (RTX side only; same protocol version 2, GB300 binary unchanged): next-layer L2 prefetch, `PEER3_PREFETCH_K`**
(sidecar env, 0..8, default 0 = off). Layer calls arrive in order and the RTX idles between them. After each call
with work, once its done is out, CTAs 1..G-1 bulk-prefetch (`cp.async.bulk.prefetch.L2`) the next layer's k most
popular cold experts into the 128 MB L2; CTA 0 learns popularity online (calls touching each cold expert, per layer,
halved every 1024 calls) and keeps the top 8. It stops early if the next call has already been released; results
are unchanged. Measured (T=6, popularity-skewed picks, next layer's weights dropped from L2 before each prefetch):
2 / 4 / 6 distinct experts 48.6 / 68.0 / 93.1 us -> 41.3 / 49.5 / 79.4 us with k=4.

**Determinism:**
- No atomics touch data. Every reduction has a fixed order. The plan and the K splits depend only on the call and the grid size, which is always all 188 SMs.
- Output is bit-identical across replays and relaunches.

## Exactness
**v3 against the fp64 reference of the MegaMoE/checkpoint recipe** (`tests/ref.py`: bf16 gate/up, clamp 10, route weight before FC2, MXFP8 intermediate with MegaMoE's UE8M0 rule, bf16 per-slot outputs, fp32 top-k-order combine, bf16):
- 99.999% of bf16 elements are bit-equal. The rest are off by 1 bf16 ulp from fp32 accumulation order. Max abs is 2e-3 at T=64.
- Coverage: all cross-GPU shapes, T = 1..64, and the A2000 logic tests.

**The send:** bit-identical to production's `peer_fused2.route_send2` on 60 cases (layers 0/10/39, T = 1..64, int64 and int32 ids). This covers hot ids, flags, MXFP8 rows, cold ids and weights.

**Production b12x on the same inputs** (`../b12x-cmp`, production's race configs forced):
- Against the same fp64 recipe: 3.8% rms error, cosine 0.99929, 5% of bf16 elements equal.
- Replay-to-replay: M=1 is deterministic. At M>=2 it differs by 0.9% rms between replays, from unordered bf16 atomics.
- v3 removes this noise, so greedy token-identity checks between builds become possible.

## Measured (RTX PRO 6000 + GB300, cross-GPU)
Weights are streamed from DRAM: 16 rotating experts, a 300 MB working set larger than the RTX L2. Each figure is the GPU time of graph-replayed `[send, finish]` on the GB300 (publish, RTX work, acknowledgement, add).
- **v3 round trip:** about 23.5 + 11.9 us per distinct cold expert, from 1.6 TB/s weight streaming. 1 expert is 35.4 us, 2 is 50.8, 4 is 71.1.
- **b12x for comparison**, graph replay + sync only, without the poll, detect or ack overhead: 2 experts 71 us, 3 experts 85, 4 experts 97, 7 experts 132.

GB300 cost inside a serving graph:

| | per layer |
|---|---|
| peer3 send | 3.4-5.0 us (T=1..64) |
| production route_send2 | 5.5-6.1 us |
| send + finish with an instantly available acknowledgement | 9.7-12.4 us |

## Enable
`serve/serve.sh` does all of this.  For a manual setup:

1. `fastpath/peer3/` is in place and `fastpath/components.py` imports `fastpath_peer3`.
2. Remove both stale buffers: `rm -f /dev/shm/vllm_peer_tier2 /dev/shm/vllm_peer_tier3`.
3. Sidecar: `-v "$PWD/fastpath/peer3":/opt/ds41/peer3:ro -e PEER3_DIR=/opt/ds41/peer3
   -e PEER3_PREBUILT=/opt/ds41/peer3/prebuilt -e CUDA_MODULE_LOADING=EAGER
   -e VLLM_EXP_PEER3_SHM=/dev/shm/vllm_peer_tier3`, running
   `python3 /opt/ds41/peer3/sidecar/peer_server3.py --rowmap ... --model /model` instead of the v2 server.
4. vLLM: `DS41_FASTPATH=peer3[,...]`, `-e PEER3_PREBUILT=/fastpath/peer3/prebuilt
   -e VLLM_EXP_PEER3_SHM=/dev/shm/vllm_peer_tier3 -e W4_BOUNDARY_PEER3=1`, keep `MEGA_PEER=2` and the other
   hook flags; prefill still uses v2.  A/B flag (through `DS41_ENV`): `PEER3_PDL=0|1` (default 1).  Test mode:
   `PEER3_DETERMINISTIC_PREFILL=1` (T > 64 through v3 in 64-token chunks: slow, bit-reproducible greedy runs).
   Do **not** set `CUDA_MODULE_LOADING=EAGER` in the vLLM container; it loads every module at start-up.
5. Rollback: the stock sidecar command and `DS41_FASTPATH` without `peer3`.  `peer_server3.py` also serves a
   v2-only GB300 correctly.

## Rules for anything that runs a persistent kernel (learned here)
- **Module loads.** CUDA's lazy loading, and any runtime module load (Triton JIT, CuTe JIT), stalls behind a spinning kernel. So does a device-wide synchronize; torch's graph capture does one. Preload modules, capture graphs and compile kernels before starting the kernel, or stop it first. The sidecar stops the v3 kernel before any b12x or CUDA work.
- **Cluster-scoped PTX on sm_120.** `cp.async.bulk.shared::cluster` and `fence.mbarrier_init.release.cluster` made a plain launch reserve about 3.75 GB of device memory, or fail with out-of-memory. Use `.shared::cta` and `fence.proxy.async.shared::cta`.
- **No stack frame.** No printf and no dynamically indexed local arrays; check STACK:0 with `cuobjdump --dump-resource-usage`.
