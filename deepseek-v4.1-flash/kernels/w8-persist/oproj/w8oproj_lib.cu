// libw8oproj.so (oproj-sk2): plain C ABI of the W8 wo_a + MXFP8 quantize kernel (src/woa_kernel.cuh) for the
// "w8oproj" fastpath component (fastpath_w8oproj.py, ctypes; no torch, no JIT).
// (split, chains) = K parts per 128-row tile, accumulation chains per CTA (chains 0 = 1, or 2 for split 4):
//   (1, 1) production's chain (bitwise with DeepGEMM; the shipped v2.3 kernel), 64 CTAs
//   (2, 2) one CTA per tile, two 16-K-block chains on two issuer warps, z = p0 + p1, 64 CTAs   [in-CTA split-K]
//   (2, 1) cluster-2 per tile, one 16-K-block chain per CTA, z = s0 + s1 (DSMEM), 128 CTAs       [same sums as (2, 2)]
//   (4, 2) cluster-2 per tile, two 8-K-block chains per CTA, z = (p0 + p1) + (p2 + p3), 128 CTAs
// Splits 2 / 4 are the deterministic split-K of the exactness policy: fixed fp32 sum order, not bitwise.
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "woa_kernel.cuh"

namespace wa = w8::woa;

extern "C" struct W8oArgs {
  const void* x;       // fp8 attention output view [S][8][4096] (stride(2) == 1)
  long long x_s0, x_s1;  // its token / group strides in bytes
  const void* xs;      // int32 packed ue8m0 scales: word (t, g, kb) at g * xs_sg + kb * xs_skb + t
  long long xs_sg, xs_skb;
  const void* w;       // fp8 wo_a [8][1024][4096] contiguous
  const void* wsf;     // tile-major UTCCP atoms [64][32][512] (built at load)
  void* zq;            // out fp8 [S][8192]
  void* zsf;           // out FlashInfer 128x4 ue8m0 [ceil(S/128)*128 * 256] (pad rows zero)
  void* tl;            // optional u64 [128][4]
  void* ticket;        // split 2: u32 [2], zero at rest
  void* diag;
  long long S, split, pdl;
  long long prologue;  // woa::Params::prologue (bit 0: no L2-tail prefetch, ...)
  void* z;             // optional out bf16 [S][8192] (tests)
  void* z32;           // optional out fp32 [S][8192]: the accumulator before the bf16 round (error reports)
  long long chains;    // accumulation chains per CTA (0: 1, or 2 for split 4)
};

static PFN_cuTensorMapEncodeTiled_v12000 encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (fn == nullptr) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) == cudaSuccess &&
        q == cudaDriverEntryPointSuccess)
      fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}
static int map(CUtensorMap* m, int rank, const void* base, const uint64_t* dims, const uint64_t* strides,
               const uint32_t* box) {
  const uint32_t elem[5] = {1, 1, 1, 1, 1};
  if (encode() == nullptr) return 2;
  return encode()(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank, const_cast<void*>(base), dims, strides, box, elem,
                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS ? 0 : 2;
}

template <int S, int SPLIT, int C>
static int launch_w(const wa::Params& p, bool pdl, cudaStream_t stream) {
  auto k = wa::woa_kernel<S, SPLIT, C>;
  cudaLaunchAttribute at[2];
  int n = 0;
  if (pdl) {
    at[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[n++].val.programmaticStreamSerializationAllowed = 1;
  }
  if (SPLIT > 1) {
    at[n].id = cudaLaunchAttributeClusterDimension;
    at[n].val.clusterDim.x = SPLIT;
    at[n].val.clusterDim.y = 1;
    at[n++].val.clusterDim.z = 1;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(wa::kTiles * SPLIT);
  cfg.blockDim = dim3(w8::kThreads);
  cfg.dynamicSmemBytes = wa::kSmemMax;
  cfg.stream = stream;
  cfg.attrs = at;
  cfg.numAttrs = n;
  return cudaLaunchKernelEx(&cfg, k, p) == cudaSuccess ? 0 : 5;
}

template <int S, int SPLIT, int C>
static int prepare_w() {
  cudaFuncAttributes fa;
  if (cudaFuncGetAttributes(&fa, wa::woa_kernel<S, SPLIT, C>) != cudaSuccess) return 4;
  return cudaFuncSetAttribute(wa::woa_kernel<S, SPLIT, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, wa::kSmemMax) ==
                 cudaSuccess ? 0 : 4;
}

template <int S>
static int launch_split(const wa::Params& p, int split, int chains, bool pdl, cudaStream_t st) {
  if (split == 1 && chains == 1) return launch_w<S, 1, 1>(p, pdl, st);
  if (split == 2 && chains == 2) return launch_w<S, 1, 2>(p, pdl, st);
  if (split == 2 && chains == 1) return launch_w<S, 2, 1>(p, pdl, st);
  if (split == 4 && chains == 2) return launch_w<S, 2, 2>(p, pdl, st);
  return 1;
}

extern "C" {

// 0: sizeof(W8oArgs), 1: offsetof(S), 2: offsetof(diag), 3: supported splits (bit p: p K parts), 4: max width (every T
// in 1..max, every split), 5: offsetof(prologue), 6: offsetof(z32), 7: offsetof(chains)
long long w8o_abi(int which) {
  switch (which) {
    case 0: return sizeof(W8oArgs);
    case 1: return offsetof(W8oArgs, S);
    case 2: return offsetof(W8oArgs, diag);
    case 3: return (1 << 1) | (1 << 2);
    case 4: return 16;
    case 5: return offsetof(W8oArgs, prologue);
    case 6: return offsetof(W8oArgs, z32);
    case 7: return offsetof(W8oArgs, chains);
    default: return -1;
  }
}

#define W8O_WIDTHS(X) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11) X(12) X(13) X(14) X(15) X(16)

// Load every width's kernel and set its SMEM limit now (outside graph capture).
int w8o_prepare() {
  int r = 0;
#define W8O_PREP(W) r |= prepare_w<W, 1, 1>() | prepare_w<W, 2, 1>();
  W8O_WIDTHS(W8O_PREP)
#undef W8O_PREP
  return r;
}

// 0 ok; 1 width / split; 2 tensor map; 5 launch
int w8o_launch(const W8oArgs* a, void* stream) {
  const int S = static_cast<int>(a->S);
  const int split = static_cast<int>(a->split);
  const int chains = a->chains > 0 ? static_cast<int>(a->chains) : (split == 4 ? 2 : 1);
  if (!((split == 1 && chains == 1) || (split == 2 && (chains == 1 || chains == 2)) || (split == 4 && chains == 2)))
    return 1;
  const int cl = split / chains;  // CTAs per tile (cluster size)
  wa::Params p{};
  {
    const uint64_t dims[3] = {128, 8192, 32};
    const uint64_t strides[2] = {4096, 128};
    const uint32_t box[3] = {128, 128, 2};
    if (map(&p.tm_w, 3, a->w, dims, strides, box)) return 2;
  }
  {
    const int npad = S <= 8 ? 8 : 16;
    const uint64_t dims[4] = {128, static_cast<uint64_t>(S), 32, 8};
    const uint64_t strides[3] = {static_cast<uint64_t>(a->x_s0), 128, static_cast<uint64_t>(a->x_s1)};
    const uint32_t box[4] = {128, static_cast<uint32_t>(npad), static_cast<uint32_t>(32 / cl), 1};  // one CTA's K range
    if (map(&p.tm_x, 4, a->x, dims, strides, box)) return 2;
  }
  p.w_base = a->w;
  p.wsf = static_cast<const uint8_t*>(a->wsf);
  p.xs = static_cast<const uint32_t*>(a->xs);
  p.xs_sg = static_cast<int>(a->xs_sg);
  p.xs_skb = static_cast<int>(a->xs_skb);
  p.zq = static_cast<uint8_t*>(a->zq);
  p.zsf = static_cast<uint8_t*>(a->zsf);
  p.z = static_cast<__nv_bfloat16*>(a->z);
  p.z32 = static_cast<float*>(a->z32);
  p.diag = static_cast<w8::Diag*>(a->diag);
  p.tl = static_cast<unsigned long long*>(a->tl);
  p.ticket = static_cast<uint32_t*>(a->ticket);
  p.prologue = static_cast<uint32_t>(a->prologue);
  const bool pdl = a->pdl != 0;
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  switch (S) {
#define W8O_CASE(W) case W: return launch_split<W>(p, split, chains, pdl, st);
    W8O_WIDTHS(W8O_CASE)
#undef W8O_CASE
    default: return 1;
  }
}

}  // extern "C"
