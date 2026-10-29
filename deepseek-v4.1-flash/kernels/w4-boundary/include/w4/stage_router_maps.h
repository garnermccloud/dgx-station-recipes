// Host side of the W4 router stages: the RouterGate weight tensor maps (one TMA per ring slot).
// Encode with the caller's cuTensorMapEncodeTiled (e.g. W8's encode_fn()); gate_w is the layer's [384][5120] bf16.
#pragma once

#include <cuda.h>
#include <cudaTypedefs.h>

#include "w4/stage_router.cuh"

namespace w4::stage {

// dims innermost first (elements), strides in bytes for dims 1.., 128-B swizzle (64-column inner box)
inline CUresult encode_router_gate_maps(PFN_cuTensorMapEncodeTiled_v12000 enc, const void* gate_w, RouterGateMaps* m) {
  const cuuint32_t elem[5] = {1, 1, 1, 1, 1};
  auto mk = [&](CUtensorMap* out, int rank, const cuuint64_t* dims, const cuuint64_t* strides, const cuuint32_t* box) {
    return enc(out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, const_cast<void*>(gate_w), dims, strides, box, elem,
               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  };
  // dot: (64 cols, expert row, 64-col chunk c < 4, rank vw < 4, segment i < 5): col = 1024 i + 256 vw + 64 c + e
  const cuuint64_t dd[5] = {64, 384, 4, 4, 5}, ds[4] = {10240, 128, 512, 2048};
  const cuuint32_t db[5] = {64, 8, 4, 1, 5};
  // LL: (64 cols, row, chunk c < 4, rank z < 6, tile j < 4): col = 256 (z + 6 j) + 64 c + e; box j = 4 (z < 2) or 3
  const cuuint64_t ld[5] = {64, 384, 4, 6, 4}, ls[4] = {10240, 128, 512, 3072};
  const cuuint32_t lb4[5] = {64, 16, 4, 1, 4}, lb3[5] = {64, 16, 4, 1, 3};
  // Mega: (64 cols, row, chunk c < 10, split z < 8): col = 640 z + 64 c + e
  const cuuint64_t md[4] = {64, 384, 10, 8}, ms[3] = {10240, 128, 1280};
  const cuuint32_t mb[4] = {64, 16, 10, 1};
  CUresult r = mk(&m->dot, 5, dd, ds, db);
  if (r == CUDA_SUCCESS) r = mk(&m->ll4, 5, ld, ls, lb4);
  if (r == CUDA_SUCCESS) r = mk(&m->ll3, 5, ld, ls, lb3);
  if (r == CUDA_SUCCESS) r = mk(&m->mega, 4, md, ms, mb);
  return r;
}

}  // namespace w4::stage
