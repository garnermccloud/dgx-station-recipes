// Host-side TMA tensor-map construction without linking libcuda (driver entry point lookup).
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace w3 {

using EncodeTiledFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                   const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                   CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

inline EncodeTiledFn encode_tiled_fn() {
  static EncodeTiledFn fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) != cudaSuccess ||
        q != cudaDriverEntryPointSuccess || p == nullptr)
      throw std::runtime_error("cuTensorMapEncodeTiled entry point not found");
    return reinterpret_cast<EncodeTiledFn>(p);
  }();
  return fn;
}

// 2D row-major byte matrix [rows, cols] (cols contiguous), box [box_rows, box_cols], 128 B swizzle.
inline CUtensorMap make_tmap_2d_u8(const void* base, uint64_t rows, uint64_t cols, uint64_t row_stride_bytes,
                                   uint32_t box_rows, uint32_t box_cols, CUtensorMapSwizzle swz) {
  CUtensorMap m;
  cuuint64_t dims[2] = {cols, rows};
  cuuint64_t strides[1] = {row_stride_bytes};
  cuuint32_t box[2] = {box_cols, box_rows};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = encode_tiled_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides, box,
                                 estr, CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) throw std::runtime_error("cuTensorMapEncodeTiled failed: " + std::to_string(int(r)));
  return m;
}

// 2D row-major bf16 matrix [rows, cols], box [box_rows, box_cols] (box_cols * 2 bytes = 128 for SW128).
inline CUtensorMap make_tmap_2d_bf16(const void* base, uint64_t rows, uint64_t cols, uint64_t row_stride_bytes,
                                     uint32_t box_rows, uint32_t box_cols, CUtensorMapSwizzle swz) {
  CUtensorMap m;
  cuuint64_t dims[2] = {cols, rows};
  cuuint64_t strides[1] = {row_stride_bytes};
  cuuint32_t box[2] = {box_cols, box_rows};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = encode_tiled_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dims, strides,
                                 box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) throw std::runtime_error("cuTensorMapEncodeTiled(bf16) failed: " + std::to_string(int(r)));
  return m;
}

}  // namespace w3
