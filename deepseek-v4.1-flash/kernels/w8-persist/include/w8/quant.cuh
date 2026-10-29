// W8: flashinfer's MXFP8 quantization recipe (cute-dsl MXFP8QuantizeSwizzledKernel), and the SW128 tile offset.
// amax over 32 bf16 values -> normalized = amax * fp32(1/448) -> UE8M0 = biased exponent, +1 if mantissa != 0 (except
// tiny subnormals), clamp 254, 0 for amax <= 0 -> inv = 2^(127 - e) -> e4m3 = cvt.rn.satfinite(clamp(v * inv, +-448)).
#pragma once
#include <cstdint>

namespace w8 {

constexpr float kInv448 = 1.0f / 448.0f;

__device__ __forceinline__ uint32_t ue8m0_ceil(float v) {
  const uint32_t bits = __float_as_uint(v);
  const uint32_t e = (bits >> 23) & 255u, mant = bits & 0x7fffffu;
  uint32_t bump = mant != 0 ? 1u : 0u;
  if (e == 0 && mant <= 0x400000u) bump = 0;
  uint32_t r = e + bump;
  r = r > 254u ? 254u : r;
  return v <= 0.f ? 0u : r;
}
__device__ __forceinline__ float ue8m0_inv(uint32_t e) {
  int ne = 254 - static_cast<int>(e);
  ne = ne < 0 ? 0 : ne;
  return e == 0 ? 0.f : __uint_as_float(static_cast<uint32_t>(ne) << 23);
}
__device__ __forceinline__ uint32_t e4m3_from_f32(float x) {
  x = fminf(x, 448.f);
  x = fmaxf(x, -448.f);
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.f), "f"(x));
  return r & 0xffu;
}

// Byte offset of (row r, byte k in 0..127) inside one SW128 K-major tile of 128-byte rows.
__host__ __device__ constexpr uint32_t sw128_off(uint32_t r, uint32_t k) {
  return (r >> 3) * 1024 + (r & 7) * 128 + ((((k >> 4) ^ (r & 7)) & 7) << 4) + (k & 15);
}

}  // namespace w8
