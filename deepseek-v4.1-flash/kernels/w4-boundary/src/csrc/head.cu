// W4 DSpark drafter head: production's tail of DSparkDeepseekV4Model.forward,
//   h = mhc_post_tilelang(x, residual, post, comb)    residual_new[o] = bf16(fma chain of mhc_core post_elem)
//   out = hc_collapse_triton(h, pre)                   acc = 0; acc = acc + pre[i] * h[i] (i = 0..3, no fma); bf16
// in one launch, bit for bit: one thread per 8 hidden elements of a token.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "w4/mhc_core.cuh"
#include "w4/params.h"

namespace {

using w4::HeadParams;

constexpr int kH = 5120, kR = 4, kThreads = 256, kItems = kH / 8;  // 640 items per token

__global__ void __launch_bounds__(kThreads) head_kernel(const HeadParams p) {
  asm volatile("griddepcontrol.wait;" ::: "memory");  // a no-op unless launched as a PDL dependent
  const int item = blockIdx.x * kThreads + threadIdx.x;
  if (item >= p.T * kItems) return;
  const int t = item / kItems, h0 = (item % kItems) * 8;
  float x[8], r[kR][8];
  w4::unpack8(__ldg(reinterpret_cast<const uint4*>(p.x + (size_t)t * kH + h0)), x);
#pragma unroll
  for (int j = 0; j < kR; ++j)
    w4::unpack8(__ldg(reinterpret_cast<const uint4*>(p.residual + ((size_t)t * kR + j) * kH + h0)), r[j]);
  float post[kR], comb[kR][kR], pre[kR];
#pragma unroll
  for (int i = 0; i < kR; ++i) {
    post[i] = p.post[t * kR + i];
    pre[i] = p.pre[t * kR + i];
#pragma unroll
    for (int j = 0; j < kR; ++j) comb[j][i] = p.comb[t * kR * kR + j * kR + i];
  }
  float o[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const float re[kR] = {r[0][e], r[1][e], r[2][e], r[3][e]};
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < kR; ++i) {
      const float col[kR] = {comb[0][i], comb[1][i], comb[2][i], comb[3][i]};
      const float rn = w4::bf2f(w4::f2bf(w4::post_elem(post[i], x[e], col, re)));  // the stored residual_new
      acc = __fadd_rn(acc, __fmul_rn(pre[i], rn));
    }
    o[e] = acc;
  }
  *reinterpret_cast<uint4*>(p.out + (size_t)t * kH + h0) = w4::pack8(o);
}

}  // namespace

extern "C" int w4_head_launch(const HeadParams* p, int pdl, cudaStream_t stream) {
  if (p->T < 1) return 0;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl;
  cudaLaunchConfig_t c = {};
  c.gridDim = dim3((p->T * kItems + kThreads - 1) / kThreads);
  c.blockDim = dim3(kThreads);
  c.stream = stream;
  c.attrs = at;
  c.numAttrs = 1;
  return (int)cudaLaunchKernelEx(&c, head_kernel, *p);
}
