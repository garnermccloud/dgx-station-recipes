// RTX: cost of the per-call plan (plan_experts + plan_pairs) in isolation, warm, one CTA per SM like the persistent
// kernel: is the ~3 us seen in-call the code itself or something around it?
#include <cstdio>
#include "../csrc/peer3_rtx_kernel.cuh"
using namespace peer3;
__global__ void __launch_bounds__(kThreads, 1) bench(RtxParams p, unsigned long long* out, int T, int reps) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
  if (threadIdx.x < T) {  // 1 cold pick per token on expert (t % 2), the rest hot
    uint64_t ids = 0xFFFFFFFFFFFFull;
    ids = (ids & ~0xFFull) | uint64_t(threadIdx.x % 2);
    sm.ids[threadIdx.x] = make_word0(1, ids);
  }
  __syncthreads();
  unsigned long long t[4] = {0, 0, 0, 0};
  for (int r = 0; r < reps; ++r) {
    const unsigned long long a = clock64();
    if (threadIdx.x < 32) build_rows(sm, T);
    __syncthreads();
    const unsigned long long b = clock64();
    if (threadIdx.x < 32) plan_experts(p, sm, T, gridDim.x);
    __syncthreads();
    const unsigned long long c = clock64();
    if (threadIdx.x < 32) plan_pairs(sm, T);
    __syncthreads();
    const unsigned long long d = clock64();
    if (r > 0) t[0] += b - a, t[1] += c - b, t[2] += d - c;
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = t[0] / (reps - 1);
    out[1] = t[1] / (reps - 1);
    out[2] = t[2] / (reps - 1);
    out[3] = sm.n_exp;
  }
}
int main() {
  unsigned long long* d;
  cudaMalloc(&d, 64);
  RtxParams p{};
  p.target_items = 0;
  p.part1_rows = kPart1Rows;
  p.part2_rows = kPart2Rows;
  const size_t smem = sizeof(Smem);
  cudaFuncSetAttribute(bench, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
  for (int T : {1, 6, 16, 64}) {
    bench<<<188, kThreads, smem>>>(p, d, T, 100);
    cudaError_t e = cudaDeviceSynchronize();
    unsigned long long h[4];
    cudaMemcpy(h, d, 32, cudaMemcpyDeviceToHost);
    printf("T=%2d: build_rows %llu cycles, plan_experts %llu, plan_pairs %llu (n_exp %llu) %s\n", T, h[0], h[1], h[2],
           h[3], cudaGetErrorString(e));
  }
  return 0;
}
