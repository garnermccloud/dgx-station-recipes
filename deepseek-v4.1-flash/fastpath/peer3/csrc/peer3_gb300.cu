// Peer tier v3, GB300 side: standalone CUDA-graph-safe kernels over the device stages in peer3_gb300.cuh.
//   send(topk_ids, topk_weights, row_map, x, hot_ids, has, seg, state, layer[, stats])   grid T, one CTA per token
//   finish(y, has, seg, state, stats, max_spins)                          grid T, CTA 0 always waits
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "peer3_gb300.cuh"

namespace {

constexpr int kSendThreads = 160;  // one MXFP8 block of 32 values per thread

template <typename IdT>
__global__ void __launch_bounds__(kSendThreads) send_kernel(const IdT* __restrict__ topk_ids, const float* __restrict__ topk_w,
                                                  const int* __restrict__ row_map, const __nv_bfloat16* __restrict__ x,
                                                  int64_t x_stride, int T, int layer, IdT* __restrict__ hot_ids,
                                                  int8_t* __restrict__ has, char* seg, const uint32_t* state,
                                                  const peer3::GbStats* stats) {
  const int t = blockIdx.x;
  __shared__ uint64_t ids_s;
  const uint32_t tag = peer3::current_tag(state, layer, T);
  if (t == 0 && threadIdx.x == 32) peer3::post_prev_wait(seg, stats, tag);
  if (threadIdx.x < 32) {
    const uint64_t ids48 =
        peer3::split_routes(topk_ids + t * peer3::kTopk, row_map, hot_ids + t * peer3::kTopk, threadIdx.x);
    if (threadIdx.x == 0) {
      ids_s = ids48;
      has[t] = peer3::any_cold(ids48) ? 1 : 0;
    }
    peer3::post_route_ids(seg, tag, t, topk_ids + t * peer3::kTopk, threadIdx.x);
  }
  __syncthreads();
  peer3::send_token_cta(seg, tag, t, ids_s, topk_w + t * peer3::kTopk, x + t * x_stride);
}

__device__ __forceinline__ void grid_dependency_wait() {
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

constexpr int kFinishThreads = 256;
constexpr int kFinishChunks = (peer3::kHidden / 8 + kFinishThreads - 1) / kFinishThreads;  // 16-byte chunks per thread

// Launched as a programmatic dependent of MegaMoE (PEER3_PDL): everything that does not read y -- the wait for the
// RTX and the cold row's host reads -- runs before griddepcontrol.wait, so it overlaps the kernel boundary. Without
// the launch attribute griddepcontrol.wait is a no-op and the kernel is the plain sequence.
__global__ void __launch_bounds__(kFinishThreads) finish_kernel(__nv_bfloat16* y, int64_t y_stride, const int8_t* has,
                                                               int T, int layer, const char* seg, uint32_t* state,
                                                               peer3::GbStats* stats, long long max_spins) {
  const int t = blockIdx.x;
  __shared__ int ok;
  const uint32_t tag = peer3::current_tag(state, layer, T);
  const bool mine = has[t] != 0;
  if (mine || t == 0) {
    if (threadIdx.x == 0) ok = peer3::wait_done(seg, tag, max_spins, t == 0 ? stats : nullptr) ? 1 : 0;
    __syncthreads();
    if (mine && ok) {
      const char* out = seg + peer3::kOffOut + uint64_t(t) * peer3::kHidden * 2;
      uint4 o[kFinishChunks];
#pragma unroll
      for (int i = 0; i < kFinishChunks; ++i) {
        const int c = threadIdx.x + i * kFinishThreads;
        if (c < peer3::kHidden / 8) o[i] = peer3::ld_volatile_v4(out + c * 16);
      }
      grid_dependency_wait();  // MegaMoE's y is complete and visible from here on
      uint4* y4 = reinterpret_cast<uint4*>(y + t * y_stride);
#pragma unroll
      for (int i = 0; i < kFinishChunks; ++i) {
        const int c = threadIdx.x + i * kFinishThreads;
        if (c < peer3::kHidden / 8) y4[c] = peer3::add_bf16x8(y4[c], o[i]);
      }
    }
  }
  // Every CTA waits for MegaMoE before it exits, so this kernel's completion implies MegaMoE's for whatever
  // depends on it (also a programmatic dependent of this kernel).
  grid_dependency_wait();
  __syncthreads();
  if (threadIdx.x == 0) {
    // Every CTA read `state` above; the last to arrive advances the call counter and resets the arrivals.
    if (peer3::atom_add_acq_rel_gpu(state + 1, 1u) == uint32_t(T - 1)) {
      state[1] = 0;
      peer3::advance(state);
      __threadfence();
    }
  }
}

void check(const torch::Tensor& t, const char* name, bool cuda = true) {
  TORCH_CHECK(!cuda || t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void send(torch::Tensor topk_ids, torch::Tensor topk_w, torch::Tensor row_map, torch::Tensor x, torch::Tensor hot_ids,
          torch::Tensor has, int64_t seg_ptr, torch::Tensor state, int64_t layer, std::optional<torch::Tensor> stats) {
  const peer3::GbStats* gs = stats.has_value() && stats->numel() ? reinterpret_cast<const peer3::GbStats*>(stats->data_ptr()) : nullptr;
  TORCH_CHECK(!stats.has_value() || stats->numel() == 0 || stats->numel() * 8 >= int64_t(sizeof(peer3::GbStats)),
              "stats needs GB_STATS_WORDS int64");
  const int T = static_cast<int>(topk_ids.size(0));
  TORCH_CHECK(T >= 1 && T <= peer3::kMaxTokens, "peer3 send handles 1..", peer3::kMaxTokens, " tokens, got ", T);
  TORCH_CHECK(layer >= 0 && layer < peer3::kMaxLayers, "peer3 layer out of range");
  TORCH_CHECK(topk_ids.size(1) == peer3::kTopk && x.size(1) == peer3::kHidden && x.dtype() == torch::kBFloat16);
  TORCH_CHECK(topk_w.dtype() == torch::kFloat32 && row_map.dtype() == torch::kInt32 && has.dtype() == torch::kInt8);
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0, "x rows must be 16-byte aligned and unit-stride");
  check(topk_ids, "topk_ids"); check(topk_w, "topk_weights"); check(row_map, "row_map"); check(hot_ids, "hot_ids");
  auto stream = at::cuda::getCurrentCUDAStream();
  char* seg = reinterpret_cast<char*>(seg_ptr);
  if (topk_ids.dtype() == torch::kInt64) {
    send_kernel<int64_t><<<T, kSendThreads, 0, stream>>>(topk_ids.data_ptr<int64_t>(), topk_w.data_ptr<float>(),
        row_map.data_ptr<int>(), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x.stride(0), T, int(layer),
        hot_ids.data_ptr<int64_t>(), has.data_ptr<int8_t>(), seg, reinterpret_cast<const uint32_t*>(state.data_ptr()), gs);
  } else {
    TORCH_CHECK(topk_ids.dtype() == torch::kInt32, "topk_ids must be int32 or int64");
    send_kernel<int32_t><<<T, kSendThreads, 0, stream>>>(topk_ids.data_ptr<int32_t>(), topk_w.data_ptr<float>(),
        row_map.data_ptr<int>(), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x.stride(0), T, int(layer),
        hot_ids.data_ptr<int32_t>(), has.data_ptr<int8_t>(), seg, reinterpret_cast<const uint32_t*>(state.data_ptr()), gs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void finish(torch::Tensor y, torch::Tensor has, int64_t seg_ptr, torch::Tensor state, torch::Tensor stats,
            int64_t max_spins, int64_t layer, bool pdl) {
  const int T = static_cast<int>(y.size(0));
  TORCH_CHECK(y.dtype() == torch::kBFloat16 && y.size(1) == peer3::kHidden && y.stride(1) == 1 && y.stride(0) % 8 == 0);
  TORCH_CHECK(stats.numel() == 0 || stats.numel() * 8 >= int64_t(sizeof(peer3::GbStats)), "stats needs GB_STATS_WORDS int64");
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(T);
  cfg.blockDim = dim3(kFinishThreads);
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, finish_kernel, reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), y.stride(0),
      static_cast<const int8_t*>(has.data_ptr<int8_t>()), T, int(layer), reinterpret_cast<const char*>(seg_ptr),
      reinterpret_cast<uint32_t*>(state.data_ptr()),
      stats.numel() ? reinterpret_cast<peer3::GbStats*>(stats.data_ptr()) : nullptr, static_cast<long long>(max_spins)));
}

// Benchmark helper: a 1-CTA responder on the GB300 itself that acknowledges every call at once (done = tag), so the
// GB300 side's own cost (send + finish, in a CUDA graph) can be measured without the RTX. Exits on ctrl != 0.
__global__ void autoack_kernel(char* seg) {
  if (threadIdx.x != 0) return;
  uint32_t last = 0;
  for (;;) {
    const uint64_t w0 = peer3::ld_acquire_sys_u64(seg + peer3::kOffRec);
    const uint32_t tag = peer3::word0_tag(w0);
    if (tag != 0 && tag != last) {
      peer3::st_release_sys_u64(seg + peer3::kOffDone, tag);
      last = tag;
    }
    if (peer3::ld_acquire_sys_u64(seg + peer3::kOffCtrl) != 0) return;
  }
}

cudaStream_t autoack_stream() {
  static cudaStream_t s = nullptr;
  if (s == nullptr) C10_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  return s;
}
void autoack_start(int64_t seg_ptr) {
  autoack_kernel<<<1, 32, 0, autoack_stream()>>>(reinterpret_cast<char*>(seg_ptr));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void autoack_wait() { C10_CUDA_CHECK(cudaStreamSynchronize(autoack_stream())); }

// CUDA lazy loading loads a module on its first launch, which needs the device to drain: with a persistent kernel
// spinning on the same device (single-GPU tests) that deadlocks. preload() forces the load up front.
void preload() {
  cudaFuncAttributes attr;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, send_kernel<int64_t>));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, send_kernel<int32_t>));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, finish_kernel));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, autoack_kernel));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preload", &preload, "peer3: load the kernels now (lazy loading would stall behind a persistent kernel)");
  m.def("autoack_start", &autoack_start, "benchmarks: 1-CTA local responder that acknowledges every call");
  m.def("autoack_wait", &autoack_wait, "benchmarks: wait for the responder to exit (set the control word first)");
  m.def("send", &send, "peer3: route split + per-token MXFP8 publish (one CTA per token)", py::arg("topk_ids"),
        py::arg("topk_weights"), py::arg("row_map"), py::arg("x"), py::arg("hot_ids"), py::arg("has"), py::arg("seg"),
        py::arg("state"), py::arg("layer"), py::arg("stats") = py::none());
  m.def("finish", &finish, "peer3: wait for the RTX acknowledgement, add cold rows, advance the call counter "
        "(pdl: launch as a programmatic dependent of the previous kernel)", py::arg("y"), py::arg("has"), py::arg("seg"),
        py::arg("state"), py::arg("stats"), py::arg("max_spins"), py::arg("layer"), py::arg("pdl") = false);
  m.attr("SEG_BYTES") = peer3::kSegBytes;
  m.attr("OFF_DONE") = peer3::kOffDone;
  m.attr("OFF_REC") = peer3::kOffRec;
  m.attr("OFF_XQ") = peer3::kOffXq;
  m.attr("OFF_XS") = peer3::kOffXs;
  m.attr("OFF_OUT") = peer3::kOffOut;
  m.attr("OFF_CTRL") = peer3::kOffCtrl;
  m.attr("OFF_RTX_STATE") = peer3::kOffRtxState;
  m.attr("OFF_RTX_STATS") = peer3::kOffRtxStats;
  m.attr("OFF_STAT_RING") = peer3::kOffStatRing;
  m.attr("STAT_RING") = peer3::kStatRing;
  m.attr("OFF_VERSION") = peer3::kOffVersion;
  m.attr("PROTO_VERSION") = peer3::kProtoVersion;
  m.attr("GB_STATS_WORDS") = int64_t(sizeof(peer3::GbStats) / 8);
  m.attr("MAX_TOKENS") = peer3::kMaxTokens;
  m.attr("MAGIC") = peer3::kMagic;
}
