// Peer tier v3, RTX side: host launcher for the persistent kernel (torch extension).
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cstdlib>
#include <optional>
#include <vector>

#include "peer3_rtx_kernel.cuh"

namespace {

// The persistent kernel runs on its own non-blocking stream: work on the legacy default stream must never queue
// behind a kernel that only ends when asked to.
cudaStream_t serve_stream() {
  static cudaStream_t s = nullptr;
  if (s == nullptr) C10_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  return s;
}

bool running() {
  const cudaError_t e = cudaStreamQuery(serve_stream());
  if (e == cudaErrorNotReady) return true;
  C10_CUDA_CHECK(e);
  return false;
}

void wait_exit() { C10_CUDA_CHECK(cudaStreamSynchronize(serve_stream())); }

// Load the module now and report the kernel's resources: (local bytes, regs, static smem, max threads per block).
std::vector<int64_t> preload() {
  cudaFuncAttributes a;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, peer3::rtx_persistent_kernel));
  return {int64_t(a.localSizeBytes), a.numRegs, int64_t(a.sharedSizeBytes), a.maxThreadsPerBlock};
}

int64_t launch(int64_t seg_ptr, torch::Tensor layers, torch::Tensor work, torch::Tensor xstage, torch::Tensor part1,
               torch::Tensor part2, torch::Tensor inter, torch::Tensor inter_sf, int64_t idle_limit_ns,
               int64_t wait_limit_ns, int64_t target_items, int64_t grid, std::optional<torch::Tensor> flush) {
  TORCH_CHECK(layers.is_cuda() && layers.numel() == int64_t(sizeof(peer3::LayerWeights)) * peer3::kMaxLayers,
              "layers must hold kMaxLayers LayerWeights on the device");
  TORCH_CHECK(work.is_cuda() && work.numel() >= int64_t(sizeof(peer3::RtxWork)), "work buffer too small");
  TORCH_CHECK(xstage.numel() * xstage.element_size() >= int64_t(peer3::kMaxTokens) * (peer3::kHidden + peer3::kScalesPerRow),
              "xstage too small");
  // The plan keeps ks * pairs within these capacities; 2 * kMaxPairs rows always suffice (ks1 >= 2, ks2 >= 1).
  TORCH_CHECK(part1.numel() >= int64_t(2 * peer3::kMaxPairs) * peer3::kUpTiles * 512, "part1 too small");
  TORCH_CHECK(part2.numel() >= int64_t(peer3::kMaxPairs) * peer3::kOutTiles * 256, "part2 too small");
  TORCH_CHECK(inter.numel() >= int64_t(peer3::kMaxPairs) * peer3::kInter);
  TORCH_CHECK(inter_sf.numel() >= int64_t(peer3::kMaxPairs) * (peer3::kInter / 32));
  peer3::RtxParams p{};
  p.seg = reinterpret_cast<char*>(seg_ptr);
  p.layers = reinterpret_cast<const peer3::LayerWeights*>(layers.data_ptr());
  p.work = reinterpret_cast<peer3::RtxWork*>(work.data_ptr());
  p.xstage = reinterpret_cast<uint8_t*>(xstage.data_ptr());
  p.part1 = part1.data_ptr<float>();
  p.part2 = part2.data_ptr<float>();
  p.inter = reinterpret_cast<uint8_t*>(inter.data_ptr());
  p.inter_sf = reinterpret_cast<uint8_t*>(inter_sf.data_ptr());
  p.idle_limit_ns = static_cast<unsigned long long>(idle_limit_ns);
  p.wait_limit_ns = static_cast<unsigned long long>(wait_limit_ns);
  p.target_items = int(target_items);
  p.hint_k = std::getenv("PEER3_HINT_K") ? std::atoi(std::getenv("PEER3_HINT_K")) : 0;
  TORCH_CHECK(p.hint_k >= 0 && p.hint_k <= 4, "PEER3_HINT_K must be 0..4");
  p.hint_stagger_ns = std::getenv("PEER3_HINT_STAGGER_NS") ? std::atoll(std::getenv("PEER3_HINT_STAGGER_NS")) : 11000;
  p.hint_min_layer = std::getenv("PEER3_HINT_MIN_LAYER") ? std::atoi(std::getenv("PEER3_HINT_MIN_LAYER")) : 10;
  p.prefetch_k = std::getenv("PEER3_PREFETCH_K") ? std::atoi(std::getenv("PEER3_PREFETCH_K")) : 0;
  TORCH_CHECK(p.prefetch_k >= 0 && p.prefetch_k <= peer3::kPrefetchMax, "PEER3_PREFETCH_K must be 0..", peer3::kPrefetchMax);
  p.bench_discard = std::getenv("PEER3_BENCH_DISCARD") ? std::atoi(std::getenv("PEER3_BENCH_DISCARD")) : 0;
  p.part1_rows = int(part1.numel() / (peer3::kUpTiles * 512));
  p.part2_rows = int(part2.numel() / (peer3::kOutTiles * 256));
  p.flush = flush.has_value() ? reinterpret_cast<const uint4*>(flush->data_ptr()) : nullptr;
  p.flush_bytes = flush.has_value() ? uint64_t(flush->numel()) * flush->element_size() : 0;
  const int dev = xstage.get_device();
  int sms = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  const int g = grid > 0 ? int(std::min<int64_t>(grid, sms)) : sms;
  const size_t smem = sizeof(peer3::Smem);
  C10_CUDA_CHECK(cudaFuncSetAttribute(peer3::rtx_persistent_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      int(smem)));
  int per_sm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, peer3::rtx_persistent_kernel,
                                                               peer3::kThreads, smem));
  TORCH_CHECK(per_sm >= 1 && g <= per_sm * sms, "persistent grid does not fit co-resident");
  if (std::getenv("PEER3_PLAIN_LAUNCH") != nullptr) {
    // Plain launch: co-residency is then only guaranteed by grid <= SMs on an otherwise idle device.
    peer3::rtx_persistent_kernel<<<g, peer3::kThreads, smem, serve_stream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    void* args[] = {&p};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(peer3::rtx_persistent_kernel), dim3(g),
                                               dim3(peer3::kThreads), args, smem, serve_stream()));
  }
  return g;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch", &launch, "peer3: launch the persistent RTX cold-expert kernel on its own stream (returns grid)",
        py::arg("seg_ptr"), py::arg("layers"), py::arg("work"), py::arg("xstage"), py::arg("part1"), py::arg("part2"),
        py::arg("inter"),
        py::arg("inter_sf"), py::arg("idle_limit_ns"), py::arg("wait_limit_ns"), py::arg("target_items"),
        py::arg("grid"), py::arg("flush") = py::none());
  m.def("preload", &preload, "peer3: load the RTX kernel module now; returns (local bytes, regs, smem, max threads)");
  m.def("running", &running, "peer3: is the persistent kernel still running");
  m.def("wait_exit", &wait_exit, "peer3: block until the persistent kernel exited (raises on a device error)");
  m.attr("WORK_BYTES") = sizeof(peer3::RtxWork);
  m.attr("WORK_OFF_STATS") = offsetof(peer3::RtxWork, stats);
  m.attr("WORK_OFF_EV_SUM") = offsetof(peer3::RtxWork, ev_sum);
  m.attr("WORK_OFF_EV_MIN_SUM") = offsetof(peer3::RtxWork, ev_min_sum);
  m.attr("WORK_OFF_TOP_N") = offsetof(peer3::RtxWork, top_n);
  m.attr("WORK_OFF_TOP") = offsetof(peer3::RtxWork, top);
  m.attr("WORK_OFF_POP") = offsetof(peer3::RtxWork, pop);
  m.attr("PREFETCH_MAX") = peer3::kPrefetchMax;
  m.attr("NUM_EV_SLOTS") = peer3::kNumEvSlots;
  m.attr("LAYER_BYTES") = sizeof(peer3::LayerWeights);
  m.attr("SMEM_BYTES") = sizeof(peer3::Smem);
  m.attr("MAX_LAYERS") = peer3::kMaxLayers;
  m.attr("MAX_PAIRS") = peer3::kMaxPairs;
  m.attr("PART1_FLOATS") = int64_t(peer3::kPart1Rows) * peer3::kUpTiles * 512;
  m.attr("PART2_FLOATS") = int64_t(peer3::kPart2Rows) * peer3::kOutTiles * 256;
  m.attr("SEG_BYTES") = peer3::kSegBytes;
  m.attr("OFF_CTRL") = peer3::kOffCtrl;
  m.attr("OFF_DONE") = peer3::kOffDone;
  m.attr("OFF_READY") = peer3::kOffReady;
  m.attr("OFF_CAP") = peer3::kOffCap;
  m.attr("CAP_READY") = peer3::kCapReady;
  m.attr("OFF_RTX_STATE") = peer3::kOffRtxState;
  m.attr("OFF_VERSION") = peer3::kOffVersion;
  m.attr("PROTO_VERSION") = peer3::kProtoVersion;
  m.attr("OFF_STAT_RING") = peer3::kOffStatRing;
  m.attr("STAT_RING") = peer3::kStatRing;
}
