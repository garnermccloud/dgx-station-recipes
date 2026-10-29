// Standalone (PDL) launcher for W1's decode attention: ONE persistent launch that runs the split stage, then the
// combine stage, CTAs handing partials over through epoch-tagged flags (the W8 model, so no kernel boundary between
// the stages). Warp 8 (the producer warp, idle for attention) issues W7's L2 prefetch of the next GEMMs' weights.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include "w1/stage_attn_combine.cuh"
#include "w1/stage_attn_cx.cuh"
#include "w1/stage_attn_par.cuh"
#include "w1/stage_attn_seq.cuh"
#include "w1/stage_attn_ss.cuh"
#include "w1/stage_attn_split.cuh"
#include "w7/w7_prefetch.cuh"

namespace {

using Split = w1::AttnSplit<16, 2>;
using Combine = w1::AttnCombine<10>;
constexpr uint32_t kSmem = (Split::kSmemBytes > Combine::kSmemBytes ? Split::kSmemBytes : Combine::kSmemBytes);

using w1::cta_stamp;

__global__ void __launch_bounds__(w8::kBlockThreads, 1) attn_kernel(const __grid_constant__ w1::AttnParams p) {
    extern __shared__ uint8_t smem_raw[];
    cta_stamp(p, 0);
    __shared__ uint32_t tmem_slot;
    uint8_t* smem = smem_raw + ((1024u - (w1::smem_u32(smem_raw) & 1023u)) & 1023u);
    const int warp = threadIdx.x >> 5;
    // Every CTA has started once all of them trigger, so the dependent grid can never take an SM a CTA of this grid
    // still needs (the combine stage waits on other CTAs).
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (warp == w8::kProducerWarp) {
        if ((threadIdx.x & 31) == 0 && p.prefetch_n > 0)
            w7::prefetch_stage(p.prefetch_table, p.prefetch_n, blockIdx.x, gridDim.x);
        return;
    }
    if (warp == 0) w1::tc_alloc(&tmem_slot, Split::kTmemCols);
    w1::tc_fence_before();
    w8::compute_sync();
    w1::tc_fence_after();
    const uint32_t tmem = tmem_slot;

    cta_stamp(p, 1);
    w8::StandaloneSync sync;
    sync.wait_extern();
    sync.tag = *reinterpret_cast<volatile uint32_t*>(p.epoch) + 1u;
    w8::Ring ring;
    const int S = p.s_q;
    const int n_split = Split::num_tasks(p, S);
    for (int task = blockIdx.x; task < n_split; task += gridDim.x) {
        Split::consume(p, S, task, sync, ring, smem, tmem);
        w8::compute_sync();
        if (threadIdx.x == 0) {
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            w8::st_relaxed(p.flags + task, sync.tag);
        }
    }
    cta_stamp(p, 2);
    const int n_comb = Combine::num_tasks(p, S);
    for (int task = blockIdx.x; task < n_comb; task += gridDim.x) {
        Combine::consume(p, S, task, sync, ring, smem, 0);
        w8::compute_sync();
    }
    cta_stamp(p, 3);
    if (warp == 0) {
        w1::tc_fence_after();
        w1::tc_dealloc(tmem, Split::kTmemCols);
    }
    // launch epoch: the last CTA to finish bumps it (the next launch reads it after its griddepcontrol.wait)
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(p.epoch + 1, 1u) == gridDim.x - 1) {
            p.epoch[1] = 0;
            __threadfence();
            atomicExch(p.epoch, sync.tag);
        }
    }
}


// Bitwise mode: one task per (token, 32-head group), FlashMLA's arithmetic in order.
__global__ void __launch_bounds__(w8::kBlockThreads, 1) attn_seq_kernel(const __grid_constant__ w1::AttnParams p) {
    extern __shared__ uint8_t smem_raw[];
    __shared__ uint32_t tmem_slot;
    uint8_t* smem = smem_raw + ((1024u - (w1::smem_u32(smem_raw) & 1023u)) & 1023u);
    const int warp = threadIdx.x >> 5;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (warp == w8::kProducerWarp) {
        if ((threadIdx.x & 31) == 0 && p.prefetch_n > 0)
            w7::prefetch_stage(p.prefetch_table, p.prefetch_n, blockIdx.x, gridDim.x);
        return;
    }
    if (warp == 0) w1::tc_alloc(&tmem_slot, w1::AttnSeq::kTmemCols);
    w1::tc_fence_before();
    w8::compute_sync();
    w1::tc_fence_after();
    const uint32_t tmem = tmem_slot;
    w8::StandaloneSync sync;
    w8::Ring ring;
    const int n = w1::AttnSeq::num_tasks(p, p.s_q);
    for (int task = blockIdx.x; task < n; task += gridDim.x) {
        w1::AttnSeq::consume(p, p.s_q, task, sync, ring, smem, tmem);
        w8::compute_sync();
    }
    if (warp == 0) {
        w1::tc_fence_after();
        w1::tc_dealloc(tmem, w1::AttnSeq::kTmemCols);
    }
}


// Bitwise-parallel mode: score tasks (block-major) then PV tasks, one persistent launch.
__global__ void __launch_bounds__(w8::kBlockThreads, 1) attn_par_kernel(const __grid_constant__ w1::AttnParams p) {
    extern __shared__ uint8_t smem_raw[];
    __shared__ uint32_t tmem_slot;
    __shared__ uint32_t tag_slot;
    __shared__ __align__(8) uint64_t par_bars[3];
    uint8_t* smem = smem_raw + ((1024u - (w1::smem_u32(smem_raw) & 1023u)) & 1023u);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    w1::ParBars bars;
    bars.score = par_bars;
    bars.q = par_bars + 1;
    bars.pvc = par_bars + 2;
    w8::StandaloneSync sync;
    const int nA = p.s_q * p.par_nb, nB = p.s_q * 8;
    if (warp == w8::kProducerWarp) {
        // launch tag = epoch + 1, handed to the compute warps (named barrier 3); the last CTA to arrive publishes it
        // as the new epoch (every CTA read the old one before arriving)
        uint32_t e = 0;
        if (lane == 0) {
            sync.wait_extern();
            e = w8::ld_acquire(p.epoch);
            tag_slot = e + 1u;
        }
        __syncwarp();
        asm volatile("bar.arrive %0, %1;" ::"n"(w1::ParTag::kBarrier), "n"(w8::kBlockThreads) : "memory");
        if (lane == 0) {
            uint32_t old;
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(p.epoch + 1) : "memory");
            if (old == gridDim.x - 1) {
                w8::st_relaxed(p.epoch + 1, 0u);
                asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p.epoch), "r"(e + 1u) : "memory");
            }
            if (p.prefetch_n > 0) w7::prefetch_stage(p.prefetch_table, p.prefetch_n, blockIdx.x, gridDim.x);
        }
        return;
    }
    cta_stamp(p, 0);
    bars.init();
    if (warp == 0) w1::tc_alloc(&tmem_slot, w1::ParScore::kTmemCols);
    w1::tc_fence_before();
    w8::compute_sync();
    w1::tc_fence_after();
    const uint32_t tmem = tmem_slot;
    w1::ParTag tg;
    tg.slot = &tag_slot;
    sync.wait_extern();
    cta_stamp(p, 1);
    const w1::ParScratch ps{p.par_cmax, p.par_psum, p.par_pimg, p.par_hint, p.par_nb};
    for (int task = blockIdx.x; task < nA + nB; task += gridDim.x) {
#ifdef W1_DBG_TWICE
        for (int rep = 0; rep < 2; ++rep) {  // debug: the second pass runs with a warm instruction cache
#endif
        if (task < nA)
            w1::ParScore::consume(p, ps, task, sync, tg, smem, tmem, bars);
        else
            w1::ParPV::consume(p, ps, task - nA, sync, tg, smem, tmem, bars);
        w8::compute_sync();
#ifdef W1_DBG_TWICE
        }
#endif
    }
    tg.get();  // consume warp 8's arrival even if every task of this CTA was empty
    cta_stamp(p, 7);
    if (warp == 0) {
        w1::tc_fence_after();
        w1::tc_dealloc(tmem, w1::ParScore::kTmemCols);
    }
}

// Split-dims sequential mode (SS, stage_attn_ss.cuh): no cross-CTA hand-over; task = (token, 32-head group, T dim tiles)
template <int T>
__global__ void __launch_bounds__(w8::kBlockThreads, 1) attn_ss_kernel(const __grid_constant__ w1::AttnParams p) {
    using SS = w1::AttnSS<T>;
    extern __shared__ uint8_t smem_raw[];
    __shared__ uint32_t tmem_slot;
    __shared__ __align__(8) uint64_t ss_bars[6 + SS::kMaxBlocks];
    uint8_t* smem = smem_raw + ((1024u - (w1::smem_u32(smem_raw) & 1023u)) & 1023u);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    cta_stamp(p, 0);
    typename SS::Bars bars;
    bars.qk = ss_bars;
    bars.tready = ss_bars + 2;
    bars.pready = ss_bars + 4;
    bars.pvc = ss_bars + 6;
    if (threadIdx.x == 0) {
        for (int i = 0; i < 2; ++i) w1::mbar_init(ss_bars + i, 1);
        for (int i = 2; i < 6; ++i) w1::mbar_init(ss_bars + i, 8);  // one arrival per compute warp
        for (int i = 0; i < SS::kMaxBlocks; ++i) w1::mbar_init(ss_bars + 6 + i, 1);
        w1::fence_mbar_init();
    }
    if (warp == 0) w1::tc_alloc(&tmem_slot, SS::kTmemCols);
    w1::tc_fence_before();
    __syncthreads();  // all 288: barriers and the TMEM address
    w1::tc_fence_after();
    const uint32_t tmem = tmem_slot;
    w8::StandaloneSync sync;
    const int ntask = p.s_q * SS::kTasksPerToken;
    if (warp == w8::kProducerWarp) {
        // warp 8: W7 prefetch, then the MMA issuer
        if (lane == 0) {
            if (p.prefetch_n > 0) w7::prefetch_stage(p.prefetch_table, p.prefetch_n, blockIdx.x, gridDim.x);
            for (int task = blockIdx.x; task < ntask; task += gridDim.x) SS::issue_loop(p, task, sync, smem, tmem, bars);
        }
        return;
    }
    sync.wait_extern();
    cta_stamp(p, 1);
    for (int task = blockIdx.x; task < ntask; task += gridDim.x) {
        SS::consume(p, task, sync, smem, tmem, bars);
        w8::compute_sync();
    }
    cta_stamp(p, 7);
    if (warp == 0) {
        w1::tc_fence_after();
        w1::tc_dealloc(tmem, SS::kTmemCols);
    }
}

// Cluster mode (CX, stage_attn_cx.cuh): one 2-CTA cluster per token, CTA r = FlashMLA's QK rail r (atoms of parity r).
// The two CTAs of a cluster are co-scheduled; no CTA waits on another cluster.
__global__ void __launch_bounds__(w1::AttnCX::kThreads, 1) attn_cx_kernel(const __grid_constant__ w1::AttnParams p) {
    using CX = w1::AttnCX;
    extern __shared__ uint8_t smem_raw[];
    __shared__ uint32_t tmem_slot;
    __shared__ __align__(8) uint64_t cx_bars[CX::Bars::kCount];
    uint8_t* smem = smem_raw + ((1024u - (w1::smem_u32(smem_raw) & 1023u)) & 1023u);
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int r = (int)w1::cluster_rank();
    CX::Bars bars;
    bars.carve(cx_bars);
    if (threadIdx.x == 0) bars.init();
    if (threadIdx.x < 32) w1::tc_alloc(&tmem_slot, CX::kTmemCols);
    w1::tc_fence_before();
    __syncthreads();
    w1::tc_fence_after();
    // publishes the barrier init to the partner; every warp waits on the cluster barrier before its first remote op
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    CX::run(p, smem, tmem_slot, bars, r, (int)(blockIdx.x >> 1));
    w1::cluster_sync_all();  // no DSMEM traffic into either CTA is in flight past this point
    if (threadIdx.x < 32) {
        w1::tc_fence_after();
        w1::tc_dealloc(tmem_slot, CX::kTmemCols);
    }
}

int num_sms() {
    static int n = 0;
    if (n == 0) {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
    }
    return n;
}

constexpr uint32_t kSmemPar =
    (w1::ParScore::kSmemBytes > w1::ParPV::kSmemBytes ? w1::ParScore::kSmemBytes : w1::ParPV::kSmemBytes) + 1024;

// one-time host setup of the bitwise kernels (smem opt-in); idempotent
uint32_t configure_par() {
    static bool done = false;
    if (!done) {
        TORCH_CHECK(cudaFuncSetAttribute(attn_par_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemPar) ==
                    cudaSuccess);
        num_sms();
        done = true;
    }
    return kSmemPar;
}
template <int T>
uint32_t configure_ss() {
    static bool done = false;
    constexpr uint32_t smem = w1::AttnSS<T>::kSmemBytes + 1024;
    if (!done) {
        TORCH_CHECK(cudaFuncSetAttribute(attn_ss_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) ==
                    cudaSuccess);
        done = true;
    }
    return smem;
}
uint32_t configure_cx() {
    static bool done = false;
    constexpr uint32_t smem = w1::AttnCX::kSmemBytes + 1024;
    if (!done) {
        TORCH_CHECK(cudaFuncSetAttribute(attn_cx_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) ==
                    cudaSuccess);
        done = true;
    }
    return smem;
}
void launch_cx(const w1::AttnParams& p, int S, bool pdl) {
    const uint32_t smem = configure_cx();
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(2 * S);
    cfg.blockDim = dim3(w1::AttnCX::kThreads);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    attr[1].id = cudaLaunchAttributeClusterDimension;
    attr[1].val.clusterDim.x = 2;
    attr[1].val.clusterDim.y = 1;
    attr[1].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 2;
    cudaError_t e = cudaLaunchKernelEx(&cfg, attn_cx_kernel, p);
    TORCH_CHECK(e == cudaSuccess, "w1 cx launch failed: ", cudaGetErrorString(e));
}
template <int T>
void launch_ss(const w1::AttnParams& p, int S, bool pdl, int64_t max_ctas) {
    const uint32_t smem = configure_ss<T>();
    int grid = std::min(S * w1::AttnSS<T>::kTasksPerToken, num_sms());
    if (max_ctas > 0) grid = std::min<int>(grid, (int)max_ctas);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(w8::kBlockThreads);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaError_t e = cudaLaunchKernelEx(&cfg, attn_ss_kernel<T>, p);
    TORCH_CHECK(e == cudaSuccess, "w1 ss launch failed: ", cudaGetErrorString(e));
}

// Kernels whose CTAs wait on other CTAs' tasks (split -> combine flags, score -> PV tagged words) launch cooperative:
// the whole grid is resident before any CTA runs, so a waiting CTA never spins on a CTA that kernels of other
// streams keep from being scheduled. PDL as before.
void set_handoff_attrs(cudaLaunchAttribute (&attr)[2], bool pdl) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    attr[1].id = cudaLaunchAttributeCooperative;
    attr[1].val.cooperative = 1;
}

const int32_t* opt_i32(const c10::optional<torch::Tensor>& t) {
    return t.has_value() ? t->data_ptr<int32_t>() : nullptr;
}

}  // namespace

// One decode attention call (split + combine). See w1attn/ops.py for the argument contract.
void w1_attn_decode(torch::Tensor q, torch::Tensor positions, torch::Tensor cos_sin, torch::Tensor sink,
                    double sm_scale_log2, torch::Tensor swa_cache, torch::Tensor swa_idx, torch::Tensor swa_lens,
                    c10::optional<torch::Tensor> comp_cache, c10::optional<torch::Tensor> comp_idx,
                    c10::optional<torch::Tensor> comp_lens, torch::Tensor out_fp8, torch::Tensor out_sf,
                    torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor flags, torch::Tensor epoch,
                    c10::optional<torch::Tensor> prefetch_table, int64_t n_chunks, int64_t nsub, bool pdl,
                    int64_t max_ctas, c10::optional<torch::Tensor> dbg_ts) {
    const int S = (int)q.size(0);
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && q.stride(-1) == 1 && q.size(1) == 64 && q.size(2) == 512);
    TORCH_CHECK(swa_cache.scalar_type() == torch::kUInt8 && swa_cache.size(-1) == 528 && swa_cache.stride(1) == 528);
    const bool bitwise = n_chunks == 0;
    const bool ss = n_chunks == -2;
    const bool par = n_chunks == -1;
    const bool cx = n_chunks == -3;
    TORCH_CHECK(bitwise || par || cx || (ss && (nsub == 1 || nsub == 2 || nsub == 4)) ||
                (nsub >= 1 && nsub <= 2 && n_chunks >= 1 && n_chunks <= 10));
    w1::AttnParams p = {};
    p.q = reinterpret_cast<const uint16_t*>(q.data_ptr());
    p.q_tok_stride = q.stride(0);
    p.positions = positions.data_ptr();
    p.positions_i64 = positions.scalar_type() == torch::kInt64;
    p.cos_sin = cos_sin.data_ptr<float>();
    p.sink = sink.data_ptr<float>();
    p.sm_scale_log2 = (float)sm_scale_log2;
    p.s_q = S;
    p.swa = swa_cache.data_ptr<uint8_t>();
    p.swa_block_stride = swa_cache.stride(0);
    p.swa_page = (int)swa_cache.size(1);
    p.swa_idx = swa_idx.data_ptr<int32_t>();
    p.swa_idx_stride = swa_idx.stride(0);
    p.swa_width = (int)swa_idx.size(-1);
    p.swa_lens = swa_lens.data_ptr<int32_t>();
    if (comp_cache.has_value()) {
        TORCH_CHECK(comp_cache->scalar_type() == torch::kUInt8 && comp_cache->size(-1) == 288 &&
                    comp_cache->stride(1) == 288);
        p.comp = comp_cache->data_ptr<uint8_t>();
        p.comp_block_stride = comp_cache->stride(0);
        p.comp_page = (int)comp_cache->size(1);
        p.comp_idx = comp_idx->data_ptr<int32_t>();
        p.comp_idx_stride = comp_idx->stride(0);
        p.comp_width = (int)comp_idx->size(-1);
        p.comp_lens = opt_i32(comp_lens);
    }
    const int total = p.swa_width + (p.comp ? p.comp_width : 0);
    TORCH_CHECK(bitwise || par || ss || cx || (int64_t)n_chunks * nsub * 64 >= total, "plan does not cover all slots");
    p.n_chunks = (int)n_chunks;
    p.nsub = (int)nsub;
    p.head_groups = 4;
    p.comb_heads = 4;
    TORCH_CHECK(part_o.numel() >= (int64_t)S * std::max<int64_t>(n_chunks, 0) * 64 * 512 &&
                part_ml.numel() >= (int64_t)S * std::max<int64_t>(n_chunks, 0) * 128);
    TORCH_CHECK(flags.numel() >= (int64_t)S * std::max<int64_t>(n_chunks, 0) * 4 && epoch.numel() >= 2);
    TORCH_CHECK(!bitwise || comp_lens.has_value() || !comp_cache.has_value());
    p.part_o = part_o.data_ptr<float>();
    p.part_ml = part_ml.data_ptr<float>();
    p.flags = reinterpret_cast<uint32_t*>(flags.data_ptr());
    p.epoch = reinterpret_cast<uint32_t*>(epoch.data_ptr());
    if (prefetch_table.has_value()) {
        p.prefetch_table = prefetch_table->data_ptr<int64_t>();
        p.prefetch_n = (int)prefetch_table->size(0);
    }
    p.out_fp8 = reinterpret_cast<uint8_t*>(out_fp8.data_ptr());
    p.out_sf = out_sf.data_ptr<int32_t>();
    p.sf_stride_tok = out_sf.stride(0);
    p.sf_stride_grp = out_sf.stride(1);
    p.sf_stride_hd = out_sf.stride(2);
    p.dbg_ts = dbg_ts.has_value() ? reinterpret_cast<unsigned long long*>(dbg_ts->data_ptr()) : nullptr;

    if (cx) {
        TORCH_CHECK((total + 63) / 64 <= w1::AttnCX::kMaxBlocks, "too many key blocks for the CX mode");
        TORCH_CHECK(p.swa_width <= 128, "CX: window slots must fit the first key pair");
        TORCH_CHECK((p.swa_page & (p.swa_page - 1)) == 0 && (!p.comp || (p.comp_page & (p.comp_page - 1)) == 0),
                    "CX needs power-of-two pages");
        launch_cx(p, S, pdl);
        return;
    }
    if (ss) {
        TORCH_CHECK((total + 63) / 64 <= w1::AttnSS<1>::kMaxBlocks, "too many key blocks for the SS mode");
        if (nsub == 1)
            launch_ss<1>(p, S, pdl, max_ctas);
        else if (nsub == 2)
            launch_ss<2>(p, S, pdl, max_ctas);
        else
            launch_ss<4>(p, S, pdl, max_ctas);
        return;
    }
    if (par) {
        const int NB = (total + 63) / 64;
        TORCH_CHECK(NB <= w1::ParPV::kMaxBlocks, "too many key blocks for the bitwise-parallel mode");
        // carve the tagged-word scratch out of part_o (never cleared: tags are unique per launch)
        uint64_t* w = reinterpret_cast<uint64_t*>(p.part_o);
        p.par_cmax = w;  w += (int64_t)S * NB * 128;
        p.par_psum = w;  w += (int64_t)S * NB * 128;
        p.par_pimg = w;  w += (int64_t)S * NB * 2048;
        p.par_hint = w;  w += (int64_t)S * NB;
        TORCH_CHECK(reinterpret_cast<float*>(w) <= part_o.data_ptr<float>() + part_o.numel(),
                    "scratch too small for the bitwise-parallel mode");
        p.par_nb = NB;
        const uint32_t smem_par = configure_par();
        int grid = std::min(S * NB + S * 8, num_sms());
        if (max_ctas > 0) grid = std::min<int>(grid, (int)max_ctas);
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(grid);
        cfg.blockDim = dim3(w8::kBlockThreads);
        cfg.dynamicSmemBytes = smem_par;
        cfg.stream = c10::cuda::getCurrentCUDAStream().stream();
        cudaLaunchAttribute attr[3] = {};
        set_handoff_attrs(*reinterpret_cast<cudaLaunchAttribute (*)[2]>(attr), pdl);
        attr[2].id = cudaLaunchAttributePriority;
        attr[2].val.priority = -1;  // high-priority attention grid: preempt best-effort woa/LL CTAs
        cfg.attrs = attr;
        cfg.numAttrs = 3;
        cudaError_t e = cudaLaunchKernelEx(&cfg, attn_par_kernel, p);
        TORCH_CHECK(e == cudaSuccess, "w1 par launch failed: ", cudaGetErrorString(e));
        return;
    }
    if (bitwise) {
        static bool cfg_seq = false;
        constexpr uint32_t smem_seq = w1::AttnSeq::kSmemBytes + 1024;
        if (!cfg_seq) {
            TORCH_CHECK(cudaFuncSetAttribute(attn_seq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_seq) ==
                        cudaSuccess);
            cfg_seq = true;
        }
        int grid = std::min(w1::AttnSeq::num_tasks(p, S), num_sms());
        if (max_ctas > 0) grid = std::min<int>(grid, (int)max_ctas);
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(grid);
        cfg.blockDim = dim3(w8::kBlockThreads);
        cfg.dynamicSmemBytes = smem_seq;
        cfg.stream = c10::cuda::getCurrentCUDAStream().stream();
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        cudaError_t e = cudaLaunchKernelEx(&cfg, attn_seq_kernel, p);
        TORCH_CHECK(e == cudaSuccess, "w1 seq launch failed: ", cudaGetErrorString(e));
        return;
    }
    static bool configured = false;
    if (!configured) {
        TORCH_CHECK(cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem + 1024) ==
                    cudaSuccess);
        configured = true;
    }
    const int tasks = std::max(Split::num_tasks(p, S), Combine::num_tasks(p, S));
    int grid = std::min(tasks, num_sms());
    if (max_ctas > 0) grid = std::min<int>(grid, (int)max_ctas);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(w8::kBlockThreads);
    cfg.dynamicSmemBytes = kSmem + 1024;
    cfg.stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaLaunchAttribute attr[2];
    set_handoff_attrs(attr, pdl);
    cfg.attrs = attr;
    cfg.numAttrs = 2;
    cudaError_t e = cudaLaunchKernelEx(&cfg, attn_kernel, p);
    TORCH_CHECK(e == cudaSuccess, "w1 launch failed: ", cudaGetErrorString(e));
}

// Everything a launch needs from the host, done ahead of CUDA-graph capture.
void w1_attn_prepare() {
    configure_par();
    configure_cx();
    configure_ss<1>();
    configure_ss<2>();
    configure_ss<4>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("w1_attn_decode", &w1_attn_decode);
    m.def("w1_attn_prepare", &w1_attn_prepare);
}
