// MegaMoE all-token L1 head; production scheduler and v11 combine contract.
#pragma once

#include <cstdint>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

// W6 peer3 finish fold (tools/megamoe_p3fold_patch.py): the peer tier v3 protocol pieces the combine needs, the same
// instructions as peer3_gb300.cuh (wait_done, add_bf16x8); offsets come from the control block, not constants here.
namespace w6p3 {
constexpr uint32_t kCtlWords = 32;           // 128 bytes of l2_empty_count entries
constexpr uint32_t kCtlHas = 64;             // byte offset of has[64]
struct GbStats {                             // = peer3::GbStats
    unsigned long long calls, waited, wait_ns, timeouts;
    unsigned long long hist[16];
    unsigned long long last_wait_ns, last_tag;
};
struct Ctl {
    unsigned long long seg, state, stats, max_spins;
    uint32_t off_done, off_rec, off_out, hint;
    uint32_t pad1[4];
    int8_t has[64];
};
static_assert(sizeof(Ctl) == kCtlWords * 4, "fold control block is 128 bytes");
// v11 (combine-in-seam): fold control + 256 B; the block's 512-B l2_empty_count tail holds both
struct C11 {
    uint32_t skip_combine, magic;
    unsigned long long slot_base, slot_stride, token_stride;
};
constexpr uint32_t kC11Offset = 256;
constexpr uint32_t kC11Magic = 0x31314336u;  // "6C11"
__device__ __forceinline__ uint32_t ld_volatile_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ uint64_t ld_acquire_sys_u64(const void* p) {
    uint64_t v;
    asm volatile("ld.acquire.sys.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint64_t ld_volatile_u64(const void* p) {
    uint64_t v;
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ uint2 ld_volatile_v2(const void* p) {
    uint2 v;
    asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}
__device__ __forceinline__ uint64_t ld_volatile_shared_u64(const void* p) {
    return *reinterpret_cast<const volatile uint64_t*>(p);
}
__device__ __forceinline__ int ld_volatile_s8(const void* p) {
    int v;
    asm volatile("ld.volatile.global.s8 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
// The call's tag: token record 0's word0, published by the call's send before MegaMoE (release, sys)
__device__ __forceinline__ uint32_t call_tag(const char* seg, const Ctl* c) {
    return static_cast<uint32_t>(ld_volatile_u64(seg + c->off_rec) >> 48);
}
// peer3::wait_done (one thread): acquire-poll the done word until it carries `tag`, up to max_spins polls
__device__ __forceinline__ bool wait_done(const char* seg, const Ctl* c, uint32_t tag, GbStats* stats) {
    const uint64_t t0 = globaltimer();
    const long long max_spins = static_cast<long long>(c->max_spins);
    long long spins = 0;
    uint64_t done = ld_acquire_sys_u64(seg + c->off_done);
    while (done != tag and spins < max_spins) {
        done = ld_acquire_sys_u64(seg + c->off_done);
        ++ spins;
    }
    const bool ok = done == tag;
    if (stats != nullptr) {
        const uint64_t ns = globaltimer() - t0;
        atomicAdd(&stats->calls, 1ull);
        atomicAdd(&stats->wait_ns, ns);
        if (not ok) atomicAdd(&stats->timeouts, 1ull);
        int b = 0;
        for (uint64_t us = ns / 1000; us > 0 and b < 15; us >>= 1) ++ b;
        atomicAdd(&stats->hist[b], 1ull);
        stats->last_wait_ns = ns;
        stats->last_tag = tag;
    }
    return ok;
}
__device__ __forceinline__ uint32_t next_counter(uint32_t c) { return c >= 15 ? 1 : c + 1; }
// y = bf16(float(y) + float(o)) per bf16 pair (peer3::add_bf16x8 on 4 channels)
__device__ __forceinline__ uint2 add_bf16x4(uint2 y, const uint2 o) {
    const uint32_t yw[2] = {y.x, y.y}, ow[2] = {o.x, o.y};
    uint32_t r[2];
    #pragma unroll
    for (int j = 0; j < 2; ++ j) {
        const float y0 = __uint_as_float(yw[j] << 16) + __uint_as_float(ow[j] << 16);
        const float y1 = __uint_as_float(yw[j] & 0xFFFF0000u) + __uint_as_float(ow[j] & 0xFFFF0000u);
        const __nv_bfloat162 v = __floats2bfloat162_rn(y0, y1);
        r[j] = *reinterpret_cast<const uint32_t*>(&v);
    }
    return make_uint2(r[0], r[1]);
}
}  // namespace w6p3

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumSharedExperts,
    uint32_t kNumTopk,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t STORE_BLOCK_M,
    uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N,
    uint32_t kNumRingTokens,
    uint32_t kNumSFRingTokens,
    uint32_t kNumStages,
    uint32_t kNumBytesPerPull,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    float kActivationAlpha,
    float kActivationBeta,
    bool kFastMath,
    typename weight_dtype_t,
    bool kHasShared = (kNumSharedExperts > 0),
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks,
    uint32_t kNumRingBlocks = kNumRingTokens / BLOCK_M,
    uint32_t kNumSharedSFTokens = layout::get_num_max_shared_sf_tokens(kNumMaxTokensPerRank),
    typename task_info_t = sched::TaskInfo<kHasShared>
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm100_fp8_fp4_mega_moe_impl(void* y,
                            int* cumulative_local_expert_recv_stats,
                            const uint32_t num_tokens,
                            const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_output,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights_sf) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator2Sm;

    // Template checks
    DG_STATIC_ASSERT(kNumDispatchThreads % 128 == 0, "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128, "Invalid number of MMA non-epilogue threads");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of MMA epilogue and combine threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");

    // Thread indices
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_weights_sf);
    }

    // Workspaces and Buffer
    const auto buffer = layout::MegaMoEBuffer(
        sym_buffer.get_base_ptr(),
        kHidden, kIntermediateHidden,
        kNumRanks, kNumExperts,
        kNumMaxTokensPerRank, kNumTopk,
        kNumRingTokens, kNumSFRingTokens,
        /*with_sf=*/ true,
        kNumSharedExperts
    );
    const auto workspace = buffer.workspace;

    using L2KBlockDependency = sched::L2KBlockDependency<L1_SHAPE_N, BLOCK_N, BLOCK_K>;

    // SF and its buffer configs
    constexpr uint32_t kGranK = 32;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    DG_STATIC_ASSERT(SF_BLOCK_M == math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems), "Invalid SF_BLOCK_M");
    DG_STATIC_ASSERT(SF_BLOCK_N == BLOCK_N, "No padding is needed for SFB");

    // UTCCP 4x32 transpose index mapping within each 128-element group
    // W6 peer3 finish fold control block (tools/megamoe_p3fold_patch.py; fastpath_megamoe.p3fold_ctl gives its address)
    DG_STATIC_ASSERT(kNumRingBlocks <= layout::MegaMoESignals::kNumMaxRingBlocks - 4 * w6p3::kCtlWords,
                     "fold control block overlaps the ring");
    w6p3::Ctl* const w6p3_ctl = reinterpret_cast<w6p3::Ctl*>(
        workspace.get_l2_empty_count_ptr(0) + (layout::MegaMoESignals::kNumMaxRingBlocks - 4 * w6p3::kCtlWords));
    w6p3::C11* const w6c11 = reinterpret_cast<w6p3::C11*>(reinterpret_cast<char*>(w6p3_ctl) + w6p3::kC11Offset);
    const auto transform_sf_token_idx = [](const uint32_t& token_idx_in_expert) {
        const uint32_t idx = token_idx_in_expert % BLOCK_M;
        return token_idx_in_expert / BLOCK_M * SF_BLOCK_M +
               (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
    };

    // Data types
    // NOTES: activations and shared weights are FP8 (e4m3); routed weights may be FP8 or FP4 (e2m1)
    using a_dtype_t = cutlass::float_e4m3_t;
    using shared_b_dtype_t = cutlass::float_e4m3_t;
    constexpr bool kIsWeightFP8 = cute::is_same_v<weight_dtype_t, cutlass::float_e4m3_t>;
    DG_STATIC_ASSERT(kIsWeightFP8 or cute::is_same_v<weight_dtype_t, cutlass::detail::float_e2m1_unpacksmem_t>, "Invalid routed weight type");

    // MMA configs
    // NOTES: always swap A/B, 2-CTA MMA, and matrices are K-major
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;
    constexpr uint32_t UMMA_N = BLOCK_M;  // Swap AB
    constexpr uint32_t UMMA_BLOCK_K = 128;
    constexpr uint32_t UMMA_K = 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2;  // Multicast on A
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_M % 16 == 0, "Invalid block M");
    DG_STATIC_ASSERT(BLOCK_N == LAYOUT_AD_M, "Invalid block N");

    // Swizzle configs
    constexpr uint32_t kSwizzleAMode = 128;
    constexpr uint32_t kSwizzleBMode = 128;
    constexpr uint32_t kSwizzleCDMode = 128;
    DG_STATIC_ASSERT(BLOCK_N % kSwizzleCDMode == 0, "Invalid block N");

    // Epilogue configs
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;

    // Shared memory
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    // Scheduler configs
    constexpr uint32_t kNumScheduleStages = 2;
    constexpr uint32_t kNumScheduleConsumerThreads = 2 * kNumEpilogueThreads;

    // Shared memory sizes
    // NOTES: FP8 CD output for L1 (2 TMA stages, BLOCK_N/2 post-SwiGLU), BF16 output for L2 (no TMA, a single stage)
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;
    constexpr uint32_t AMAX_REDUCTION_WARP_BUFFER_SIZE = STORE_BLOCK_M / 2; // float2

    struct SharedStorage {
        alignas(kSharedMemoryAlignment) uint32_t expert_token_count[kNumExperts];
        alignas(kSharedMemoryAlignment) uint8_t dispatch_send_buffer[kNumDispatchWarps][kNumBytesPerPull];
        union {
            alignas(kSharedMemoryAlignment) cutlass::float_e4m3_t l1[kNumEpilogueWarpgroups][kNumTMAStoreStages][STORE_BLOCK_M * L1_OUT_BLOCK_N];
            alignas(kSharedMemoryAlignment) nv_bfloat16 l2[kNumEpilogueWarpgroups][STORE_BLOCK_M * BLOCK_N];
        } smem_d;
        alignas(kSharedMemoryAlignment) a_dtype_t smem_a[kNumStages][LOAD_BLOCK_M * BLOCK_K];
        alignas(kSharedMemoryAlignment) weight_dtype_t smem_b[kNumStages][LOAD_BLOCK_N * BLOCK_K];
        uint32_t smem_sfa[kNumStages][SF_BLOCK_M * (BLOCK_K / 128)];
        uint32_t smem_sfb[kNumStages][SF_BLOCK_N * (BLOCK_K / 128)];
        float2 amax_reduction[kNumEpilogueWarps][AMAX_REDUCTION_WARP_BUFFER_SIZE];
        task_info_t task_infos[kNumScheduleStages];
        Barrier dispatch_barriers[kNumDispatchWarps];
        Barrier full_barriers[kNumStages];
        Barrier empty_barriers[kNumStages];
        Barrier tmem_full_barriers[kNumEpilogueStages];
        Barrier tmem_empty_barriers[kNumEpilogueStages];
        Barrier combine_barriers[kNumEpilogueWarps * 2];
        Barrier task_info_full_barriers[kNumScheduleStages];
        Barrier task_info_empty_barriers[kNumScheduleStages];
        uint32_t tmem_ptr_in_smem;
    };
    constexpr uint32_t kNumReusableSmemBytes = offsetof(SharedStorage, dispatch_barriers);
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage*>(smem_buffer);
    // W6 peer3 fold done hint: [0..1] the epilogue's %gridid once it reaches the cleanup barrier, [2] seen
    uint32_t* const w6p3_smem = reinterpret_cast<uint32_t*>(shared_storage.dispatch_send_buffer[0]);

    // Send buffers
    constexpr auto pull_layout = layout::Data(kNumBytesPerPull);
    const auto smem_send_buffers = layout::Buffer(
        pull_layout, kNumDispatchWarps, 1,
        static_cast<void*>(shared_storage.dispatch_send_buffer));

    // Tensor memory size
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // A cluster sync is essential for 2CTA tensor memory allocation
    comm::cluster_sync_with_relaxed_arrive();

    // Initialization
    if (warp_idx == 0) {
        // Clean shared memory
        if (cute::elect_one_sync()) {
            // The bytes must be 8 bytes aligned
            ptx::st_shared_bulk(
                shared_storage.expert_token_count,
                math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment)
            );
        }
    } else if (warp_idx == 1) {
        // Init m-barriers for dispatch
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            shared_storage.dispatch_barriers[i].init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                // Arrive at 2 CTAs, A + B
                shared_storage.full_barriers[i].init(2 * 2);
                shared_storage.empty_barriers[i].init(1);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
                // Arrive at all CTAs
                shared_storage.tmem_full_barriers[i].init(1);
                // Arrive only at the leader CTA
                shared_storage.tmem_empty_barriers[i].init(2 * kNumEpilogueThreads);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                shared_storage.combine_barriers[i].init(1);
            #pragma unroll
            for (uint32_t i = 0; i < kNumScheduleStages; ++ i) {
                shared_storage.task_info_full_barriers[i].init(1);
                shared_storage.task_info_empty_barriers[i].init(kNumScheduleConsumerThreads);
            }
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 3) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, &shared_storage.tmem_ptr_in_smem);
    }
    cute::cluster_sync();

    // Early shared-weight issue: only B and the scheduling warp may start
    // before the predecessor completes. They access immutable descriptors and
    // CTA-local metadata. Activation, dispatch, epilogue and peer reads wait.
    const bool early_shared_weights = kHasShared and kNumRanks == 1 and
        (kNumExperts % 256) != 0 and BLOCK_M == 16 and num_tokens <= 16;
    if (not early_shared_weights or
        (warp_idx != kNumDispatchWarps + 1 and warp_idx != kNumDispatchWarps + 3))
        cudaGridDependencySynchronize();

    // Task scheduler
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank,
        kNumSMs, kNumRanks,
        kNumRingBlocks,
        kNumSharedExperts>(
            workspace,
            shared_storage.task_info_full_barriers,
            shared_storage.task_info_empty_barriers,
            shared_storage.task_infos
    );

    // MEGAMOE_FAST_DISPATCH: single rank, decode-sized call (see tools/megamoe_fd_patch.py)
    constexpr bool kFastDispatchPossible = kNumRanks == 1 and (kNumExperts % 256) != 0;
    const bool fast_dispatch = kFastDispatchPossible and num_tokens <= 64;
    // All-token head: these descriptors are prepared by the predecessor before
    // griddepcontrol.wait completes. One block per expert, with original counts.
    const bool allt_head = kHasShared and fast_dispatch and BLOCK_M == 16 and num_tokens <= 16;
    const auto allt_slot = [&](const uint32_t token, const uint32_t expert) {
        uint32_t slot = kNumTopk;
        if (token < num_tokens) {
            #pragma unroll
            for (uint32_t k = 0; k < kNumTopk; ++ k)
                if (buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[token * kNumTopk + k] ==
                    static_cast<int64_t>(expert))
                    slot = k;
        }
        return slot;
    };
    uint32_t* const fd_counts = shared_storage.expert_token_count;  // [kNumExperts] = ready flag (alignment padding)
    if (fast_dispatch)
        scheduler.use_local_counts(fd_counts);

    // MMA pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Intra-SM Barrier indices
    constexpr uint32_t kDispatchBarrierIdx = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx = 3;

    // NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 2;

    // Adjust registers
    // NOTES: more experts per rank will cost more schedulers' registers
    constexpr bool kUseMoreEpilogueRegisters = kNumExpertsPerRank <= 64;
    constexpr uint32_t kNumDispatchRegisters = kUseMoreEpilogueRegisters ? 48 : 96;
    constexpr uint32_t kNumNonEpilogueRegisters = kUseMoreEpilogueRegisters ? 40 : 88;
    constexpr uint32_t kNumEpilogueRegisters = kUseMoreEpilogueRegisters ? 208 : 160;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    // Grid sync index assignments (dispatch and epilogue use separate counters to avoid conflicts)
    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    // Different warp roles
    if (warp_idx < kNumDispatchWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();
        // W6 peer3 fold: SM 0's cleanup waits and advances for armed calls (has[0] is reset only by that cleanup)
        const bool w6p3_armed_d = kNumRanks == 1 and w6p3_ctl->seg != 0 and w6p3::ld_volatile_s8(w6p3_ctl->has) != -1;
        const bool w6p3_armed_sm0 = w6p3_armed_d and sm_idx == 0;
        const bool w6p3_hint_d = w6p3_armed_d and w6p3_ctl->hint != 0;

        // Dispatch warps
        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            // TODO: figure out better unrolling
            // Now, `unroll` is better than `unroll 8`
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                // Allocate slots for each token-topk
                int expert_idx = -1;
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    expert_idx = static_cast<int>(
                        buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[i * kNumTopk + lane_idx]);
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };

        if (fast_dispatch) {
            // Every SM counts all the call's routes itself: no send-count atomics, grid syncs or barrier
            const int64_t* fd_topk = buffer.input_topk_idx_buffer.get_base_ptr<int64_t>();
            for (uint32_t i = thread_idx; i < num_tokens * kNumTopk; i += kNumDispatchThreads) {
                const int64_t e = fd_topk[i];
                if (e >= 0)
                    atomicAdd_block(fd_counts + e, 1);
            }
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
            if (thread_idx == 0) {
                __threadfence_block();
                *reinterpret_cast<volatile uint32_t*>(fd_counts + kNumExperts) = 1u;
            }
            // Combine readiness tags this launch's grid index, as the stock dispatch pushes it
            if (sm_idx == 0 and thread_idx < kNumRanks)
                *sym_buffer.map(workspace.get_peer_grid_idx_ptr(sym_buffer.rank_idx), thread_idx) = ptx::get_grid_idx() + 1;
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
        } else {
        // Count experts' tokens
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
           atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
        });
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Get SM offset (~6.5 us)
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(shared_storage.expert_token_count[i]);
            shared_storage.expert_token_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Write source indices (~2 us with 512 tokens)
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const auto dst_slot_idx = atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });

        // Grid sync
        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );

        // Write expert count
        if (sm_idx == 0) {
            // Push this launch's grid index to every peer for tagging combine readiness; +1 differs from a zeroed workspace
            DG_STATIC_ASSERT(kNumRanks <= kNumDispatchThreads, "Insufficient threads for the grid index push");
            if (thread_idx < kNumRanks)
                *sym_buffer.map(workspace.get_peer_grid_idx_ptr(sym_buffer.rank_idx), thread_idx) = ptx::get_grid_idx() + 1;
            __syncwarp();

            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_status & 0xffffffff;
                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Barrier before pulling
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* After the grid sync above, there is no more writes by other SMs (except 0) */ false,
            /* After the NVLink barrier, there is a grid sync */ true
        );
        }

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Pull token data and SF from remote ranks into local L1 buffer
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = &shared_storage.dispatch_barriers[warp_idx];

        // Per-rank counts for current expert (re-loaded when expert changes)
        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        // Wait token data arrival
        scheduler.fetch_expert_recv_count();

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        if (not allt_head)
        for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
            // Advance expert until within the range
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++ current_expert_idx >= kNumExpertsPerRank)
                    break;

                // Update pool block offset for the new expert
                expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);

                // Move start and end to the next expert
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
            }

            // Finish all tokens
            if (current_expert_idx >= kNumExpertsPerRank)
                break;

            // Load per-rank counts when expert changes
            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
                const uint32_t fd_num = fast_dispatch ? scheduler.get_num_tokens(current_expert_idx) : 0u;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    // TODO: this is not coalesced
                    stored_rank_count[i] = j < kNumRanks ? (fast_dispatch ? fd_num :
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx))) : 0;
                }
            }

            // Round-robin rank selection via iterative min-peeling
            uint32_t current_rank_in_expert_idx;
            uint32_t remaining[kNumRanksPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                remaining[i] = stored_rank_count[i];
            uint32_t offset = 0;
            uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t slot_idx = token_idx_in_expert;
            uint32_t token_idx_in_rank;
            while (true) {
                // Compute active count and min across all ranks
                // NOTES: reduce within each lane first, then warp-reduce once
                uint32_t num_actives_in_lane = 0;
                uint32_t min_in_lane = 0xffffffff;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    num_actives_in_lane += remaining[i] > 0;
                    if (remaining[i] > 0)
                        min_in_lane = cute::min(min_in_lane, remaining[i]);
                }
                const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                // Hit in the current round
                const uint32_t num_round_tokens = length * num_active_ranks;
                if (slot_idx < num_round_tokens) {
                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t num_seen_ranks = 0;
                    current_rank_in_expert_idx = 0;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                        const uint32_t num_active_lanes = __popc(mask);
                        if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                            current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                        num_seen_ranks += num_active_lanes;
                    }
                    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                    break;
                }

                // Move into the next round
                slot_idx -= num_round_tokens;
                offset += length;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] -= cute::min(remaining[i], length);
            }

            // Read source token-topk index (written by remote dispatch via NVLink)
            uint32_t src_token_topk_idx;
            if (fast_dispatch) {
                // The (token_idx_in_expert)-th route of this expert in (token, slot) order
                const int64_t* fd_topk = buffer.input_topk_idx_buffer.get_base_ptr<int64_t>();
                const uint32_t fd_n = num_tokens * kNumTopk;
                uint32_t fd_seen = 0;
                src_token_topk_idx = 0;
                for (uint32_t c = 0; c < fd_n; c += 32) {
                    const uint32_t i = c + lane_idx;
                    const bool hit = i < fd_n and fd_topk[i] == static_cast<int64_t>(current_expert_idx);
                    const uint32_t m = __ballot_sync(0xffffffffu, hit);
                    const uint32_t cnt = __popc(m);
                    if (token_idx_in_expert < fd_seen + cnt) {
                        src_token_topk_idx = c + (__fns(m, 0, static_cast<int>(token_idx_in_expert - fd_seen + 1)));
                        break;
                    }
                    fd_seen += cnt;
                }
            } else {
                src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                    current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
            }
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx = src_token_topk_idx % kNumTopk;

            // Hidden bytes are divided into chunks
            constexpr uint32_t kNumChunks = kHidden / kNumBytesPerPull;
            DG_STATIC_ASSERT(kNumChunks * kNumBytesPerPull == kHidden, "kNumBytesPerPull must divide hidden");

            // TMA load token from remote rank and store into local
            const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            const uint32_t pool_block_idx = pool_token_idx / BLOCK_M;

            // Wait for ring buffer slot to be available (previous consumer must have finished all N blocks)
            constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
            const auto l1_empty_count_target = (pool_block_idx / kNumRingBlocks) * kNumL1BlockNs;
            if (l1_empty_count_target > 0) {
                const auto empty_ptr = workspace.get_l1_empty_count_ptr(pool_block_idx % kNumRingBlocks);
                while (ptx::ld_acq(empty_ptr) < l1_empty_count_target);
            }

            const auto src_base_ptr = sym_buffer.map(
                buffer.input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(), current_rank_in_expert_idx);
            const auto dst_base_ptr = buffer.l1_token_buffer.get_data_buffer(pool_token_idx % kNumRingTokens).get_base_ptr();
            const auto issue_and_wait_pull_store = [&](const uint32_t& i) {
                ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                ptx::tma_store_1d(
                    math::advance_ptr(dst_base_ptr, i * kNumBytesPerPull),
                    pull_buffer.get_base_ptr(), kNumBytesPerPull
                );
                cute::tma_store_arrive();
                ptx::tma_store_wait<0>();
            };
            if (cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumChunks; ++ i) {
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        math::advance_ptr(src_base_ptr, i * kNumBytesPerPull),
                        pull_mbarrier, kNumBytesPerPull
                    );
                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kNumBytesPerPull);
                    i != (kNumChunks - 1) ? issue_and_wait_pull_store(i) : void();
                }
            }
            __syncwarp();

            // Load the weight first, so that its remote latency overlaps with the SF copy below
            const auto weight = *sym_buffer.map(
                buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                current_rank_in_expert_idx);

            // Load and store SF (overlaps with last chunk's TMA load from remote)
            constexpr uint32_t kNumSFUint32 = kHidden / 128;
            DG_STATIC_ASSERT(kNumSFUint32 > 0 and kHidden % 128 == 0, "Invalid SF");
            const auto remote_sf_ptr = sym_buffer.map(
                buffer.input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(),
                current_rank_in_expert_idx);
            const auto local_sf_ptr = buffer.l1_sf_buffer.get_base_ptr<uint32_t>();
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t token_idx_in_block = token_idx_in_expert % BLOCK_M;
            const auto sf_ring_token_idx = ring_block_idx * SF_BLOCK_M +
                transform_sf_token_idx(token_idx_in_block);
            #pragma unroll
            for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFUint32, 32u); ++ i) {
                const uint32_t j = i * 32 + lane_idx;
                if (j < kNumSFUint32)
                    local_sf_ptr[j * kNumSFRingTokens + sf_ring_token_idx] = remote_sf_ptr[j];
            }
            __syncwarp();

            // Store weights and metadata
            if (cute::elect_one_sync()) {
                *buffer.l1_topk_weights_buffer.get_data_buffer(pool_token_idx % kNumRingTokens).template get_base_ptr<float>() = weight;

                // Write source metadata for combine write-back (logical pool token)
                *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                    {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                // Complete last chunk's store
                issue_and_wait_pull_store(kNumChunks - 1);
                const bool is_last_token = (token_idx == expert_end_idx - 1);
                ptx::red_add_rel(
                    workspace.get_l1_full_count_ptr(pool_block_idx % kNumRingBlocks), 
                    is_last_token ? BLOCK_M - (token_idx_in_expert % BLOCK_M) : 1u
                );
            }
            __syncwarp();
        }

        // W6 peer3 fold done hint: poll while idle, until done or this SM's epilogue arrives at the barrier below
        if (w6p3_hint_d and warp_idx == 0) {
            if (lane_idx == 0) {
                const char* seg = reinterpret_cast<const char*>(w6p3_ctl->seg);
                const uint32_t tag = w6p3::call_tag(seg, w6p3_ctl);
                const uint64_t me = ptx::get_grid_idx();
                const long long max_spins = static_cast<long long>(w6p3_ctl->max_spins);
                long long spins = 0;
                uint64_t done = w6p3::ld_acquire_sys_u64(seg + w6p3_ctl->off_done);
                while (done != tag and spins < max_spins and w6p3::ld_volatile_shared_u64(w6p3_smem) != me) {
                    __nanosleep(128);
                    done = w6p3::ld_acquire_sys_u64(seg + w6p3_ctl->off_done);
                    ++ spins;
                }
                // 1 = acknowledged, 2 = spin bound exhausted (the call's cold adds are skipped), 0 = epilogue first
                reinterpret_cast<volatile uint32_t*>(w6p3_smem)[2] = done == tag ? 1u : (spins >= max_spins ? 2u : 0u);
            }
            __syncwarp();
        }

        // Clean workspace for the next usage, and also do cumulative stats
        // NOTES: it is overlapped with combine reduction epilogue
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            // SM 0: clear expert send count and schedule task counters
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
            if (warp_idx == 0 and cute::elect_one_sync()) {
                *workspace.get_l1_task_count_ptr() = 0;
                *workspace.get_l2_task_count_ptr() = 0;
                *workspace.get_shared_l1_task_count_ptr() = 0;
                *workspace.get_shared_l2_task_count_ptr() = 0;
            }
            __syncwarp();
            for (uint32_t i = thread_idx; i < workspace.num_shared_l2_pool_blocks; i += kNumDispatchThreads)
                *workspace.get_shared_l2_full_count_ptr(i) = 0;
            __syncwarp();
            if (w6p3_armed_sm0 and warp_idx == 0) {
                // Every SM read has[] before the epilogue grid sync; reset it for the next call
                if (lane_idx < 16)
                    reinterpret_cast<uint32_t*>(w6p3_ctl->has)[lane_idx] = 0xFFFFFFFFu;
                // The call's one statistics wait (as the finish's CTA 0), then the call counter advances
                if (lane_idx == 0) {
                    const char* seg = reinterpret_cast<const char*>(w6p3_ctl->seg);
                    w6p3::wait_done(seg, w6p3_ctl, w6p3::call_tag(seg, w6p3_ctl),
                                    reinterpret_cast<w6p3::GbStats*>(w6p3_ctl->stats));
                    uint32_t* state = reinterpret_cast<uint32_t*>(w6p3_ctl->state);
                    state[0] = w6p3::next_counter(state[0]);
                }
                __syncwarp();
            }
        } else {
            // Other SMs: clean blocks
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                // Read expert token count before clearing
                const auto num_recv_tokens = fast_dispatch ? scheduler.get_num_tokens(i) : static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);

                // Compute expert pool block offset
                expert_pool_block_offset = scheduler.get_pool_block_offset(i);

                // Wait read count ready
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                // Clean expert token count, and add cumulative results
                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if (warp_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    __syncwarp();
                }

                // Clean per-rank token count
                for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0;
                __syncwarp();

                // Clean L1 and L2 full stuffs and ring buffer counts
                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_full_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l1_empty_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_full_mask_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_empty_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                }
                __syncwarp();
            }
        }

        // Wait for all ranks to finish cleaning (single-rank fast path: kernel completion orders the next launch)
        if (not fast_dispatch)
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* Before the NVLink barrier, there is a grid sync */ true,
            /* At the end of kernel does not need to sync */ false
        );
    } else if (warp_idx == kNumDispatchWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for tokens with SFA
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            if (allt_head and not task_info.is_shared())
                task_info.valid_m = num_tokens;
            const auto tensor_map_a_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? (allt_head ? &tensor_map_shared_l1_acts : &tensor_map_l1_acts) :
                                          task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_acts :
                                          task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_acts :
                                        /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_acts;
            const auto tensor_map_sfa_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? (allt_head ? &tensor_map_shared_l1_acts_sf : &tensor_map_l1_acts_sf) :
                                            task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_acts_sf :
                                            task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_acts_sf :
                                          /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_acts_sf;
            const auto num_k_blocks = math::ceil_div(task_info.shape_k, BLOCK_K);

            // Compute pool block offset for this expert
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;

            // Wait the entire token arrival
            if (task_info.block_phase == sched::BlockPhase::Linear1 and not allt_head) {
                const auto ptr = workspace.get_l1_full_count_ptr(block_idx);
                const auto num_expected_tokens = BLOCK_M * (pool_block_idx / kNumRingBlocks + 1);
                while (ptx::ld_acq(ptr) != num_expected_tokens);
            } else if (task_info.block_phase == sched::BlockPhase::SharedLinear2) {
                const auto ptr = workspace.get_shared_l2_full_count_ptr(block_idx);
                const auto num_expected_blocks = (SHARED_L2_SHAPE_K / BLOCK_N) * 2;
                while (ptx::ld_acq(ptr) != num_expected_blocks);
            }

            L2KBlockDependency l2_k_block_dependency(workspace.get_l2_full_mask_ptr(ring_block_idx), pool_block_idx / kNumRingBlocks);
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                if (task_info.block_phase == sched::BlockPhase::Linear2)
                    l2_k_block_dependency.wait(k_block_idx);

                // Wait consumer release
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                // Compute token offsets from block index
                uint32_t m_idx = allt_head and task_info.block_phase == sched::BlockPhase::Linear1 ? 0u : block_idx * BLOCK_M;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                const uint32_t sfa_m_idx = allt_head and task_info.block_phase == sched::BlockPhase::Linear1 ? 0u : block_idx * SF_BLOCK_M;
                uint32_t sfa_k_idx = k_block_idx * (BLOCK_K / 128);

                // Add 2 CTA offsets for non-leader CTA
                if (not is_leader_cta)
                    m_idx += task_info.get_umma_aligned_valid_m() / 2;

                // TMA copy tokens and SFA, then arrive at full barrier
                if (cute::elect_one_sync()) {
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_a[stage_idx], k_idx, m_idx, 2);
                    tma::copy<SF_BLOCK_M, 1, 0>(
                        tensor_map_sfa_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfa[stage_idx], sfa_m_idx, sfa_k_idx, 2);
                    if (is_leader_cta) {
                        shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(sizeof(SharedStorage::smem_a[0]) * 2 + sizeof(SharedStorage::smem_sfa[0]) * 2);
                    } else {
                        shared_storage.full_barriers[stage_idx].arrive(0u);
                    }
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 1) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for weights with SF
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            if (allt_head and not task_info.is_shared())
                task_info.valid_m = num_tokens;
            const auto tensor_map_b_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_weights :
                                          task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights :
                                          task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_weights :
                                        /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_weights;
            const auto tensor_map_sfb_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_weights_sf :
                                            task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights_sf :
                                            task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_weights_sf :
                                          /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_weights_sf;

            const auto shape_k = task_info.shape_k;
            const auto shape_n = task_info.shape_n;
            const auto shape_sfb_k = math::ceil_div(shape_k, kGranK * 4u);
            const auto n_block_idx = task_info.n_cluster_idx * 2 + (is_leader_cta ? 0u : 1u);
            const auto num_k_blocks = math::ceil_div(shape_k, BLOCK_K);

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                // Compute weight offset
                uint32_t n_idx = task_info.is_shared() ? n_block_idx * BLOCK_N : task_info.local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                uint32_t sfb_k_idx = task_info.is_shared() ? k_block_idx * (BLOCK_K / 128) : task_info.local_expert_idx * shape_sfb_k + k_block_idx * (BLOCK_K / 128);

                // TMA copy weights with SF
                if (cute::elect_one_sync()) {
                    if (task_info.is_shared()) {
                        tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, shared_b_dtype_t>(
                            tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx], reinterpret_cast<shared_b_dtype_t*>(shared_storage.smem_b[stage_idx]), k_idx, n_idx, 2);
                        tma::copy<BLOCK_N, 1, 0>(
                            tensor_map_sfb_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2);
                        if (is_leader_cta) {
                            shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(sizeof(SharedStorage::smem_b[0]) * 2 + sizeof(SharedStorage::smem_sfb[0]) * 2);
                        } else {
                            shared_storage.full_barriers[stage_idx].arrive(0u);
                        }
                    } else {
                        tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, weight_dtype_t>(
                            tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_b[stage_idx], k_idx, n_idx, 2);
                        tma::copy<BLOCK_N, 1, 0>(
                            tensor_map_sfb_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2);
                        if (is_leader_cta) {
                            constexpr uint32_t kNumWeightBytes = sizeof(SharedStorage::smem_b[0]) * 2 /
                                (kIsWeightFP8 ? 1 : 2);
                            shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(kNumWeightBytes + sizeof(SharedStorage::smem_sfb[0]) * 2);
                        } else {
                            shared_storage.full_barriers[stage_idx].arrive(0u);
                        }
                    }
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 2) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM MMA issue warp (only the leader CTA will run)
        if (is_leader_cta) {
            // Make instruction descriptor with block scaling
            // NOTES: always swap A/B
            auto routed_instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                    weight_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,
                    UMMA_M, UMMA_N,
                    cute::UMMA::Major::K, cute::UMMA::Major::K
                >();
            auto shared_instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                shared_b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,
                UMMA_M, UMMA_N,
                cute::UMMA::Major::K, cute::UMMA::Major::K
            >();
            auto sf_desc = mma::sm100::make_sf_desc(nullptr);

            DG_STATIC_ASSERT(sizeof(weight_dtype_t) == sizeof(shared_b_dtype_t), "Weight SMEM descriptors must use identical addressing");
            DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, UMMA_BLOCK_K, kSwizzleAMode>(shared_storage.smem_a[0], 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, UMMA_BLOCK_K, kSwizzleBMode>(shared_storage.smem_b[0], 0, 0);
            const uint32_t a_desc_lo = a_desc.lo;
            const uint32_t b_desc_lo = b_desc.lo;

            // Checks for MMA instructions
            DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                             "Invalid MMA instruction shape");

            // Persistently schedule over blocks
            uint32_t current_iter_idx = 0;
            task_info_t task_info;
            while (scheduler.get_next_task(task_info)) {
            if (allt_head and not task_info.is_shared())
                task_info.valid_m = num_tokens;
                const auto num_k_blocks = task_info.shape_k / BLOCK_K;

                // Dynamic update of UMMA N based on effective M
                auto& instr_desc = task_info.is_shared() ? shared_instr_desc : routed_instr_desc;
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, task_info.get_umma_aligned_valid_m());

                // Wait tensor memory empty barrier arrival
                const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
                const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
                shared_storage.tmem_empty_barriers[accum_stage_idx].wait(accum_phase ^ 1);
                ptx::tcgen05_after_thread_sync();

                // Empty barrier arrival
                auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                    auto umma_arrive = [](const uint64_t* barrier) {
                        constexpr uint16_t kCTAMask = (1 << 2) - 1;
                        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                    };
                    umma_arrive(reinterpret_cast<uint64_t*>(&shared_storage.empty_barriers[stage_idx]));

                    // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                    if (do_tmem_full_arrive)
                        umma_arrive(reinterpret_cast<uint64_t*>(&shared_storage.tmem_full_barriers[accum_stage_idx]));
                    __syncwarp();
                };

                // Launch MMAs
                #pragma unroll 2
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    // Wait TMA load completion
                    shared_storage.full_barriers[stage_idx].wait(phase);
                    ptx::tcgen05_after_thread_sync();

                    const uint32_t a_desc_base_lo = a_desc_lo + stage_idx * sizeof(SharedStorage::smem_a[0]) / 16;
                    const uint32_t b_desc_base_lo = b_desc_lo + stage_idx * sizeof(SharedStorage::smem_b[0]) / 16;
                    if (cute::elect_one_sync()) {
                        #pragma unroll
                        for (uint32_t umma_k_block_idx = 0; umma_k_block_idx < BLOCK_K / UMMA_BLOCK_K; ++ umma_k_block_idx) {
                            // UTCCP copy SFA and SFB to TMEM
                            using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_2cta;
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                                auto smem_ptr = shared_storage.smem_sfa[stage_idx] + umma_k_block_idx * SF_BLOCK_M + i * kNumUTCCPAlignedElems;
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                            }
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                                auto smem_ptr = shared_storage.smem_sfb[stage_idx] + umma_k_block_idx * SF_BLOCK_N + i * kNumUTCCPAlignedElems;
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                            }

                            // Issue UMMA
                            #pragma unroll
                            for (uint32_t k = 0; k < UMMA_BLOCK_K / UMMA_K; ++ k) {
                                const auto runtime_instr_desc =
                                    mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k, k);
                                a_desc.lo = mma::sm100::advance_umma_desc_lo<
                                    cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_desc_base_lo, umma_k_block_idx * UMMA_BLOCK_K * LOAD_BLOCK_M * sizeof(a_dtype_t), k * UMMA_K);
                                b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                    cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, weight_dtype_t>(b_desc_base_lo, umma_k_block_idx * UMMA_BLOCK_K * LOAD_BLOCK_N * sizeof(weight_dtype_t), k * UMMA_K);
                                ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
                                    b_desc, a_desc, accum_stage_idx * UMMA_N,
                                    k_block_idx > 0 or umma_k_block_idx > 0 or k > 0, runtime_instr_desc,
                                    kTmemStartColOfSFB, kTmemStartColOfSFA);
                            }
                        }
                    }
                    __syncwarp();

                    // Commit to the mbarrier object
                    // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                    empty_barrier_arrive(k_block_idx == num_k_blocks - 1);
                }
            }

            // To safely deconstruct barriers, we need another round of waits
            if (current_iter_idx > 0) {
                const auto accum_phase_idx = ((current_iter_idx - 1) / kNumEpilogueStages) & 1;
                shared_storage.tmem_empty_barriers[(current_iter_idx - 1) % kNumEpilogueStages].wait(accum_phase_idx);
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 3) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // Do mainloop by the leader CTA
        if (is_leader_cta)
            scheduler.mainloop(num_tokens);
    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&shared_storage.tmem_ptr_in_smem) == 0);

        // GEMM epilogue warps
        const auto epilogue_warp_idx = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const auto epilogue_wg_idx = epilogue_warp_idx / 4;
        const auto epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const auto warp_idx_in_wg = epilogue_warp_idx % 4;
        DG_STATIC_ASSERT((kNumDispatchWarps + kNumMMANonEpilogueWarps) % 4 == 0 and
                         kNumEpilogueWarps % 4 == 0, "Invalid epilogue warps");

        // TODO: support effective block M
        // NOTES:
        //  - 2 warpgroups divide the whole BM into BM / 2
        //  - 4 warps divide the whole BN into BN / 4
        //  - BM / 2 is further divided into stored blocks, i.e. with `STORE_BLOCK_M` size
        //  - `STORE_BLOCK_M` in further divided into `ATOM_M`
        constexpr uint32_t WG_BLOCK_M = BLOCK_M / kNumEpilogueWarpgroups;
        constexpr uint32_t ATOM_M = 8;
        constexpr uint32_t kNumBankGroupBytes = 16u;
        constexpr uint32_t kNumAtomsPerStore = STORE_BLOCK_M / ATOM_M;
        DG_STATIC_ASSERT(BLOCK_M % kNumEpilogueWarpgroups == 0, "Invalid block M");
        DG_STATIC_ASSERT(WG_BLOCK_M % STORE_BLOCK_M == 0, "Invalid warpgroup block M");
        DG_STATIC_ASSERT(STORE_BLOCK_M % ATOM_M == 0, "Invalid store block M");
        DG_STATIC_ASSERT(BLOCK_N == 128, "Invalid block N");

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Persistently schedule over blocks
        uint32_t current_iter_idx = 0;
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            if (allt_head and not task_info.is_shared())
                task_info.valid_m = num_tokens;
            // Wait UMMA arrival
            const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
            const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
            shared_storage.tmem_full_barriers[accum_stage_idx].wait(accum_phase);
            ptx::tcgen05_after_thread_sync();

            // Now we can release the task
            scheduler.release_task_info();

            // Compute offsets
            // NOTES: use shuffle here to let NVCC know warp divergence won't happen
            const uint32_t valid_m = ptx::exchange(task_info.valid_m, 0);
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;
            const uint32_t ring_m_idx = ring_block_idx * BLOCK_M;  // Ring-buffer offset for reusable data buffers
            const uint32_t m_idx = block_idx * BLOCK_M;
            const uint32_t pool_m_idx = pool_block_idx * BLOCK_M;  // Full-pool offset for non-ring metadata
            const uint32_t n_block_idx = task_info.n_cluster_idx * 2 + (is_leader_cta ? 0u : 1u);
            uint32_t n_idx = n_block_idx * BLOCK_N;

            if (task_info.block_phase == sched::BlockPhase::Linear1 or task_info.block_phase == sched::BlockPhase::SharedLinear1) {
                if (not task_info.is_shared()) {
                    // Wait L2 block empty
                    const auto l2_empty_ptr = workspace.get_l2_empty_count_ptr(ring_block_idx);
                    const auto num_expected_blocks = (L2_SHAPE_N / BLOCK_N) * (pool_block_idx / kNumRingBlocks);
                    while (ptx::ld_acq(l2_empty_ptr) != num_expected_blocks);
                }

                // Unified L1 epilogue: SwiGLU in-place using granularity 8 interleaved weights
                // With `SM100_TMEM_LOAD_16dp256b1x`, gate/up pairs are:
                float stored_cached_weight = 1.0f;

                #pragma unroll
                for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                    // Early break if the entire store block is beyond the valid token range
                    if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {
                        ptx::tcgen05_before_thread_sync();
                        shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        break;
                    }

                    // Iterate all atoms in the store block
                    float2 activation_values[kNumAtomsPerStore][2];
                    float2 amax_values[kNumAtomsPerStore];
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        const uint32_t j = s * kNumAtomsPerStore + i;

                        // Load weights from global into register cache per 32 tokens
                        DG_STATIC_ASSERT(32 % ATOM_M == 0, "Invalid block size");
                        if (not task_info.is_shared() and (j * ATOM_M) % 32 == 0 and
                            (WG_BLOCK_M % 32 == 0 or j * ATOM_M + lane_idx < WG_BLOCK_M)) {
                            if (allt_head) {
                                const uint32_t token = epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M + lane_idx;
                                const uint32_t slot = allt_slot(token, task_info.local_expert_idx);
                                stored_cached_weight = slot < kNumTopk ?
                                    buffer.input_topk_weights_buffer.get_base_ptr<float>()[token * kNumTopk + slot] : 0.0f;
                            } else {
                                stored_cached_weight = *buffer.l1_topk_weights_buffer
                                    .get_data_buffer(ring_m_idx + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M + lane_idx)
                                    .template get_base_ptr<float>();
                            }
                        }

                        // Load weights from register cache
                        const float2 weights = {
                            ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 0),
                            ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 1)
                        };

                        // Load from TMEM
                        uint2 raw_values[4];
                        uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M;
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                               raw_values[0].x, raw_values[0].y, raw_values[1].x, raw_values[1].y);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                               raw_values[2].x, raw_values[2].y, raw_values[3].x, raw_values[3].y);
                        cutlass::arch::fence_view_async_tmem_load();

                        // Signal tensor memory consumed on the last atom
                        if (j == WG_BLOCK_M / ATOM_M - 1) {
                            ptx::tcgen05_before_thread_sync();
                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        }

                        // Apply SwiGLU: gate * sigmoid(alpha * gate) * (up + beta)
                        auto fp32_values = reinterpret_cast<float2*>(raw_values);
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            auto bf16_gate = __float22bfloat162_rn(fp32_values[k * 2 + 0]);
                            auto bf16_up =   __float22bfloat162_rn(fp32_values[k * 2 + 1]);

                            // Clamp
                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                                bf16_gate = __hmin2(bf16_gate, {kActivationClamp, kActivationClamp});
                                bf16_up = __hmax2(bf16_up, {-kActivationClamp, -kActivationClamp});
                                bf16_up = __hmin2(bf16_up, {kActivationClamp, kActivationClamp});
                            }

                            // SwiGLU
                            const auto gate = __bfloat1622float2(bf16_gate);
                            auto sigmoid_input = gate;
                            if constexpr (kActivationAlpha != 1.0f)
                                sigmoid_input = __fmul2_rn(
                                    sigmoid_input, {kActivationAlpha, kActivationAlpha});
                            auto neg_gate_exp = make_float2(
                                kFastMath ? __expf(-sigmoid_input.x) : expf(-sigmoid_input.x),
                                kFastMath ? __expf(-sigmoid_input.y) : expf(-sigmoid_input.y));
                            const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
                            float2 gated;
                            if constexpr (kFastMath) {
                                gated = __fmul2_rn(
                                    gate, {math::fast_rcp(denom.x), math::fast_rcp(denom.y)});
                            } else {
                                gated = {gate.x / denom.x, gate.y / denom.y};
                            }
                            auto up = __bfloat1622float2(bf16_up);
                            if constexpr (kActivationBeta != 0.0f)
                                up = __fadd2_rn(up, {kActivationBeta, kActivationBeta});
                            activation_values[i][k] = __fmul2_rn(
                                __fmul2_rn(gated, up), weights);
                        }

                        // Amax reduction (thread-level)
                        float2 thread_local_amax = {0.f, 0.f};
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            thread_local_amax.x = fmaxf(thread_local_amax.x, fabsf(activation_values[i][k].x));
                            thread_local_amax.y = fmaxf(thread_local_amax.y, fabsf(activation_values[i][k].y));
                        }

                        // Amax reduction (warp-level)
                        amax_values[i].x = math::warp_reduce<4, true>(
                            thread_local_amax.x, math::ReduceMax<float>());
                        amax_values[i].y = math::warp_reduce<4, true>(
                            thread_local_amax.y, math::ReduceMax<float>());

                        // Reduce amax (warp-pair-level)
                        if (lane_idx < 4)
                            shared_storage.amax_reduction[epilogue_warp_idx][i * (ATOM_M / 2) + lane_idx] = amax_values[i];
                        __syncwarp();
                    }

                    // Wait shared memory release from previous TMA store
                    // And fence `shared_storage.amax_reduction`
                    const uint32_t tma_stage_idx = s % kNumTMAStoreStages;
                    ptx::tma_store_wait<kNumTMAStoreStages - 1>();
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Cast to FP8 E4M3 and store into shared memory
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        // Reduce amax (warp-pair-level)
                        const float2 wp_amax =
                            shared_storage.amax_reduction[epilogue_warp_idx ^ 1][i * (ATOM_M / 2) + lane_idx % 4];
                        amax_values[i].x = fmaxf(amax_values[i].x, wp_amax.x);
                        amax_values[i].y = fmaxf(amax_values[i].y, wp_amax.y);

                        // Calculate SF
                        const uint2 sf_exp = {math::get_ue8m0_sf_exp(amax_values[i].x),
                                              math::get_ue8m0_sf_exp(amax_values[i].y)};
                        const float2 sf_inv = {math::get_ue8m0_sf_inv<float>(sf_exp.x),
                                               math::get_ue8m0_sf_inv<float>(sf_exp.y)};

                        // Cast
                        const float2 upper = __fmul2_rn(activation_values[i][0], sf_inv);
                        const float2 lower = __fmul2_rn(activation_values[i][1], sf_inv);
                        const auto fp8x4_values = __nv_fp8x4_e4m3(make_float4(upper.x, upper.y, lower.x, lower.y));

                        // STSM
                        uint32_t row = lane_idx;
                        uint32_t col = warp_idx_in_wg;
                        const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l1[epilogue_wg_idx][tma_stage_idx])
                            + i * ATOM_M * L1_OUT_BLOCK_N
                            + row * L1_OUT_BLOCK_N
                            // Use 64B swizzle for SwiGLU, so divided by 2
                            + (col ^ (row / 2)) * kNumBankGroupBytes;
                        ptx::SM100_U8x4_STSM_T<__nv_fp8x4_e4m3>::copy(fp8x4_values, smem_ptr);

                        // Store SF to `buffer.l2_sf_buffer` as UE8M0 (MN-major layout)
                        // Only one warp per pair writes (both hold the same SF after cross-warp reduce)
                        // Each lane < 4 holds SF for 2 rows (sf.x and sf.y)
                        if (warp_idx_in_wg % 2 == 0 and lane_idx < 4) {
                            const uint32_t k_idx = n_block_idx * 2 + warp_idx_in_wg / 2;
                            const uint32_t k_uint_idx = k_idx / 4, byte_idx = k_idx % 4;
                            const uint32_t mn_stride = (task_info.is_shared() ? kNumSharedSFTokens : kNumSFRingTokens) * sizeof(uint32_t);
                            const auto sf_base_ptr = task_info.is_shared() ?
                                buffer.shared_l2_sf_buffer.get_base_ptr<uint8_t>() : buffer.l2_sf_buffer.get_base_ptr<uint8_t>();
                            // NOTES: consecutive tokens (t, t + 1) are in the same 32-group, so `sf_idx` differs by 4
                            // NOTES: originally there was:
                            //   - `const uint32_t token_idx_in_expert = task_info.m_block_idx * BLOCK_M + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M + lane_idx * 2
                            //   - `task_info.pool_block_idx * SF_BLOCK_M + transform_sf_token_idx(token_idx_in_expert)`
                            // We find out that
                            //   1. `task_info.m_block_idx * BLOCK_M` mod `BLOCK_M` is 0, and `epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M + lane_idx * 2` is always < `BLOCK_M`, so we can put `task_info.m_block_idx * BLOCK_M` outside
                            //   2. `lane_idx * 2` controls the lowest 3 bit of `token_idx_in_expert`, and `transform_sf_token_idx` is a bitwise-independent transformation if the input is less than `BLOCK_M`, so we can put `lane_idx * 2` outside
                            // This reduce the number of computation instructions.
                            const uint32_t token_base_idx = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;
                            __builtin_assume(token_base_idx < BLOCK_M);
                            const auto sf_token_idx = block_idx * SF_BLOCK_M
                                + transform_sf_token_idx(token_base_idx) + (lane_idx * 2) * 4;
                            const auto sf_addr = k_uint_idx * mn_stride + sf_token_idx * static_cast<uint32_t>(sizeof(uint32_t)) + byte_idx;
                            sf_base_ptr[sf_addr] = static_cast<uint8_t>(sf_exp.x);
                            sf_base_ptr[sf_addr + 4 * static_cast<uint32_t>(sizeof(uint32_t))] = static_cast<uint8_t>(sf_exp.y);
                        }
                        __syncwarp();
                    }
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Issue TMA store after all atoms in this store block
                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        const auto tensor_map_l1_output_ptr = task_info.is_shared() ? &tensor_map_shared_l1_output : &tensor_map_l1_output;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            tensor_map_l1_output_ptr,
                            shared_storage.smem_d.l1[epilogue_wg_idx][tma_stage_idx],
                            out_n_idx,
                            m_idx + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                }

                // Notify L2 and increment L1 empty count
                // TODO: less epilogue sync scope
                ptx::tma_store_wait<0>();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                    if (task_info.is_shared()) {
                        ptx::red_add_rel(
                            workspace.get_shared_l2_full_count_ptr(pool_block_idx), 1u);
                    } else {
                        L2KBlockDependency::arrive(workspace.get_l2_full_mask_ptr(ring_block_idx), n_block_idx);

                        // Increment L1 empty count for this physical slot (one per N block)
                        ptx::red_add(
                            workspace.get_l1_empty_count_ptr(ring_block_idx), 1u);
                    }
                }
                __syncwarp();
            } else {
                // Increment L2 empty count for this physical slot (one per N block)
                if (not task_info.is_shared()) {
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        ptx::red_add(
                            workspace.get_l2_empty_count_ptr(ring_block_idx), 1u);
                    }
                    __syncwarp();
                }

                DG_STATIC_ASSERT(STORE_BLOCK_M % 8 == 0, "Invalid store M");
                constexpr uint32_t kNumRowsPerWarp = STORE_BLOCK_M / 8;

                // L2 BF16 epilogue: write GEMM output to remote combine buffer via NVLink
                #pragma unroll
                for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                    // Early break if the entire store block is beyond the valid token range
                    // TODO: check performance
                    if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {
                        ptx::tcgen05_before_thread_sync();
                        shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        break;
                    }

                    // Read the source metadata of this warp's rows before the TMEM loads to overlap the latency
                    layout::TokenSrcMetadata cached_src_metadata[kNumRowsPerWarp];
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + j * 8 + warp_idx_in_wg * 2 + lane_idx / 16;
                        if (m_idx_in_block < valid_m)
                            cached_src_metadata[j] = task_info.is_shared() ?
                                layout::TokenSrcMetadata(sym_buffer.rank_idx, pool_m_idx + m_idx_in_block, kNumTopk) :
                                allt_head ? layout::TokenSrcMetadata(sym_buffer.rank_idx, m_idx_in_block,
                                    allt_slot(m_idx_in_block, task_info.local_expert_idx)) :
                                *workspace.get_token_src_metadata_ptr(pool_m_idx + m_idx_in_block);
                    }
                    __syncwarp();

                    #pragma unroll
                    for (uint32_t i = 0; i < STORE_BLOCK_M / ATOM_M; ++ i) {
                        // Load from TMEM using .16x256b shape to satisfy STSM layout requirements
                        // Start from lane index 0 and 16
                        uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;
                        uint32_t values[ATOM_M];
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                               values[0], values[1], values[2], values[3]);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                               values[4], values[5], values[6], values[7]);
                        cutlass::arch::fence_view_async_tmem_load();

                        // Wait shared memory release from previous NVLink store
                        // NOTES: skip for the first store block since the prior full barrier already ensures completion
                        if (i == 0 and s > 0)
                            ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                        // Signal tensor memory consumed
                        if (s == WG_BLOCK_M / STORE_BLOCK_M - 1 and i == STORE_BLOCK_M / ATOM_M - 1) {
                            ptx::tcgen05_before_thread_sync();
                            shared_storage.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                        }

                        // Store into shared memory
                        // NOTES: each lane provides its own address for stmatrix; 2 warps share a BF16 swizzle atom
                        uint32_t row = lane_idx % 8;
                        uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
                        const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l2[epilogue_wg_idx]) +
                            (warp_idx_in_wg / 2) * STORE_BLOCK_M * kSwizzleCDMode +
                            i * ATOM_M * kSwizzleCDMode +
                            row * (kNumBankGroupBytes * 8) +
                            (col ^ row) * kNumBankGroupBytes;
                        ptx::SM90_U32x4_STSM_T<uint32_t>::copy(
                            math::cast_into_bf16_and_pack(values[0], values[1]),
                            math::cast_into_bf16_and_pack(values[2], values[3]),
                            math::cast_into_bf16_and_pack(values[4], values[5]),
                            math::cast_into_bf16_and_pack(values[6], values[7]),
                            smem_ptr
                        );
                    }

                    // Wait shared memory ready
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Write into remote buffers
                    // Each warp writes 2 rows (lane_idx/16 splits the warp into two halves, one per row)
                    const uint32_t row_in_atom = (warp_idx_in_wg * 2 + lane_idx / 16) % ATOM_M;
                    const uint32_t bank_group_idx = lane_idx % 8;

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_store = j * 8 + warp_idx_in_wg * 2 + lane_idx / 16;
                        const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + row_in_store;

                        // Skip padding rows beyond the actual token count for this expert
                        if (m_idx_in_block >= valid_m)
                            break;

                        const auto& [dst_rank_idx, dst_token_idx, dst_topk_idx] = cached_src_metadata[j];
                        // An absent route must never overwrite any v11 slot,
                        // including the shared expert slot at kNumTopk.
                        if (allt_head and not task_info.is_shared() and dst_topk_idx == kNumTopk)
                            continue;

                        // Read from shared memory
                        const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l2[epilogue_wg_idx]) +
                            (lane_idx % 16 / 8) * STORE_BLOCK_M * kSwizzleCDMode +
                            row_in_store * kSwizzleCDMode +
                            (bank_group_idx ^ row_in_atom) * kNumBankGroupBytes;
                        const auto packed = ptx::ld_shared(reinterpret_cast<float4*>(smem_ptr));

                        // Write into remote
                        const auto dst_token = buffer.combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                               .get_data_buffer(dst_token_idx);
                        const auto dst_ptr = math::advance_ptr<float4>(
                            dst_token.get_base_ptr(),
                            n_idx * static_cast<uint32_t>(sizeof(nv_bfloat16)) + (lane_idx % 16) * static_cast<uint32_t>(sizeof(float4)));
                        *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                    }
                }

                // Ensure the next epilogue safe to use shared memory
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            }
        }

        // Deallocate tensor memory
        // NOTES: must be called by the same logical warp ID on both CTAs
        if (epilogue_warp_idx == 0)
            Allocator().free(0, kNumTmemCols);

        // Combine: reduce top-k results and write back
        // NOTES: reuse shared memory from start up to the barriers
        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        // 3 slots of chunk is needed: 2 load stages and 1 store
        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;

        // NOTES: either 1 or 2 chunks for simplicity
        // NOTES: Restrict on both smem and register
        constexpr uint32_t kNumChunks =
            kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= kNumReusableSmemBytes and kHidden <= 32 * kNumMaxRegistersForBuffer ? 1 : 2;
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / kNumChunks <= kNumReusableSmemBytes, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements (one per lane)");
        DG_STATIC_ASSERT(kNumTopk + (kNumSharedExperts > 0 ? 1u : 0u) <= 32u, "Top-k + shared must fit in a single warp");

        // Verify combined shared memory budget at runtime
        DG_DEVICE_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumChunkBytes <= kNumReusableSmemBytes);

        // Per-warp buffer: 2 stage load buffers + 1 store buffer
        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer  = math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);

        // Per-warp barriers
        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return &shared_storage.combine_barriers[i + epilogue_warp_idx * 2];
        });

        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;

        // Peers' grid indices for tagging their combine readiness; the load overlaps with the grid sync
        DG_STATIC_ASSERT(kNumRanks <= kNumEpilogueThreads, "Insufficient threads for combine readiness");
        uint64_t peer_grid_idx = 0;
        if (sm_idx == 0 and epilogue_thread_idx < kNumRanks)
            peer_grid_idx = *workspace.get_peer_grid_idx_ptr(epilogue_thread_idx);

        // W6 peer3 fold: has[0..63] -> cold mask, read before the grid sync (SM 0 resets has[] after it)
        uint64_t w6p3_mask = 0;
        bool w6p3_armed = false;
        if constexpr (kNumRanks == 1) {
            __syncwarp();
            if (w6p3_ctl->seg != 0) {
                const int h0 = w6p3::ld_volatile_s8(w6p3_ctl->has + lane_idx);
                const int h1 = w6p3::ld_volatile_s8(w6p3_ctl->has + 32 + lane_idx);
                w6p3_armed = __shfl_sync(0xffffffffu, h0, 0) != -1;
                w6p3_mask = static_cast<uint64_t>(__ballot_sync(0xffffffffu, h0 == 1)) |
                            (static_cast<uint64_t>(__ballot_sync(0xffffffffu, h1 == 1)) << 32);
            }
        }

        // W6 v11: every CTA reads skip_combine before the grid sync (SM 0 clears it after); the producer (W4's router)
        // stores 1 before the launch, ordered by this kernel's griddepcontrol.wait
        const bool w6c11_skip = kNumRanks == 1 and not w6p3_armed and w6p3::ld_volatile_u32(&w6c11->skip_combine) == 1u;

        // All local L2 writes are done after this grid sync
        comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>(
            workspace, sm_idx, epilogue_thread_idx,
            [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
        );

        // Notify remote ranks; ordered before the cleanup barrier by the dispatch/epilogue sync below
        if (sm_idx == 0 and epilogue_thread_idx < kNumRanks)
            ptx::st_rel_sys(sym_buffer.map(workspace.get_combine_ready_grid_idx_ptr(sym_buffer.rank_idx), epilogue_thread_idx), peer_grid_idx);

        // W6 peer3 fold done hint: tell this SM's polling dispatch lane that the epilogue has arrived
        const bool w6p3_hint = w6p3_armed and w6p3_ctl->hint != 0;
        if (w6p3_hint and epilogue_thread_idx == 0)
            *reinterpret_cast<volatile uint64_t*>(w6p3_smem) = ptx::get_grid_idx();

        // Barrier with dispatch warps, so that they can do clean workspace
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        const uint32_t w6p3_seen = w6p3_hint ? reinterpret_cast<volatile uint32_t*>(w6p3_smem)[2] : 0u;

        // W6 v11: self-clear skip_combine (every CTA read it before the grid sync) and export the slot layout
        if (sm_idx == 0 and epilogue_thread_idx == 0) {
            w6c11->skip_combine = 0u;
            w6c11->slot_base = reinterpret_cast<unsigned long long>(
                buffer.combine_token_buffer.get_rank_buffer(0).get_data_buffer(0).get_base_ptr());
            w6c11->slot_stride = buffer.combine_token_buffer.get_num_bytes_per_rank();
            w6c11->token_stride = kNumHiddenBytes;
            w6c11->magic = w6p3::kC11Magic;
        }

        // Iterate over all token chunks, 1 token 1 topk latency: ~3 us
        const auto grid_idx = ptx::get_grid_idx() + 1;
        if constexpr (kNumRanks == 1) {
            // W6 lane-parallel combine (tools/megamoe_fast_combine_patch.py)
            constexpr uint32_t kPieceChannels = 128;
            constexpr uint32_t kNumPieces = kHidden / kPieceChannels;
            constexpr uint32_t kNumSlots = kNumTopk + (kNumSharedExperts > 0 ? 1u : 0u);
            DG_STATIC_ASSERT(kNumPieces * kPieceChannels == kHidden, "fast combine: hidden must be a multiple of 128");
            const char* const w6p3_seg = reinterpret_cast<const char*>(w6p3_ctl->seg);
            const uint32_t w6p3_off_out = w6p3_armed ? w6p3_ctl->off_out : 0u;
            // seen: this SM's dispatch lane observed the acknowledgement (acquire), ordered by the cleanup barrier
            bool w6p3_waited = w6p3_seen != 0u, w6p3_ok = w6p3_seen == 1u;
            for (uint32_t item = epilogue_warp_idx * kNumSMs + sm_idx; not w6c11_skip and item < num_tokens * kNumPieces;
                 item += kNumSMs * kNumEpilogueWarps) {
                const uint32_t token_idx = item / kNumPieces;
                const uint32_t piece = item % kNumPieces;
                // W6 peer3 fold: a cold token's first item in this warp waits for the RTX (no statistics)
                const bool w6p3_cold = w6p3_armed and token_idx < 64 and ((w6p3_mask >> token_idx) & 1ull);
                if (w6p3_cold and not w6p3_waited) {
                    uint32_t ok = 0;
                    if (lane_idx == 0)
                        ok = w6p3::wait_done(w6p3_seg, w6p3_ctl, w6p3::call_tag(w6p3_seg, w6p3_ctl), nullptr) ? 1u : 0u;
                    w6p3_ok = __shfl_sync(0xffffffffu, ok, 0) != 0;
                    w6p3_waited = true;
                    __syncwarp();  // orders lane 0's acquire before every lane's segment reads
                }
                const bool w6p3_add = w6p3_cold and w6p3_ok;
                const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                    static_cast<int>(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[token_idx * kNumTopk + lane_idx]) :
                    (kNumSharedExperts > 0 and lane_idx == kNumTopk ? static_cast<int>(kNumTopk) : -1);
                const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);
                const bool is_routed = lane_idx < kNumTopk and stored_topk_slot_idx >= 0;
                const auto peer_ready_ptr = workspace.get_combine_ready_grid_idx_ptr(
                    is_routed ? static_cast<uint32_t>(stored_topk_slot_idx) / kNumExpertsPerRank : 0);
                comm::wait_until([&]() { return __all_sync(0xffffffff, not is_routed or ptx::ld_acq_sys(peer_ready_ptr) == grid_idx); },
                                 [&]() { printf("DeepGEMM combine peers timeout: rank=%u, token=%u\n", sym_buffer.rank_idx, token_idx); });
                // all present slots' 4 channels of this lane in flight at once
                const uint32_t byte_off = piece * kPieceChannels * static_cast<uint32_t>(sizeof(nv_bfloat16)) + lane_idx * 8u;
                uint2 w6p3_out = make_uint2(0u, 0u);
                if (w6p3_add)
                    w6p3_out = w6p3::ld_volatile_v2(w6p3_seg + w6p3_off_out +
                                                    static_cast<uint64_t>(token_idx) * kNumHiddenBytes + byte_off);
                uint2 v[kNumSlots];
                #pragma unroll
                for (uint32_t k = 0; k < kNumSlots; ++ k) {
                    v[k] = make_uint2(0u, 0u);
                    if ((total_mask >> k) & 1u) {
                        const auto src = math::advance_ptr<uint8_t>(
                            buffer.combine_token_buffer.get_rank_buffer(k).get_data_buffer(token_idx).get_base_ptr(), byte_off);
                        v[k] = __ldcg(reinterpret_cast<const uint2*>(src));
                    }
                }
                float2 r0 = {0.f, 0.f}, r1 = {0.f, 0.f};
                #pragma unroll
                for (uint32_t k = 0; k < kNumSlots; ++ k) {
                    if ((total_mask >> k) & 1u) {
                        ptx::accumulate(r0, *reinterpret_cast<const nv_bfloat162*>(&v[k].x));
                        ptx::accumulate(r1, *reinterpret_cast<const nv_bfloat162*>(&v[k].y));
                    }
                }
                const nv_bfloat162 o0 = __float22bfloat162_rn(r0), o1 = __float22bfloat162_rn(r1);
                uint2 o;
                o.x = *reinterpret_cast<const uint32_t*>(&o0);
                o.y = *reinterpret_cast<const uint32_t*>(&o1);
                if (w6p3_add)
                    o = w6p3::add_bf16x4(o, w6p3_out);
                *reinterpret_cast<uint2*>(math::advance_ptr<uint8_t>(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + byte_off)) = o;
            }
        } else
        for (uint32_t token_chunk_idx = epilogue_warp_idx * kNumSMs + sm_idx; token_chunk_idx < num_tokens * kNumChunks; token_chunk_idx += kNumSMs * kNumEpilogueWarps) {
            const uint32_t token_idx = token_chunk_idx / kNumChunks;
            const uint32_t chunk_idx = token_chunk_idx % kNumChunks;

            // Read top-k slot indices: each lane reads one slot, then broadcast via exchange
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[token_idx * kNumTopk + lane_idx]) :
                (kNumSharedExperts > 0 and lane_idx == kNumTopk ? static_cast<int>(kNumTopk) : -1);
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            // Wait for the ranks of the selected experts to finish their L2 writes
            const bool is_routed = lane_idx < kNumTopk and stored_topk_slot_idx >= 0;
            const auto peer_ready_ptr = workspace.get_combine_ready_grid_idx_ptr(
                is_routed ? static_cast<uint32_t>(stored_topk_slot_idx) / kNumExpertsPerRank : 0);
            comm::wait_until([&]() { return __all_sync(0xffffffff, not is_routed or ptx::ld_acq_sys(peer_ready_ptr) == grid_idx); },
                             [&]() { printf("DeepGEMM combine peers timeout: rank=%u, token=%u\n", sym_buffer.rank_idx, token_idx); });

            const uint32_t chunk_byte_offset = chunk_idx * kNumChunkBytes;

            // Move mask and load
            uint32_t mask = total_mask;
            const auto move_mask_and_load = [&](const uint32_t& i) {
                if (mask) {
                    // Move
                    const uint32_t slot_idx = __ffs(mask) - 1;
                    mask ^= 1 << slot_idx;

                    // Load
                    if (cute::elect_one_sync()) {
                        const auto src_ptr = math::advance_ptr<uint8_t>(
                            buffer.combine_token_buffer.get_rank_buffer(slot_idx)
                                                .get_data_buffer(token_idx).get_base_ptr(),
                            chunk_byte_offset);
                        ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                        ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                    }
                    __syncwarp();
                    return true;
                }
                return false;
            };

            // Load the first selection
            bool do_reduce = move_mask_and_load(load_stage_idx);

            // Accumulate all top-k contributions for this chunk in float registers
            float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
            while (do_reduce) {
                // Prefetch next top-k into the buffer while current is being accumulated
                do_reduce = move_mask_and_load(load_stage_idx ^ 1);

                // Accumulate
                combine_load_barriers[load_stage_idx]->wait(combine_phase);
                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                    const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                }
                cutlass::arch::fence_view_async_shared();
                combine_phase ^= load_stage_idx;
                load_stage_idx ^= 1;
            }

            // Cast
            #pragma unroll
            for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                uint4 casted;
                auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                #pragma unroll
                for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                    casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);

                // Wait share memory release and write
                if (j == 0) {
                    ptx::tma_store_wait<0>();
                    __syncwarp();
                }
                ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                               casted.x, casted.y, casted.z, casted.w);
            }
            __syncwarp();

            // TMA store the token chunk
            if (cute::elect_one_sync()) {
                cute::tma_store_fence();
                ptx::tma_store_1d(
                    math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                    combine_store_buffer, kNumChunkBytes);
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace deep_gemm
