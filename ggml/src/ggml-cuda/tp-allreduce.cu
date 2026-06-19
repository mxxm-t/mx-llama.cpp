#include "tp-allreduce.cuh"

#include <type_traits>

namespace ggml_cuda_tp {

// ============================================================================
// Barrier primitives — GPU-side synchronization across devices
// ============================================================================
//
// Protocol (per CTA block):
//   1. Writer: store incremented flag to PEER's signal buffer (cross-device write)
//   2. Reader: poll OWN signal buffer until all peers have written (local read)
//   3. __syncthreads() to synchronize within the block
//
// Signal buffers are in fine-grained/uncached memory so cross-device writes
// are immediately visible without explicit cache flushes.

#if !defined(GGML_USE_HIP)
// ---- NVIDIA CUDA path ----

static __device__ __forceinline__ void st_flag_volatile(FlagType * flag_addr, FlagType flag) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.release.sys.global.u32 [%1], %0;" :: "r"(flag), "l"(flag_addr));
#else
    asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" :: "r"(flag), "l"(flag_addr));
#endif
}

static __device__ __forceinline__ FlagType ld_flag_volatile(FlagType * flag_addr) {
    FlagType flag;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
#else
    asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;" : "=r"(flag) : "l"(flag_addr));
#endif
    return flag;
}

template <int NRANKS>
static __device__ __forceinline__ void barrier_start(
        const RankSignals & sg, Signal * self_sg, int rank) {
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if (threadIdx.x < NRANKS) {
        // Write our flag to peer threadIdx.x's signal buffer
        st_flag_volatile(&sg.signals[threadIdx.x]->start[blockIdx.x][rank], flag);
        // Wait until peer threadIdx.x has written to our buffer
        while (ld_flag_volatile(&self_sg->start[blockIdx.x][threadIdx.x]) != flag)
            ;
    }
    __syncthreads();
    if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

template <int NRANKS>
static __device__ __forceinline__ void barrier_end(
        const RankSignals & sg, Signal * self_sg, int rank) {
    __syncthreads();
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if (threadIdx.x < NRANKS) {
        // Relaxed semantics — no downstream reads depend on this barrier's data.
        asm volatile("st.volatile.global.u32 [%1], %0;" :: "r"(flag),
                     "l"(&sg.signals[threadIdx.x]->end[blockIdx.x][rank]));
        FlagType val;
        do {
            asm volatile("ld.volatile.global.u32 %0, [%1];"
                         : "=r"(val)
                         : "l"(&self_sg->end[blockIdx.x][threadIdx.x]));
        } while (val != flag);
    }
    if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

#else
// ---- AMD HIP/ROCm path ----
// On MI50/gfx906 PCIe (no XGMI), RELEASE/ACQUIRE is required for correctness —
// RELAXED produces garbage even with fine-grained staging (tested). vLLM uses
// RELAXED on MI300X where XGMI provides HW cache coherence. PCIe has none;
// we rely on the C++11 atomic ordering to emit the right fences.
//
// SYSTEM scope on both sides is required: the load must re-fetch HBM each
// iteration (DEVICE scope + RELAXED can cache in L1/L2 even for "uncached"
// memory under some access patterns).

template <int NRANKS>
static __device__ __forceinline__ void barrier_start(
        const RankSignals & sg, Signal * self_sg, int rank) {
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if (threadIdx.x < NRANKS) {
        __hip_atomic_store(
            &sg.signals[threadIdx.x]->start[blockIdx.x][rank],
            flag, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        while (__hip_atomic_load(
                   &self_sg->start[blockIdx.x][threadIdx.x],
                   __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < flag)
            ;
    }
    __syncthreads();
    if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

template <int NRANKS>
static __device__ __forceinline__ void barrier_end(
        const RankSignals & sg, Signal * self_sg, int rank) {
    __syncthreads();
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if (threadIdx.x < NRANKS) {
        __hip_atomic_store(
            &sg.signals[threadIdx.x]->end[blockIdx.x][rank],
            flag, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        while (__hip_atomic_load(
                   &self_sg->end[blockIdx.x][threadIdx.x],
                   __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < flag)
            ;
    }
    if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

#endif // GGML_USE_HIP

// ============================================================================
// One-shot AllReduce kernel
// ============================================================================
//
// Each rank reads from ALL ranks' partial buffers via peer access, sums locally,
// writes result to its own output buffer. Uses 128-bit packed loads (float4)
// for bandwidth efficiency.
//
// Data flow:
//   barrier_start  — ensure all GEMMs have written their partials
//   packed reduce  — read float4 from each peer, sum, write to local output
//   barrier_end    — ensure all reads complete before next op overwrites partials

template <int NRANKS>
__global__ void __launch_bounds__(kThreads, 1)
k_cross_device_reduce_1stage(
        RankData * __restrict__ _dp,
        RankSignals              sg,
        Signal *   __restrict__  self_sg,
        float *    __restrict__  result,
        int                      rank,
        int64_t                  n_elements) {

    // Copy RankData to registers (64 bytes — fits in register file).
    // Passing as a pointer + loading once keeps SGPR pressure low on gfx906
    // (versus pass-by-value, which bloats the kernel arg region and hurts
    // occupancy; measured ~6% TG regression on 4×MI50).
    auto dp = *_dp;

    // Start barrier: all ranks' partial data is ready
    barrier_start<NRANKS>(sg, self_sg, rank);

    // Reduce using float4 (128-bit packed loads/stores). Use non-temporal loads
    // for peer reads on HIP — peer data is one-shot (no reuse) and normal
    // cached loads can return stale L2 entries from prior ARs on MI50 without
    // XGMI cache coherence. NT loads go direct to HBM over PCIe.
    const int64_t n_float4 = n_elements / 4;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_float4;
         idx += (int64_t)gridDim.x * blockDim.x) {

        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
#if defined(GGML_USE_HIP)
            const float * base = ((const float *)dp.ptrs[r]) + idx * 4;
            const float x = __builtin_nontemporal_load(base + 0);
            const float y = __builtin_nontemporal_load(base + 1);
            const float z = __builtin_nontemporal_load(base + 2);
            const float w = __builtin_nontemporal_load(base + 3);
#else
            const float4 v = ((const float4 *)dp.ptrs[r])[idx];
            const float x = v.x, y = v.y, z = v.z, w = v.w;
#endif
            sum.x += x;
            sum.y += y;
            sum.z += z;
            sum.w += w;
        }
        ((float4 *)result)[idx] = sum;
    }

    // Handle remainder (n_elements not divisible by 4)
    const int64_t rem_start = n_float4 * 4;
    for (int64_t idx = rem_start + (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += (int64_t)gridDim.x * blockDim.x) {

        float sum = 0.0f;
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
#if defined(GGML_USE_HIP)
            sum += __builtin_nontemporal_load(((const float *)dp.ptrs[r]) + idx);
#else
            sum += ((const float *)dp.ptrs[r])[idx];
#endif
        }
        result[idx] = sum;
    }

    // End barrier: all reads complete — safe for next AllReduce to overwrite partials
    barrier_end<NRANKS>(sg, self_sg, rank);
}

// ----------------------------------------------------------------------------
// Broadcast + reduce kernel (default for small / TG-size ARs).
// Each rank writes its input into *every peer's* staging at offset
// rank*n_elements via PCIe peer access. PCIe peer writes go directly to the
// peer's HBM (bypassing both source and destination L2), so the producer's
// data is peer-visible without the usual cudaEventRecord/StreamWaitEvent
// L2-flush handshake.
//
// Staging layout (per rank): [N_ranks][n_elements] — rank R's input from peer
// R lives at offset R*n_elements. "Self slot" (R==rank) is unused; we read
// our own contribution from `input` (local L2-coherent read).
//
// Correctness contract: the in-kernel barrier's SYSTEM-scope RELEASE atomic
// flag store is itself a PCIe peer write; PCIe ordering guarantees writes
// from the same source arrive in-order, so the flag arrives after all prior
// data writes. Peer sees our flag → all our prior data writes are in peer's
// HBM.
//
// Each rank's outbound traffic: (N-1)*S bytes of PCIe writes (posted, async,
// low latency). Same aggregate BW as one-shot reads, but writes typically
// beat reads on PCIe 3.0.
// ----------------------------------------------------------------------------
template <int NRANKS>
__global__ void __launch_bounds__(kThreads, 1)
k_broadcast_reduce(
        RankData * __restrict__ _dp,        // peer (and self) staging pointers
        RankSignals              sg,
        Signal *   __restrict__  self_sg,
        const float * __restrict__ input,   // our own input (local device)
        float *    __restrict__  result,    // our own output (local device)
        int                      rank,
        int64_t                  n_elements) {

    auto dp = *_dp;

    // Phase 1: write our input into each PEER's staging at offset rank*n_elements.
    // We do NOT write the self-slot. Writing the self-slot would be a local
    // store to fine-grained memory, which empirically does NOT bypass L2 on
    // gfx906 even under FGP=1 — phase 2's NT load from HBM would then miss
    // the fresh self-data. This corrupts ~1/N of the reduction inputs and
    // degrades PPL (~+2 on a 10-baseline at ubatch=32, tested). Reading our
    // own contribution directly from `input` in phase 2 (local L2-coherent
    // read) avoids that path entirely.
    const int64_t n_float4 = n_elements / 4;
    const int64_t rank_offset = (int64_t) rank * n_elements;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_float4;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const float4 v = ((const float4 *) input)[idx];
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
            if (r == rank) continue;   // self-slot is NOT peer-written; read `input` directly in phase 2
            float * dst = ((float *)(uintptr_t)dp.ptrs[r]) + rank_offset + idx * 4;
            *((float4 *) dst) = v;     // peer write (bypasses source + dest L2 via PCIe)
        }
    }
    const int64_t rem_start = n_float4 * 4;
    for (int64_t idx = rem_start + (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const float v = input[idx];
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
            if (r == rank) continue;
            ((float *)(uintptr_t)dp.ptrs[r])[rank_offset + idx] = v;
        }
    }

    // Ensure every thread in this CTA has finished its peer writes before the
    // first NRANKS threads publish the completion flag below.
    __syncthreads();

    // Force all prior peer-writes to retire before we signal via barrier.
    // The RELEASE-scope atomic inside barrier_start should do this on its own,
    // but an explicit system fence is cheap and proved necessary for a 0.18
    // PPL drift that the atomic-alone variant had at ubatch=32.
    __threadfence_system();

    // Barrier: SYSTEM-scope RELEASE flag store is itself a peer write; PCIe
    // same-source ordering guarantees our prior data writes arrive first.
    barrier_start<NRANKS>(sg, self_sg, rank);

    // Phase 2: identical reduction order on EVERY rank:
    //   sum = 0 + p0 + p1 + p2 + ... + p_{N-1}
    // For r == rank we read `input` directly (local L2-coherent); for peers we
    // NT-load from the per-peer slot in our own staging (peer-written → HBM).
    // The `if (r == rank)` branch inside a #pragma unroll + compile-time NRANKS
    // resolves at compile time per unrolled iteration, so the emitted code has
    // no runtime branch — same accumulation sequence on all ranks.
    const float * local_staging = (const float *) dp.ptrs[rank];
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_float4;
         idx += (int64_t)gridDim.x * blockDim.x) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
            float4 v;
            if (r == rank) {
                v = ((const float4 *) input)[idx];   // self contribution
            } else {
                const float * base = local_staging + (int64_t)r * n_elements + idx * 4;
#if defined(GGML_USE_HIP)
                v.x = __builtin_nontemporal_load(base + 0);
                v.y = __builtin_nontemporal_load(base + 1);
                v.z = __builtin_nontemporal_load(base + 2);
                v.w = __builtin_nontemporal_load(base + 3);
#else
                v = *((const float4 *) base);
#endif
            }
            sum.x += v.x;
            sum.y += v.y;
            sum.z += v.z;
            sum.w += v.w;
        }
        ((float4 *) result)[idx] = sum;
    }
    for (int64_t idx = rem_start + (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += (int64_t)gridDim.x * blockDim.x) {
        float sum = 0.0f;
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
            float v;
            if (r == rank) {
                v = input[idx];
            } else {
                const float * base = local_staging + (int64_t)r * n_elements + idx;
#if defined(GGML_USE_HIP)
                v = __builtin_nontemporal_load(base);
#else
                v = *base;
#endif
            }
            sum += v;
        }
        result[idx] = sum;
    }

    barrier_end<NRANKS>(sg, self_sg, rank);
}

// ----------------------------------------------------------------------------
// Two-shot F32 AllReduce (reduce-scatter + allgather via peer writes).
//
// Target: large (PP-size) messages where the broadcast kernel's (N-1)*S
// per-rank outbound saturates PCIe. Two-shot brings per-rank outbound down
// to 2*(N-1)/N * S, with parallel all-to-all peer writes instead of NCCL's
// serial ring — wins on PCIe systems where every GPU has its own root-
// complex link AND the fabric can carry parallel transfers concurrently.
//
// LOSSLESS: F32 input → F32 wire → F32 reduction (F32 accumulator) → F32
// output. No BF16 round-trip anywhere; output is bit-deterministic and a
// strictly lossless function of the inputs (modulo the standard fp non-
// associativity of summation, which is fixed across runs because all ranks
// sum in the same r-order). Costs 2× the PCIe bytes per AR vs a BF16-on-
// wire variant; the trade-off vs NCCL BF16 ring is BW for precision.
//
// Staging layout (per rank, bytes):
//   [0, n_elements * 4)                    : scatter_buf — N F32 slots, each
//                                            of slice = N_elements/N elements.
//                                            Slot r holds rank r's contribution
//                                            to OUR slice (my_rank). Filled in
//                                            stage 1 by peer writes.
//   [n_elements * 4, 2 * n_elements * 4)   : allgather_buf — N F32 slots, same
//                                            layout. Slot r holds rank r's
//                                            reduced slice. Filled in stage 3
//                                            by peer writes from rank r.
// dp.ptrs[r] points to rank r's scatter_buf base; allgather_buf is at
// +n_elements floats from that base.
//
// Per-rank peer traffic: stage 1 (N-1) * slice F32 + stage 3 (N-1) * slice
// F32 = 2*(N-1)/N * S bytes outbound (S = n_elements * sizeof(float)).
//
// Requires n_elements % NRANKS == 0 and slice % 4 == 0 for float4 path. The
// caller (host dispatch) must gate on these before selecting this kernel.
// ----------------------------------------------------------------------------
template <int NRANKS>
__global__ void __launch_bounds__(kThreads, 1)
k_twoshot_f32(
        RankData * __restrict__ _dp,        // peer staging base pointers (float*)
        RankSignals              sg,
        Signal *   __restrict__  self_sg,
        const float * __restrict__ input,   // our own input (F32)
        float *    __restrict__  result,    // our own output (F32)
        int                      rank,
        int64_t                  n_elements) {

    auto dp = *_dp;

    const int64_t slice = n_elements / NRANKS;          // guaranteed divisible
    const int64_t slice_n_float4 = slice / 4;
    const int64_t my_slice_start = (int64_t) rank * slice;

    // Staging pointers — each rank's staging holds scatter_buf then allgather_buf.
    // rank r's scatter_buf base   = dp.ptrs[r]              (float*)
    // rank r's allgather_buf base = dp.ptrs[r] + n_elements (float* stride of n_elements floats)
    auto scat_ptr = [&](int r) -> float * { return (float *)(uintptr_t) dp.ptrs[r]; };
    auto ag_ptr   = [&](int r) -> float * { return ((float *)(uintptr_t) dp.ptrs[r]) + n_elements; };

    // ------------------------------------------------------------------------
    // Stage 1 (peer-write scatter): for each non-self target peer, all threads
    // grid-stride over the target's slice of our input and peer-write F32x4
    // to the target's scatter_buf at OUR slot. Round-robin target order
    // (rank+t) % NRANKS spreads PCIe contention so all ranks write to
    // different targets at the same loop iteration.
    // ------------------------------------------------------------------------
    const int64_t my_slot_offset = (int64_t) rank * slice;
#pragma unroll
    for (int t = 1; t < NRANKS; t++) {
        const int target = (rank + t) % NRANKS;
        const int64_t target_slice_start_f4 = ((int64_t) target * slice) / 4;
        for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
             idx < slice_n_float4;
             idx += (int64_t)gridDim.x * blockDim.x) {
            const float4 v = ((const float4 *) input)[target_slice_start_f4 + idx];
            float4 * dst   = (float4 *)(scat_ptr(target) + my_slot_offset + idx * 4);
            *dst = v;       // 16-byte PCIe peer write
        }
    }

    // Ensure every thread in this CTA has finished its peer writes before the
    // first NRANKS threads publish the completion flag below.
    __syncthreads();
    __threadfence_system();
    barrier_start<NRANKS>(sg, self_sg, rank);

    // ------------------------------------------------------------------------
    // Stage 2 + 3 fused: reduce our slice from scatter_buf + own input, then
    // peer-write the F32 sum to every rank's allgather_buf at OUR slot.
    // Self-contribution is read directly from `input` — no precision adjust
    // needed (peers see our slice as F32 via stage-1 peer write; same value).
    // ------------------------------------------------------------------------
    const float * own_scat = scat_ptr(rank);
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < slice_n_float4;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t local_pos = idx * 4;               // position within slice
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
            float4 v;
            if (r == rank) {
                v = ((const float4 *) input)[my_slice_start / 4 + idx];
            } else {
                const float4 * base = (const float4 *)(own_scat + (int64_t) r * slice + local_pos);
#if defined(GGML_USE_HIP)
                v.x = __builtin_nontemporal_load(((const float *) base) + 0);
                v.y = __builtin_nontemporal_load(((const float *) base) + 1);
                v.z = __builtin_nontemporal_load(((const float *) base) + 2);
                v.w = __builtin_nontemporal_load(((const float *) base) + 3);
#else
                v = *base;
#endif
            }
            sum.x += v.x;
            sum.y += v.y;
            sum.z += v.z;
            sum.w += v.w;
        }

        // Allgather peer-write: send our reduced F32 slice to every PEER's
        // allgather_buf[rank_slot][idx*4]. We do NOT local-write to OWN
        // allgather_buf — on gfx906/PCIe, kernel-initiated local writes to
        // fine-grained memory dwell in L2 until kernel exit, while stage-4
        // NT-load bypasses L2 and would see stale HBM. Instead we write our
        // own slice directly to `result` (lossless, no round-trip).
        const int64_t slot_off = my_slice_start;
#pragma unroll
        for (int t = 1; t < NRANKS; t++) {
            const int target = (rank + t) % NRANKS;
            float4 * dst = (float4 *)(ag_ptr(target) + slot_off + local_pos);
            *dst = sum;                                  // 16-byte PCIe peer write
        }
        // Own slice: write F32 sum directly to result.
        ((float4 *) result)[my_slice_start / 4 + idx] = sum;
    }

    __threadfence_system();
    barrier_end<NRANKS>(sg, self_sg, rank);
    // barrier_end has a __syncthreads() at its start but NOT at its end —
    // only the first NRANKS threads spin on peer flags. For kernels that READ
    // peer-written data after barrier_end (stage 4), the remaining threads
    // must wait for the spinning threads to confirm peer writes are visible.
    // Without this sync, threads >= NRANKS run stage-4 NT-loads before peers'
    // stage-3 writes have arrived, returning stale HBM and corrupting output.
    __syncthreads();

    // ------------------------------------------------------------------------
    // Stage 4 (local scatter to F32 result): peer slots only (own slice was
    // written directly to `result` in stage 3). Each peer's allgather_buf
    // slot is FGP-coherent (peer writes bypass source L2 → land in our HBM)
    // so NT loads return the freshly-arrived F32 values.
    // ------------------------------------------------------------------------
    const float * own_ag = ag_ptr(rank);
#pragma unroll
    for (int t = 1; t < NRANKS; t++) {
        const int src_slot = (rank + t) % NRANKS;
        const int64_t slot_base = (int64_t) src_slot * slice;
        for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
             idx < slice_n_float4;
             idx += (int64_t)gridDim.x * blockDim.x) {
            const float4 * base = (const float4 *)(own_ag + slot_base + idx * 4);
#if defined(GGML_USE_HIP)
            float4 v;
            v.x = __builtin_nontemporal_load(((const float *) base) + 0);
            v.y = __builtin_nontemporal_load(((const float *) base) + 1);
            v.z = __builtin_nontemporal_load(((const float *) base) + 2);
            v.w = __builtin_nontemporal_load(((const float *) base) + 3);
#else
            float4 v = *base;
#endif
            ((float4 *) result)[(slot_base / 4) + idx] = v;
        }
    }
}

// ============================================================================
// Host functions
// ============================================================================

void tp_custom_ar_init(CustomARContext * ctx, int nranks, const int * dev_ids) {
    if (ctx->initialized) return;

    ctx->nranks = nranks;
    for (int r = 0; r < nranks; r++) {
        ctx->dev_ids[r] = dev_ids ? dev_ids[r] : r;
    }

    // Enable peer access between all GPU pairs
    bool peer_access_ok = true;
    for (int i = 0; i < nranks; i++) {
        ggml_cuda_set_device(ctx->dev_ids[i]);
        for (int j = 0; j < nranks; j++) {
            if (i == j) continue;
            int can_access = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, ctx->dev_ids[i], ctx->dev_ids[j]));
            if (can_access) {
                cudaError_t err = cudaDeviceEnablePeerAccess(ctx->dev_ids[j], 0);
                if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                    CUDA_CHECK(err);
                }
                // cudaDeviceEnablePeerAccess leaves the sticky error state set when access
                // was already enabled. Clear it here so the first downstream cudaGetLastError()
                // (e.g. after a MUL_MAT kernel) doesn't trip on it.
                (void) cudaGetLastError();
            } else {
                GGML_LOG_WARN("TP custom AR: peer access not available between GPU %d and %d\n",
                              ctx->dev_ids[i], ctx->dev_ids[j]);
                peer_access_ok = false;
            }
        }
    }
    if (!peer_access_ok) {
        GGML_LOG_WARN("TP custom AR: disabled because full peer access is not available\n");
        return;
    }

    // Allocate signal buffers (one per rank) in fine-grained/uncached memory
    for (int rank = 0; rank < nranks; rank++) {
        ggml_cuda_set_device(ctx->dev_ids[rank]);

#if defined(GGML_USE_HIP)
        // AMD: hipDeviceMallocUncached for cross-device visibility (matches vLLM)
        void * ptr;
        CUDA_CHECK(hipExtMallocWithFlags(&ptr, sizeof(Signal), hipDeviceMallocUncached));
        ctx->d_signals[rank] = (Signal *)ptr;
#else
        // NVIDIA: regular device memory (peer access handles visibility)
        CUDA_CHECK(cudaMalloc(&ctx->d_signals[rank], sizeof(Signal)));
#endif
        CUDA_CHECK(cudaMemset(ctx->d_signals[rank], 0, sizeof(Signal)));

        // Allocate device-side RankData (holds pointers to all ranks' buffers)
        CUDA_CHECK(cudaMalloc(&ctx->d_rank_data[rank], sizeof(RankData)));

        ctx->rank_signals.signals[rank] = ctx->d_signals[rank];

        // Event used for cross-stream handshake before each AR
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx->events[rank], cudaEventDisableTiming));
    }

    // Determine whether the broadcast path is safe on this hardware.
    // The broadcast kernel relies on kernel-initiated PCIe peer writes being
    // visible to the destination rank before the in-kernel barrier's RELEASE
    // flag arrives, i.e. cache-coherent peer writes. This is true on:
    //   - NVIDIA (any CUDA-capable device): NVLink and PCIe P2P are HW-coherent
    //   - AMD gfx90a (MI200), gfx94x (MI300), gfx95x: XGMI is HW-coherent
    //   - any AMD PCIe GPU when HSA_FORCE_FINE_GRAIN_PCIE=1 forces all device
    //     allocations to fine-grained (write-through) memory. The fine-grain
    //     mechanism is generic to AMD, but this peer-write path is validated
    //     only on gfx906 (MI50), so other AMD archs get a warning below.
    // Without FGP (and off XGMI) we stay on the one-shot path, which uses
    // cudaMemcpyAsync staging + an event handshake and is always correct.
    bool hw_peer_write_coherent = false;
#if defined(GGML_USE_HIP)
    {
        hipDeviceProp_t prop;
        CUDA_CHECK(hipGetDeviceProperties(&prop, ctx->dev_ids[0]));
        const char * arch = prop.gcnArchName;   // e.g. "gfx906:sramecc+:xnack-"
        const bool is_gfx906 = (strncmp(arch, "gfx906", 6) == 0);
        const bool is_gfx9_xgmi =
            (strncmp(arch, "gfx90a", 6) == 0) ||
            (strncmp(arch, "gfx94",  5) == 0) ||
            (strncmp(arch, "gfx95",  5) == 0);
        const char * e_fgp = getenv("HSA_FORCE_FINE_GRAIN_PCIE");
        const bool fgp_on = e_fgp && e_fgp[0] != '\0' && e_fgp[0] != '0';
        hw_peer_write_coherent = is_gfx9_xgmi || fgp_on;
        const bool peer_write_experimental = fgp_on && !is_gfx906 && !is_gfx9_xgmi;
        if (peer_write_experimental) {
            GGML_LOG_WARN("TP custom AR: peer-write enabled on %s via HSA_FORCE_FINE_GRAIN_PCIE "
                          "(validated only on gfx906, verify output correctness)\n", arch);
        }
    }
#else
    hw_peer_write_coherent = true;   // CUDA / MUSA: HW-coherent peer access
#endif
    ctx->broadcast_ok = hw_peer_write_coherent;
    ctx->initialized  = true;

    const char * path      = ctx->broadcast_ok ? "broadcast F32 + twoshot F32 (peer-write, size-adaptive, lossless)"
                              :
#if defined(GGML_USE_HIP)
                                "one-shot (PCIe without HSA_FORCE_FINE_GRAIN_PCIE=1, set it to enable fast peer-write paths)"
#else
                                "one-shot (HW not recognised as peer-write coherent)"
#endif
                              ;
    GGML_LOG_INFO("TP custom AllReduce: initialized for %d GPUs, path = %s\n", nranks, path);
}

void tp_custom_ar_destroy(CustomARContext * ctx) {
    if (!ctx->initialized) return;

    for (int rank = 0; rank < ctx->nranks; rank++) {
        ggml_cuda_set_device(ctx->dev_ids[rank]);
        if (ctx->events[rank]) {
            CUDA_CHECK(cudaEventDestroy(ctx->events[rank]));
            ctx->events[rank] = nullptr;
        }
        if (ctx->d_signals[rank]) {
#if defined(GGML_USE_HIP)
            CUDA_CHECK(hipFree(ctx->d_signals[rank]));
#else
            CUDA_CHECK(cudaFree(ctx->d_signals[rank]));
#endif
            ctx->d_signals[rank] = nullptr;
        }
        if (ctx->d_rank_data[rank]) {
            CUDA_CHECK(cudaFree(ctx->d_rank_data[rank]));
            ctx->d_rank_data[rank] = nullptr;
        }
        if (ctx->d_staging[rank]) {
#if defined(GGML_USE_HIP)
            CUDA_CHECK(hipFree(ctx->d_staging[rank]));
#else
            CUDA_CHECK(cudaFree(ctx->d_staging[rank]));
#endif
            ctx->d_staging[rank] = nullptr;
        }
    }
    ctx->staging_size = 0;

    ctx->initialized = false;
}

void tp_custom_ar_allreduce(CustomARContext * ctx,
                            float ** input_ptrs,
                            float ** output_ptrs,
                            int64_t  n_elements,
                            int      nranks,
                            cudaStream_t * streams) {
    GGML_ASSERT(ctx->initialized);
    GGML_ASSERT(nranks == ctx->nranks);
    GGML_ASSERT(nranks >= 2 && nranks <= kMaxRanks);

    // Path selection. broadcast_ok was set at init based on the HW peer-write
    // coherence model (see tp_custom_ar_init).
    // Two-shot (reduce-scatter + allgather) kernel selection for large msgs.
    // F32 on the wire — fully lossless, matches the broadcast kernel's
    // precision and avoids any BF16 conversion. Per-rank outbound is
    // 2*(N-1)/N * S, so 2× the BW of an equivalent BF16-on-wire variant
    // but with parallel all-to-all peer writes (vs NCCL's serial ring).
    //
    // GGML_TP_AR_TWOSHOT can force on (=1) or off (=0); default auto picks
    // twoshot for ne ≥ ~256K elements where broadcast's (N-1)*S/N per-rank
    // outbound starts to lose to twoshot's 2*(N-1)/N*S aggregate parallel BW.
    static const int s_twoshot_env = []{
        const char * e = getenv("GGML_TP_AR_TWOSHOT");
        if (!e || !e[0]) return -1;                      // default = auto
        return (e[0] != '0') ? 1 : 0;
    }();
    const bool s_broadcast = ctx->broadcast_ok;
    // Twoshot needs slice-aligned + float4-aligned data: n_elements % (4*nranks) == 0.
    const bool twoshot_eligible =
        s_broadcast && (n_elements % (int64_t)(4 * nranks) == 0);
    const int64_t twoshot_min_ne = 262144;               // ~1 MB F32 crossover
    const bool s_twoshot =
        twoshot_eligible &&
        (s_twoshot_env == 1 || (s_twoshot_env == -1 && n_elements >= twoshot_min_ne));

    const size_t bytes_f32 = (size_t) n_elements * sizeof(float);
    // Broadcast path: F32 staging, N slots per rank (one inbox per peer + unused self slot).
    // Two-shot path: F32 staging, 2 buffers (scatter_buf + allgather_buf) each
    //   sized n_elements floats. Total = 2 * n_elements * sizeof(float) bytes.
    const size_t per_slot   = bytes_f32;
    const int    need_slots = s_twoshot ? 2 : (s_broadcast ? ctx->nranks : 1);
    const size_t need_bytes = per_slot * need_slots;

    if (ctx->staging_slots != need_slots || ctx->staging_size < need_bytes) {
        for (int rank = 0; rank < ctx->nranks; rank++) {
            ggml_cuda_set_device(ctx->dev_ids[rank]);
            if (ctx->d_staging[rank]) {
#if defined(GGML_USE_HIP)
                CUDA_CHECK(hipFree(ctx->d_staging[rank]));
#else
                CUDA_CHECK(cudaFree(ctx->d_staging[rank]));
#endif
                ctx->d_staging[rank] = nullptr;
            }
#if defined(GGML_USE_HIP)
            void * ptr = nullptr;
            CUDA_CHECK(hipExtMallocWithFlags(&ptr, need_bytes, hipDeviceMallocFinegrained));
            ctx->d_staging[rank] = (float *) ptr;
#else
            CUDA_CHECK(cudaMalloc(&ctx->d_staging[rank], need_bytes));
#endif
        }
        ctx->staging_size  = need_bytes;
        ctx->staging_slots = need_slots;
        // invalidate cached RankData pointers → forces re-upload below
        for (int i = 0; i < nranks; i++) ctx->cached_ptrs[i] = nullptr;
    }

    if (!s_broadcast) {
        // One-shot path: SDMA staging + event handshake. Used as a fallback
        // when peer-write coherence is unavailable.
        for (int rank = 0; rank < nranks; rank++) {
            ggml_cuda_set_device(ctx->dev_ids[rank]);
            CUDA_CHECK(cudaMemcpyAsync(ctx->d_staging[rank], input_ptrs[rank], bytes_f32,
                                       cudaMemcpyDeviceToDevice, streams[rank]));
        }
        for (int rank = 0; rank < nranks; rank++) {
            ggml_cuda_set_device(ctx->dev_ids[rank]);
            CUDA_CHECK(cudaEventRecord(ctx->events[rank], streams[rank]));
        }
        for (int rank = 0; rank < nranks; rank++) {
            for (int peer = 0; peer < nranks; peer++) {
                if (peer == rank) continue;
                CUDA_CHECK(cudaStreamWaitEvent(streams[rank], ctx->events[peer], 0));
            }
        }
    }

    // Update device-side RankData only when staging pointers change (rare —
    // staging buffers persist in ctx after first alloc, unless switching
    // between one-shot and broadcast staging layouts).
    bool ptrs_changed = false;
    for (int i = 0; i < nranks; i++) {
        if ((void *) ctx->d_staging[i] != ctx->cached_ptrs[i]) {
            ptrs_changed = true;
            break;
        }
    }
    if (ptrs_changed) {
        RankData h_data;
        for (int i = 0; i < nranks; i++) {
            h_data.ptrs[i] = ctx->d_staging[i];
            ctx->cached_ptrs[i] = ctx->d_staging[i];
        }
        for (int rank = 0; rank < nranks; rank++) {
            ggml_cuda_set_device(ctx->dev_ids[rank]);
            CUDA_CHECK(cudaMemcpy(ctx->d_rank_data[rank], &h_data,
                                  sizeof(RankData), cudaMemcpyHostToDevice));
        }
    }

    // Compute grid dimensions. Block count can be overridden via GGML_TP_AR_BLOCKS
    // for quick perf sweeps; default kDefaultBlocks is tuned per platform in .cuh.
    static const int s_blocks_override = []{
        const char * e = getenv("GGML_TP_AR_BLOCKS");
        if (!e || !e[0]) return 0;
        int v = atoi(e);
        return (v > 0 && v <= kMaxBlocks) ? v : 0;
    }();
    const int blocks_cap = s_blocks_override ? s_blocks_override : kDefaultBlocks;
    const int64_t packed_size = n_elements / 4;
    int blocks = std::min(blocks_cap, std::max(1, (int)((packed_size + kThreads - 1) / kThreads)));

    // Per-N kernel launches. The dispatcher below recursively matches `nranks`
    // against compile-time N from kMaxRanks down to 2; the compiler folds the
    // chain into a jump table at -O2. Adding ranks just means bumping
    // kMaxRanks — the templates instantiate automatically.
    auto launch_peer = [&](auto N_CONST) {
        constexpr int N = decltype(N_CONST)::value;
        for (int rank = 0; rank < nranks; rank++) {
            ggml_cuda_set_device(ctx->dev_ids[rank]);
            if (s_twoshot) {
                k_twoshot_f32<N><<<blocks, kThreads, 0, streams[rank]>>>(
                    ctx->d_rank_data[rank], ctx->rank_signals, ctx->d_signals[rank],
                    input_ptrs[rank], output_ptrs[rank], rank, n_elements);
            } else {
                k_broadcast_reduce<N><<<blocks, kThreads, 0, streams[rank]>>>(
                    ctx->d_rank_data[rank], ctx->rank_signals, ctx->d_signals[rank],
                    input_ptrs[rank], output_ptrs[rank], rank, n_elements);
            }
            CUDA_CHECK(cudaGetLastError());
        }
    };
    auto launch_oneshot = [&](auto N_CONST) {
        constexpr int N = decltype(N_CONST)::value;
        for (int rank = 0; rank < nranks; rank++) {
            ggml_cuda_set_device(ctx->dev_ids[rank]);
            k_cross_device_reduce_1stage<N><<<blocks, kThreads, 0, streams[rank]>>>(
                ctx->d_rank_data[rank], ctx->rank_signals, ctx->d_signals[rank],
                output_ptrs[rank], rank, n_elements);
            CUDA_CHECK(cudaGetLastError());
        }
    };

    auto dispatch = [&](auto self, auto N_CONST) -> void {
        constexpr int N = decltype(N_CONST)::value;
        if constexpr (N < 2) {
            GGML_ABORT("TP custom AR: unsupported nranks=%d (must be 2..%d)\n", nranks, kMaxRanks);
        } else if (nranks == N) {
            if (s_broadcast) launch_peer(N_CONST);
            else             launch_oneshot(N_CONST);
        } else {
            self(self, std::integral_constant<int, N - 1>{});
        }
    };
    dispatch(dispatch, std::integral_constant<int, kMaxRanks>{});
}

} // namespace ggml_cuda_tp
