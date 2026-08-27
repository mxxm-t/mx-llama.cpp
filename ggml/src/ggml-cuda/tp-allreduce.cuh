#pragma once
#include "common.cuh"

// Custom AllReduce for Tensor Parallelism — GPU-side synchronization only.
// Adapted from vLLM's custom_all_reduce.cuh for llama.cpp's single-process architecture.
//
// Key advantage over NCCL: entirely GPU-side coordination (no CPU proxy threads)
// and elimination of host-side ncclGroupStart/ncclGroupEnd overhead.
//
// Three kernel variants are provided; tp_custom_ar_allreduce selects between
// them at call time. All paths are F32-on-wire — fully lossless. The top-level
// gate in ggml-cuda.cu routes large messages to NCCL by default (NCCL's BF16
// ring beats any F32-on-wire kernel at PP-size on PCIe). Set
// GGML_TP_AR_NO_GATE=1 to keep everything in the custom path; hardware without
// peer-write broadcast support remains on the one-shot path.
//
//   broadcast (small / TG-size ARs when peer-write coherence is available):
//       each rank writes its input into every peer's staging slot via PCIe
//       peer writes, then reduces from its own staging. One kernel launch
//       per rank, no external stream-event handshake. Per-rank outbound:
//       (N-1)*S — fine for small messages (latency-bound).
//
//   twoshot (large / PP-size ARs ≥ ~256K F32; only with NO_GATE or no NCCL):
//       reduce-scatter (peer-write our input slice for each target peer)
//       then allgather (peer-write our reduced slice to every peer's
//       allgather_buf). Per-rank outbound: 2*(N-1)/N * S — half of broadcast
//       at scale. Lossless F32 wire; loses 5-10% PP vs NCCL's BF16 ring on
//       PCIe but kept as scaffold for future quantized-on-wire variants.
//
//   one-shot (fallback, always correct):
//       cudaMemcpyAsync stages input into fine-grained buffers, cross-stream
//       cudaEventRecord/cudaStreamWaitEvent handshake L2-flushes, then a
//       reduce kernel peer-reads all stagings. Used when we can't guarantee
//       peer-write coherence (e.g. MI50/PCIe without HSA_FORCE_FINE_GRAIN_PCIE=1).
//
// All paths maintain uniform per-element reduction order (p0 + p1 + … + p_N-1)
// across ranks so every rank sees bit-identical AR output.

namespace ggml_cuda_tp {

constexpr int kMaxBlocks = 36;
#ifdef GGML_USE_HIP
constexpr int kDefaultBlocks = 16;
#else
constexpr int kDefaultBlocks = 36;
#endif
// Upper bound on supported rank count. Sets compile-time array sizes for
// the Signal/RankData/RankSignals structs. 16 is "any realistic single-host
// PCIe topology"; bumping costs ~300 bytes Signal-struct + 3 template
// specializations per increment. The dispatcher in tp_custom_ar_allreduce
// instantiates each kernel for N=2..kMaxRanks automatically.
constexpr int kMaxRanks = 16;
constexpr int kThreads  = 512;

using FlagType = uint32_t;

// Signal buffer for inter-GPU synchronization.
// Allocated in fine-grained/uncached device memory for cross-device visibility.
// Two flag arrays (start/end) prevent ABA problems between consecutive barriers.
// Every kernel variant (one-shot, broadcast, twoshot) uses only these two —
// the twoshot kernel places all reader-visibility ordering on barrier_start
// (post-scatter) and barrier_end (post-allgather); no third barrier needed
// since the local-scatter-to-result stage reads only allgather_buf, which
// peer stage-1 writes of the next AR do not touch (different region).
struct Signal {
    alignas(128) FlagType start[kMaxBlocks][kMaxRanks];
    alignas(128) FlagType end[kMaxBlocks][kMaxRanks];
    alignas(128) FlagType _flag[kMaxBlocks]; // monotonic counter per block
};

// Pointers to all ranks' input data buffers (copied to device memory).
struct __align__(16) RankData {
    const void * ptrs[kMaxRanks];
};

// Pointers to all ranks' signal buffers (passed as kernel argument).
struct __align__(16) RankSignals {
    Signal * signals[kMaxRanks];
};

// Custom AllReduce context — persistent across evals.
struct CustomARContext {
    int      nranks      = 0;
    bool     initialized = false;

    // Per-device sync events — used by the one-shot path only. The N×N
    // cudaEventRecord/cudaStreamWaitEvent handshake L2-flushes peer staging
    // writes between producer and consumer streams. Broadcast/twoshot use
    // in-kernel SYSTEM-scope atomics + same-source PCIe ordering instead.
    cudaEvent_t events[kMaxRanks] = {};

    // Per-device fine-grained staging buffers. AR inputs live in regular
    // cudaMalloc memory which on MI50 is coarse-grained (non-coherent across
    // devices without a host sync). Copying the input into fine-grained memory
    // makes peer reads coherent without any host round-trip.
    //
    // Sizing depends on the path selected for the next AR call:
    //   broadcast: nranks × n_elements (each rank's staging holds an N-way
    //              inbox where peer R writes its input slice at offset R*n).
    //   twoshot:   2 × n_elements (scatter_buf followed by allgather_buf, each
    //              n_elements floats).
    //   one-shot:  n_elements (single SDMA inbox).
    // Reallocated lazily when staging_slots or staging_size disagrees with
    // the current call's needs.
    float * d_staging[kMaxRanks] = {};
    size_t  staging_size = 0;
    int     staging_slots = 0;  // 0=never allocated; otherwise 1=oneshot, 2=twoshot, nranks=broadcast

    // Signal buffer on each device (fine-grained/uncached memory)
    Signal * d_signals[kMaxRanks] = {};

    // Collected signal pointers (host-side, passed to kernel by value)
    RankSignals rank_signals = {};

    // Device-side RankData (one allocation per rank, updated when ptrs change)
    RankData * d_rank_data[kMaxRanks] = {};

    // Cached data pointers to detect changes
    void * cached_ptrs[kMaxRanks] = {};

    // Whether the broadcast kernel can be used safely on this hardware
    // (i.e. kernel-initiated peer writes are visible to the peer without
    // requiring kernel-exit for cache flush). Populated at init time:
    //  - NVIDIA/CUDA/MUSA: always true (NVLink + PCIe P2P are HW-coherent)
    //  - AMD gfx90a (MI200), gfx94x (MI300), gfx95x: true (XGMI is HW-coherent)
    //  - AMD gfx906 (MI50) PCIe: true only when HSA_FORCE_FINE_GRAIN_PCIE=1
    //  - Otherwise: false → runtime falls back to one-shot path
    bool broadcast_ok = false;

    // Multi-stage TP4 benefits from two-shot at decode sizes while standalone
    // TP4 does not. The meta backend supplies this private topology hint after
    // creating each per-stage communicator.
    bool prefer_small_twoshot = false;
    bool logged_small_twoshot = false;

    // Physical CUDA/HIP device id for each rank. The single-stage TP case has
    // dev_ids[r] == r, but multi-stage TP (vLLM-style TP x PP) gives each stage
    // a non-contiguous slice of the device set, so dev_ids[r] != r in general.
    // Every cudaSetDevice / peer-access call inside this AR context routes
    // through dev_ids[rank] to address the correct physical GPU.
    int dev_ids[kMaxRanks] = {};
};

// Initialize: allocate signal buffers in fine-grained memory, enable peer access.
// dev_ids[rank] gives the physical CUDA/HIP device id for each rank in this
// communicator. Pass nullptr to assume dev_ids[r] == r (back-compat with
// single-stage callers).
void tp_custom_ar_init(CustomARContext * ctx, int nranks, const int * dev_ids = nullptr);

// Destroy: free signal buffers and rank data.
void tp_custom_ar_destroy(CustomARContext * ctx);

// Launch the AllReduce on all devices. Returns false without launching when
// the custom staging allocation cannot be satisfied.
//   input_ptrs[rank]  : input buffer on each rank (GEMM partial results)
//   output_ptrs[rank] : output buffer on each rank (reduced result);
//                       may alias input_ptrs[rank] — both kernels read through
//                       an intermediate staging buffer
//   n_elements        : number of float elements per rank
//   nranks            : number of GPUs (must be even, 2-8)
//   streams[rank]     : CUDA/HIP stream per rank
bool tp_custom_ar_allreduce(CustomARContext * ctx,
                            float ** input_ptrs,
                            float ** output_ptrs,
                            int64_t  n_elements,
                            int      nranks,
                            cudaStream_t * streams);

// Split form of the call above, for callers that want each rank's kernel issued
// by the thread that already owns that rank's stream and already has that
// device current. Issuing all N from one thread costs N host launches plus N
// device switches at every AllReduce, and with one AllReduce per subgraph that
// fan-out is a measurable share of the token.
//
// tp_custom_ar_prepare does every piece of shared host-side work - staging
// (re)allocation, RankData upload, path selection, grid sizing - and must be
// called ONCE from a single thread before any launch_rank for that plan.
// tp_custom_ar_launch_rank then issues only that rank's kernel and mutates no
// shared state, so the N calls may run concurrently on N threads.
//
// The plan borrows ctx and the caller pointer/stream arrays by value, so it is
// valid only until the next prepare on the same ctx.
struct CustomARPlan {
    CustomARContext * ctx        = nullptr;
    float *           inputs [kMaxRanks] = {};
    float *           outputs[kMaxRanks] = {};
    cudaStream_t      streams[kMaxRanks] = {};
    // Ranks whose input tensor was not computed by their own lane must have it
    // zeroed before the collective reads it. Recorded here rather than issued
    // during prepare so it lands after that lane's subgraph.
    void *            zero_ptr[kMaxRanks] = {};
    size_t            zero_bytes = 0;
    int64_t           n_elements = 0;
    int               nranks     = 0;
    int               blocks     = 0;
    bool              twoshot    = false;
    bool              broadcast  = false;
};

// Returns false when the staging allocation cannot be satisfied. No kernels
// are launched in that case, so the caller can use its normal AllReduce path.
bool tp_custom_ar_prepare(CustomARContext * ctx,
                          float ** input_ptrs,
                          float ** output_ptrs,
                          int64_t  n_elements,
                          int      nranks,
                          cudaStream_t * streams,
                          CustomARPlan * plan);

void tp_custom_ar_launch_rank(const CustomARPlan * plan, int rank);

} // namespace ggml_cuda_tp
