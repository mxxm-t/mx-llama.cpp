// Shared layout helpers and device-side data types for the Q8_0 repacked-weight path.
#pragma once

#include "../common.cuh"
#include "../mmq.cuh"
#include "../quantize.cuh"
#include "../vecdotq.cuh"   // get_int_from_table_16, for the MXFP4 nibble expand

#include <cstddef>

#define MMQ_RP_Q8_BK 4
#define MMQ_RP_Q8_TN 2
#define MMQ_RP_Q8_BM 64
#define MMQ_RP_Q8_NROW_LANES 4
// Widths at or below this take one multi-column mat-vec pass instead of the
// tiled GEMM. Measured on Qwen3.8-27B pp512, 4x MI50 -sm tensor: the mat-vec
// leads from 2 tokens (77.2 against the tile's 14.3) through 8 (159.6 against
// 98.9), and the tile takes back over by 12, so the crossover is 8 to 12.
#define MMQ_RP_Q8_MMV_MAX_TOKENS 8
// MoE narrow batch: at or below this many tokens, run one mat-vec per
// assignment instead of the tiled GEMM. Expert token counts are per expert, so
// at these widths nearly every active expert holds a single assignment.
#define MMQ_RP_Q8_MOE_MMV_MAX_TOKENS 8

// qs plane row stride in BYTES: ne0 (one byte per quant), bumped by 16 whenever
// it lands on a multiple of 128 so row starts do not alias HBM channels. 16
// keeps uint4 alignment and costs half of the padding sub-block it replaces,
// which also only fired when n_sub was a power of two and so missed shapes like
// ne0=5120. Scales keep their own plane: the narrow mat-vec reads several rows
// per lane and wants them adjacent.
template <typename T>
static __host__ __device__ inline T repack_qs_stride(const T ne0) {
    T rs = ne0;
    if (rs % 128 == 0) {
        rs += 16;
    }
    return rs;
}

// Per-type qs row stride and scale row bytes: Q8_0 rows carry ne0 payload
// bytes and 2 B f16 scales per sub-block, MXFP4 rows carry packed nibbles
// (ne0/2 B) and 1 B e8m0 scales per sub-block. Both share the de-alias bump.
template <typename T>
static __host__ __device__ inline T repack_qs_row_stride(const ggml_type type, const T ne0) {
    return repack_qs_stride(type == GGML_TYPE_MXFP4 ? ne0 / 2 : ne0);
}
template <typename T>
static __host__ __device__ inline T repack_scale_row_bytes(const ggml_type type, const T ne0) {
    return (ne0 / 32) * (type == GGML_TYPE_MXFP4 ? 1 : 2);
}

static inline size_t repack_gcn_nbytes(const ggml_type type, const int64_t ne0, const int64_t ne1) {
    GGML_ASSERT(ne0 % 32 == 0);
    switch (type) {
        case GGML_TYPE_Q8_0:
            return (size_t) ne1 * ((size_t) repack_qs_stride(ne0) + (size_t)(ne0 / 32) * 2);
        // nibble plane row [ne0/2 B, de-aliased like the Q8_0 qs rows] + 1-byte
        // e8m0 plane [ne0/32]/row. 17 B/block = canonical size, so the repack
        // stays VRAM-neutral (a 2-byte scale slot cost +5.9% and OOMed
        // gpt-oss-120b at 2-GPU full offload). The GEMM's LDS scale array stays
        // uint16_t; only the GLOBAL load narrows.
        case GGML_TYPE_MXFP4:
            return (size_t) ne1 * ((size_t) repack_qs_stride(ne0 / 2) + (size_t)(ne0 / 32));
        default:             GGML_ABORT("unsupported repack type");
    }
}

// Bytes per sub-block in the qs plane, and uint4 loads needed to fetch one.
static __host__ __device__ inline int repack_qs_bytes(const ggml_type type) {
    return type == GGML_TYPE_MXFP4 ? 16 : 32;
}

#if defined(GGML_USE_HIP) && defined(__gfx906__)
// MXFP4 sub-block: 16 payload bytes coding 32 values as nibbles.
// get_int_from_table_16 returns v.x for the four LOW nibbles and v.y for the four
// HIGH nibbles, and stock load_tiles_mxfp4 stores them at k0 and k0 + QI_MXFP4 -
// i.e. the high nibbles of int32 slot j are values 16+4j..19+4j, NOT 4j+4...
// So all four v.x concatenate into values 0..15 and all four v.y into 16..31,
// which is exactly the lo/hi split the Q8_0 path already feeds to dp4a.
// Swapping these compiles, runs, and produces plausible-looking wrong output.
static __device__ __forceinline__ void rp_mxfp4_expand(
        const uint4 nib, uint4 & lo, uint4 & hi) {
    const int2 v0 = get_int_from_table_16((int) nib.x, kvalues_mxfp4);
    const int2 v1 = get_int_from_table_16((int) nib.y, kvalues_mxfp4);
    const int2 v2 = get_int_from_table_16((int) nib.z, kvalues_mxfp4);
    const int2 v3 = get_int_from_table_16((int) nib.w, kvalues_mxfp4);
    lo = make_uint4((uint32_t) v0.x, (uint32_t) v1.x, (uint32_t) v2.x, (uint32_t) v3.x);
    hi = make_uint4((uint32_t) v0.y, (uint32_t) v1.y, (uint32_t) v2.y, (uint32_t) v3.y);
}

// Stock MMQ folds a 0.5f into the MXFP4 tile scale
// (x_df = ggml_cuda_e8m0_to_fp32(e)*0.5f). Reproduce it exactly or the result is
// off by a factor of two per block rather than merely imprecise.
template <ggml_type WT>
static __device__ __forceinline__ float rp_scale_from_slot(const uint16_t s) {
    if constexpr (WT == GGML_TYPE_MXFP4) {
        return ggml_cuda_e8m0_to_fp32((uint8_t) s) * 0.5f;
    } else {
        return __half2float(*reinterpret_cast<const __half *>(&s));
    }
}
#endif

template <int CW>
static __device__ __forceinline__ int sX_swizzle(int lr) {
    if constexpr (CW == 64) {
        const int n  = lr >> 6;
        int       tx = lr & 63;
        tx ^= (tx >> 5) << 4;
        return (n << 6) | tx;
    } else {
        const int n  = lr >> 5;
        int       tx = lr & 31;
        tx ^= (tx >> 4) << 3;
        return (n << 5) | tx;
    }
}

struct rp_x_sub {
    uint4 q0, q1;
    float d;
};

struct block_q8_1_mmq_h {
    float  d4[4];
    int8_t qs[QK8_1_MMQ];
};

static_assert(sizeof(block_q8_1_mmq_h) == sizeof(block_q8_1_mmq),
              "Unexpected block_q8_1_mmq_h size");
static_assert(offsetof(block_q8_1_mmq_h, d4) == offsetof(block_q8_1_mmq, d4),
              "block_q8_1_mmq_h d4 offset mismatch");
static_assert(offsetof(block_q8_1_mmq_h, qs) == offsetof(block_q8_1_mmq, qs),
              "block_q8_1_mmq_h qs offset mismatch");

struct sXq_row_q8 {
    uint4 q[MMQ_RP_Q8_BK][2];
    uint4 pad;
};

static_assert(sizeof(sXq_row_q8) == (2 * MMQ_RP_Q8_BK + 1) * 16,
              "unexpected sXq row size");

static __device__ __forceinline__ uint4 rp_ldcs_u4(const uint4 * __restrict__ p) {
    return *p;
}

#if defined(GGML_USE_HIP) && defined(__gfx906__)

#define RP_DPP_ADD(name, nop, dpp_ctrl)                                      \
    static __device__ __forceinline__ float name(const float x) {            \
        float r;                                                             \
        asm volatile(                                                        \
            nop                                                              \
            "v_add_f32_dpp %0, %1, %1 " dpp_ctrl " row_mask:0xf bank_mask:0xf" \
            : "=v"(r) : "v"(x) : "memory");                                  \
        return r;                                                            \
    }

RP_DPP_ADD(rp_dpp_add_xor1, "s_nop 4\n", "quad_perm:[1,0,3,2]")
RP_DPP_ADD(rp_dpp_add_xor2, "s_nop 1\n", "quad_perm:[2,3,0,1]")
RP_DPP_ADD(rp_dpp_add_xor8, "s_nop 1\n", "row_ror:8")

#undef RP_DPP_ADD

static __device__ __forceinline__ float rp_dpp_xfer_xor4(const float x) {
    int d;
    asm volatile("v_mov_b32 %0, %1\n"
                 "s_nop 1\n"
                 "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
                 "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
                 : "=v"(d) : "v"(__float_as_int(x)) : "memory");
    return __int_as_float(d);
}

static __device__ __forceinline__ float rp_dpp_xfer_xor16(const float x) {
    int d;
    asm volatile("ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n"
                 "s_waitcnt lgkmcnt(0)\n"
                 : "=v"(d) : "v"(__float_as_int(x)) : "memory");
    return __int_as_float(d);
}

template <int width>
static __device__ __forceinline__ float rp_warp_reduce_sum(const float x) {
    static_assert(width >= 1 && (width & (width - 1)) == 0);
    float r = x;
    if constexpr (width >=  2) { r = rp_dpp_add_xor1(r); }
    if constexpr (width >=  4) { r = rp_dpp_add_xor2(r); }
    if constexpr (width >=  8) { r += rp_dpp_xfer_xor4(r); }
    if constexpr (width >= 16) { r = rp_dpp_add_xor8(r); }
    if constexpr (width >= 32) { r += rp_dpp_xfer_xor16(r); }
    if constexpr (width >= 64) { r += __shfl_xor_sync(0xffffffff, r, 32, 64); }
    return r;
}

#else

template <int width>
static __device__ __forceinline__ float rp_warp_reduce_sum(const float x) {
    return warp_reduce_sum<width>(x);
}

#endif

__device__ __forceinline__ rp_x_sub rp_x_sub_from_mmq_group(
        const block_q8_1_mmq_h * __restrict__ group, const uint32_t col, const uint32_t lk) {
    const block_q8_1_mmq_h & m = group[col];
    const uint4           * mq = reinterpret_cast<const uint4 *>(m.qs + lk * QK8_1);
    rp_x_sub out;
    out.q0 = mq[0];
    out.q1 = mq[1];
    out.d  = m.d4[lk];
    return out;
}

// ---- per-type traits ---------------------------------------------------------
// Everything the shared kernels need to know about a repacked quant type.
// Adding a type = one specialization here + one line in RP_FOREACH_TYPE below
// (plus host/device extract functions until those join the traits too).
//   geom    - per-tensor constants, built once per kernel launch
//   load_w  - fetch sub-block (wrow, sb): 32 int8 values as a lo/hi uint4 pair
//             plus the raw scale slot. THE hot path.
//   scale   - decode a raw scale slot to float.
template <ggml_type T> struct rp_traits;

template <> struct rp_traits<GGML_TYPE_Q8_0> {
    static constexpr bool raw_lds = false;
    // de-aliased qs rows [ne1 x rs], then an f16 scale plane [ne1 x n_sub].
    struct geom {
        uint32_t rs, rs_u4, n_sub;
        size_t   dplane_off;
        __host__ __device__ geom(uint32_t ne0, uint32_t ne1)
            : rs(repack_qs_row_stride(GGML_TYPE_Q8_0, ne0)), rs_u4(rs >> 4),
              n_sub(ne0 >> 5), dplane_off((size_t) ne1 * rs) {}
    };
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    static __device__ __forceinline__ void load_w(const uint8_t * __restrict__ wbase,
            const geom & g, uint32_t wrow, uint32_t sb, uint4 & lo, uint4 & hi, uint16_t & d) {
        const uint4 * qsp = reinterpret_cast<const uint4 *>(wbase);
        const size_t  q4  = (size_t) wrow * g.rs_u4 + 2 * sb;
        lo = rp_ldcs_u4(qsp + q4);
        hi = rp_ldcs_u4(qsp + q4 + 1);
        d  = reinterpret_cast<const uint16_t *>(wbase + g.dplane_off)[(size_t) wrow * g.n_sub + sb];
    }
    static __device__ __forceinline__ float scale(const uint16_t s) {
        return __half2float(*reinterpret_cast<const __half *>(&s));
    }
#endif
};

template <> struct rp_traits<GGML_TYPE_MXFP4> {
    // de-aliased nibble rows [ne1 x rs], then a 1-byte e8m0 plane [ne1 x n_sub].
    struct geom {
        uint32_t rs, rs_u4, n_sub;
        size_t   dplane_off;
        __host__ __device__ geom(uint32_t ne0, uint32_t ne1)
            : rs(repack_qs_row_stride(GGML_TYPE_MXFP4, ne0)), rs_u4(rs >> 4),
              n_sub(ne0 >> 5), dplane_off((size_t) ne1 * rs) {}
    };
    // The GEMM stages RAW nibbles (one uint4 per sub-block) and expands at
    // consume: half the LDS and half the prefetch registers of the expanded
    // form, which measured 16-22 spilled VGPRs per lane when staged expanded.
    static constexpr bool raw_lds = true;
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    static __device__ __forceinline__ void load_w(const uint8_t * __restrict__ wbase,
            const geom & g, uint32_t wrow, uint32_t sb, uint4 & lo, uint4 & hi, uint16_t & d) {
        const uint4 * qsp = reinterpret_cast<const uint4 *>(wbase);
        rp_mxfp4_expand(rp_ldcs_u4(qsp + (size_t) wrow * g.rs_u4 + sb), lo, hi);
        d = wbase[g.dplane_off + (size_t) wrow * g.n_sub + sb];
    }
    static __device__ __forceinline__ void load_w_raw(const uint8_t * __restrict__ wbase,
            const geom & g, uint32_t wrow, uint32_t sb, uint4 & raw, uint16_t & d) {
        const uint4 * qsp = reinterpret_cast<const uint4 *>(wbase);
        raw = rp_ldcs_u4(qsp + (size_t) wrow * g.rs_u4 + sb);
        d   = wbase[g.dplane_off + (size_t) wrow * g.n_sub + sb];
    }
    static __device__ __forceinline__ float scale(const uint16_t s) {
        return ggml_cuda_e8m0_to_fp32((uint8_t) s) * 0.5f;
    }
#endif
};

// One entry per supported repack type - generates the dispatch switches.
#define RP_FOREACH_TYPE(X)     X(GGML_TYPE_Q8_0)          X(GGML_TYPE_MXFP4)
void repack_q8_0_host(const block_q8_0 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1);
void repack_mxfp4_host(const block_mxfp4 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1);
void repack_host(ggml_type type, const void * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1);

const uint8_t * repack_view_get_cached(
        const ggml_tensor * view, const ggml_tensor * base, cudaStream_t stream);

// Drop cached re-packed views that live inside [base, base+size) - called when the
// owning buffer is freed so the allocator cannot hand the same addresses to a
// different model and hit a stale entry.
void repack_view_cache_purge(int device, const void * base, size_t size);

// Quantize src1 into Q8_1 rows, reusing the graph-wide activation cache when the
// same activation was already quantized to this exact layout. One activation
// feeds several matmuls back to back - q/k/v off one attention norm, router and
// routed gate/up off one ffn norm - and each re-quantized its own copy, so the
// launch count tracked the matmul count exactly. quantize_row_q8_1_cuda ignores
// type_src0 and always writes the plain row layout, so variant 0 is the same
// layout the canonical mat-vec caches under and the two share entries. The MoE
// sites pass the rows flattened as (n_cols, 1, 1) where the dense sites pass
// (ne11, ne12, ne13). That is still the same bytes: the kernel writes
// i_cont = ((i3*ne2 + i2)*ne1 + i1)*ne0 + i0, which is contiguous either way,
// and it can only collide in the cache when the strides and the size already
// match, which is the contiguous case where both walk src1 identically.
// Fusing the quantize into the mat-vec instead is the wrong shape: it is a
// shared producer, and making every block re-quantize the row cost 42 percent.
static inline block_q8_1 * repack_quantize_src1_q8_1(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        int64_t ne10, int64_t ne10_padded, int64_t s11, int64_t s12, int64_t s13,
        int64_t nrows, int64_t ne2, int64_t ne3, size_t nblocks,
        ggml_cuda_pool_alloc<block_q8_1> & fallback, cudaStream_t stream) {
    bool hit = false;
    char * cached = ggml_cuda_q8_1_cache_acquire(ctx, src1, /*variant =*/ 0, ne10_padded,
                                                 s11, s12, s13, nblocks * sizeof(block_q8_1), hit);
    block_q8_1 * dstq;
    if (cached) {
        dstq = (block_q8_1 *) cached;
    } else {
        fallback.alloc(ctx.pool(), nblocks);
        dstq = fallback.get();
    }
    if (!hit) {
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, dstq,
            src0->type, ne10, s11, s12, s13, ne10_padded, nrows, ne2, ne3, stream);
    }
    return dstq;
}
