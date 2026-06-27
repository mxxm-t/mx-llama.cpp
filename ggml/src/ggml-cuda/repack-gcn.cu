#include "repack-gcn.cuh"
#include "convert.cuh"
#include "quantize.cuh"

#include "ggml-backend-impl.h"
#include "mmid.cuh"
#include "unary.cuh"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------
// layout helpers
// ---------------------------------------------------------------------

// Sub-blocks (32 weights) per repacked row, padded by one when the
// natural count is a power of two (a power-of-two row stride aliases
// every row onto the same HBM channel: ~3x matvec penalty). Shared by
// all repacked types.
static __host__ __device__ inline int64_t repack_q4k_nsp(const int64_t ne0) {
    const int64_t n_sub = ne0 / 32;
    return (n_sub & (n_sub - 1)) == 0 ? n_sub + 1 : n_sub;
}

// Plane bytes per type:
//   Q3_K: 8 lo2 + 4 hi1 + 2 signed-scale-pair per sub-block, 2 (d) per superblock
//   Q4_K: 16 nib + 2 sc|m per sub-block, 4 (d|dmin fp16) per superblock
//   Q5_K: 16 nib + 4 qh + 2 sc|m per sub-block, 4 per superblock
//   Q6_K: 16 nib + 8 h2 + 2 signed-scale-pair per sub-block, 2 (d) per superblock
//   Q8_0: 32 qs + 2 (d fp16) per sub-block
static inline size_t repack_gcn_nbytes(const ggml_type type, const int64_t ne0, const int64_t ne1) {
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const int64_t n_blocks = ne0 / 256;
    switch (type) {
        case GGML_TYPE_Q3_K: return (size_t) ne1 * (nsp * 14 + n_blocks * 2);
        case GGML_TYPE_Q4_K: return (size_t) ne1 * (nsp * 18 + n_blocks * 4);
        case GGML_TYPE_Q5_K: return (size_t) ne1 * (nsp * 22 + n_blocks * 4);
        case GGML_TYPE_Q6_K: return (size_t) ne1 * (nsp * 26 + n_blocks * 2);
        case GGML_TYPE_Q8_0: return (size_t) ne1 * nsp * 34;
        default:             GGML_ABORT("unsupported repack type");
    }
}

bool ggml_cuda_repack_tensor_supported(const ggml_tensor * t) {
    // 2D weights (MUL_MAT) or 3D per-expert stacks (MUL_MAT_ID)
    if ((ggml_n_dims(t) != 2 && ggml_n_dims(t) != 3) || !ggml_is_contiguous(t)) {
        return false;
    }
    switch (t->type) {
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K: return t->ne[0] % 256 == 0;
        case GGML_TYPE_Q8_0: {
            // Q8_0 repack is its own opt-in: the repacked MMQ wins
            // prefill big (+43% on a pure-Q8_0 0.8B) but the repacked
            // matvec loses ~6% decode to the canonical mmvq (it was
            // tuned on MoE-expert shapes in reinstinct; on-disk Q8_0 is
            // already nearly contiguous so repack buys less). Re-tune
            // before considering it for default-on.
            static const bool q8 = [] {
                const char * e = getenv("GGML_CUDA_REPACK_Q8_0");
                return e != nullptr && e[0] != '0';
            }();
            return q8 && t->ne[0] % 32 == 0;
        }
        default:             return false;
    }
}

// ---------------------------------------------------------------------
// host-side repack (one-shot at weight upload)
// ---------------------------------------------------------------------

// ggml-quants.c's get_scale_min_k4: unpack sub-block j's 6-bit (sc, m)
// from the 12-byte packed scales array.
static inline void repack_get_scale_min_k4(const int j, const uint8_t * q, uint8_t * sc, uint8_t * m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >>   4) | ((q[j    ] >> 6) << 4);
    }
}

static void repack_q4k_host(const block_q4_K * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 256;
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const size_t  nib_len  = (size_t) ne1 * nsp * 16;
    const size_t  sm_len   = (size_t) ne1 * nsp * 2;

    // The padding sub-block (when nsp != ne0/32) must read as zero
    // weights with zero scales so the kernel can include it harmlessly.
    memset(dst, 0, nib_len + sm_len + (size_t) ne1 * n_blocks * 4);

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q4_K * b = &blocks[row * n_blocks + blk];

            // superblock plane: raw fp16 d, dmin per 256 weights
            // (block_q4_K starts with d at byte 0, dmin at byte 2 —
            // guaranteed by the ggml-common.h size/layout asserts)
            uint8_t * dd = dst + nib_len + sm_len + (size_t)(row * n_blocks + blk) * 4;
            memcpy(dd, b, 4);

            for (int s = 0; s < 8; s++) {
                const int64_t gsb = blk * 8 + s; // sub-block index within the row

                // this sub-block's 32 nibble weights: qs bytes (s/2)*32..+32,
                // even sub-blocks take low nibbles, odd take high
                const uint8_t * qs = b->qs + (s >> 1) * 32;
                uint8_t w[32];
                if ((s & 1) == 0) {
                    for (int k = 0; k < 32; k++) { w[k] = qs[k] & 0x0F; }
                } else {
                    for (int k = 0; k < 32; k++) { w[k] = qs[k] >> 4; }
                }

                // nibble plane: byte 4j+b = w[4j+b] | (w[16+4j+b] << 4),
                // so uint32 j feeds dp4a with weights 4j..4j+3 / 16+4j..+3
                uint8_t * nib = dst + (size_t)(row * nsp + gsb) * 16;
                for (int j = 0; j < 4; j++) {
                    for (int bb = 0; bb < 4; bb++) {
                        nib[j * 4 + bb] = w[4 * j + bb] | (w[16 + 4 * j + bb] << 4);
                    }
                }

                // scale plane: 6-bit sc then m as two u8
                uint8_t sc, m;
                repack_get_scale_min_k4(s, b->scales, &sc, &m);
                uint8_t * sm = dst + nib_len + (size_t)(row * nsp + gsb) * 2;
                sm[0] = sc;
                sm[1] = m;
            }
        }
    }
}

// Q5_K: like Q4_K plus a qh plane — per sub-block one u32 whose bit
// 4g+b is the 5th bit of weight b of dp4a group g.
static void repack_q5k_host(const block_q5_K * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 256;
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const size_t  nib_len  = (size_t) ne1 * nsp * 16;
    const size_t  qh_len   = (size_t) ne1 * nsp * 4;
    const size_t  sm_len   = (size_t) ne1 * nsp * 2;

    memset(dst, 0, nib_len + qh_len + sm_len + (size_t) ne1 * n_blocks * 4);

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q5_K * b = &blocks[row * n_blocks + blk];

            uint8_t * dd = dst + nib_len + qh_len + sm_len + (size_t)(row * n_blocks + blk) * 4;
            memcpy(dd, b, 4); // fp16 d, dmin lead the block

            for (int s = 0; s < 8; s++) {
                const int64_t gsb = blk * 8 + s;
                const uint8_t * qs = b->qs + (s >> 1) * 32;

                uint8_t w[32], hb[32];
                for (int k = 0; k < 32; k++) {
                    w[k]  = ((s & 1) == 0) ? (qs[k] & 0x0F) : (qs[k] >> 4);
                    hb[k] = (b->qh[k] >> s) & 1;
                }

                uint8_t * nib = dst + (size_t)(row * nsp + gsb) * 16;
                uint32_t qh_packed = 0;
                for (int j = 0; j < 4; j++) {
                    for (int bb = 0; bb < 4; bb++) {
                        nib[j * 4 + bb] = w[4 * j + bb] | (w[16 + 4 * j + bb] << 4);
                        // dp4a group 2j holds weights 4j+bb, group 2j+1 holds 16+4j+bb
                        qh_packed |= (uint32_t) hb[4 * j + bb]      << (4 * (2 * j)     + bb);
                        qh_packed |= (uint32_t) hb[16 + 4 * j + bb] << (4 * (2 * j + 1) + bb);
                    }
                }
                memcpy(dst + nib_len + (size_t)(row * nsp + gsb) * 4, &qh_packed, 4);

                uint8_t sc, m;
                repack_get_scale_min_k4(s, b->scales, &sc, &m);
                uint8_t * sm = dst + nib_len + qh_len + (size_t)(row * nsp + gsb) * 2;
                sm[0] = sc;
                sm[1] = m;
            }
        }
    }
}

// Q6_K: nibble plane + 8-byte h2 plane (the 6-bit quant's high pair per
// weight, 2 bits at position 2b of byte g) + signed per-16-weight scale
// pairs + d-only superblock plane.
static void repack_q6k_host(const block_q6_K * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 256;
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const size_t  nib_len  = (size_t) ne1 * nsp * 16;
    const size_t  h2_len   = (size_t) ne1 * nsp * 8;
    const size_t  sm_len   = (size_t) ne1 * nsp * 2;

    memset(dst, 0, nib_len + h2_len + sm_len + (size_t) ne1 * n_blocks * 2);

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q6_K * b = &blocks[row * n_blocks + blk];

            memcpy(dst + nib_len + h2_len + sm_len + (size_t)(row * n_blocks + blk) * 2, &b->d, 2);

            for (int s = 0; s < 8; s++) {
                const int64_t gsb  = blk * 8 + s;
                const int     chunk = s / 4;
                const int     quad  = s % 4;
                const int     ql_off = chunk * 64;
                const int     qh_off = chunk * 32;

                uint8_t lo[32], hi[32];
                for (int k = 0; k < 32; k++) {
                    const uint8_t qh = b->qh[qh_off + k];
                    switch (quad) {
                        case 0:  lo[k] = b->ql[ql_off + k]      & 0x0F; hi[k] =  qh       & 3; break;
                        case 1:  lo[k] = b->ql[ql_off + k + 32] & 0x0F; hi[k] = (qh >> 2) & 3; break;
                        case 2:  lo[k] = b->ql[ql_off + k]      >> 4;   hi[k] = (qh >> 4) & 3; break;
                        default: lo[k] = b->ql[ql_off + k + 32] >> 4;   hi[k] = (qh >> 6) & 3; break;
                    }
                }

                uint8_t * nib = dst + (size_t)(row * nsp + gsb) * 16;
                uint8_t h2p[8] = {};
                for (int j = 0; j < 4; j++) {
                    for (int bb = 0; bb < 4; bb++) {
                        nib[j * 4 + bb] = lo[4 * j + bb] | (lo[16 + 4 * j + bb] << 4);
                        h2p[2 * j]     |= hi[4 * j + bb]      << (2 * bb);
                        h2p[2 * j + 1] |= hi[16 + 4 * j + bb] << (2 * bb);
                    }
                }
                memcpy(dst + nib_len + (size_t)(row * nsp + gsb) * 8, h2p, 8);

                uint8_t * sm = dst + nib_len + h2_len + (size_t)(row * nsp + gsb) * 2;
                sm[0] = (uint8_t) b->scales[chunk * 8 + quad * 2];
                sm[1] = (uint8_t) b->scales[chunk * 8 + quad * 2 + 1];
            }
        }
    }
}

// Q3_K: the 3-bit quant stays sub-nibble to fit VRAM. A lo2 plane (low 2
// bits, packed like Q6_K's h2) and a hi1 plane (high bit, packed like
// Q5_K's qh) reconstruct q3 = lo2 | (hbit << 2) at compute. Symmetric
// like Q6_K (bias 4) with a signed per-16-weight scale pair (unpacked
// 6-bit scale minus 32) and a d-only superblock plane.
static void repack_q3k_host(const block_q3_K * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 256;
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const size_t  lo2_len  = (size_t) ne1 * nsp * 8;
    const size_t  hi1_len  = (size_t) ne1 * nsp * 4;
    const size_t  sm_len   = (size_t) ne1 * nsp * 2;

    memset(dst, 0, lo2_len + hi1_len + sm_len + (size_t) ne1 * n_blocks * 2);

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q3_K * b = &blocks[row * n_blocks + blk];

            memcpy(dst + lo2_len + hi1_len + sm_len + (size_t)(row * n_blocks + blk) * 2, &b->d, 2);

            // ggml-quants.c dequantize_row_q3_K: unpack the 16 6-bit scales
            uint32_t aux[4];
            memcpy(aux, b->scales, 12);
            const uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = ( aux[0]       & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = ( aux[1]       & kmask2) | (((tmp >> 2) & kmask1) << 4);
            const uint8_t * sc6 = (const uint8_t *) aux;

            for (int s = 0; s < 8; s++) {
                const int64_t gsb   = blk * 8 + s;
                const int     n     = s >> 2;
                const int     shift = 2 * (s & 3);
                const uint8_t * qs  = b->qs + n * 32;

                uint8_t  lo2[32], hb[32];
                for (int k = 0; k < 32; k++) {
                    lo2[k] = (qs[k] >> shift) & 3;
                    hb[k]  = (b->hmask[k] >> s) & 1;
                }

                // lo2 plane: byte 2j holds the low-half group j (weights
                // 4j..), byte 2j+1 the high-half (weights 16+4j..), weight
                // b at bits 2b..2b+1 -- the Q6_K h2 packing.
                uint8_t lo2p[8] = {};
                uint32_t hi1p = 0;
                for (int j = 0; j < 4; j++) {
                    for (int bb = 0; bb < 4; bb++) {
                        lo2p[2 * j]     |= lo2[4 * j + bb]      << (2 * bb);
                        lo2p[2 * j + 1] |= lo2[16 + 4 * j + bb] << (2 * bb);
                        hi1p |= (uint32_t) hb[4 * j + bb]      << (8 * j + bb);
                        hi1p |= (uint32_t) hb[16 + 4 * j + bb] << (8 * j + 4 + bb);
                    }
                }
                memcpy(dst + (size_t)(row * nsp + gsb) * 8, lo2p, 8);
                memcpy(dst + lo2_len + (size_t)(row * nsp + gsb) * 4, &hi1p, 4);

                uint8_t * sm = dst + lo2_len + hi1_len + (size_t)(row * nsp + gsb) * 2;
                sm[0] = (uint8_t) (int8_t) ((int) sc6[2 * s]     - 32);
                sm[1] = (uint8_t) (int8_t) ((int) sc6[2 * s + 1] - 32);
            }
        }
    }
}

// Q8_0: two planes — 32 aligned qs bytes per sub-block, then the fp16
// d-scales as their own stream. Same bytes as on-disk modulo padding;
// the win is alignment (one i32 load per sdot4 instead of two
// uint16 loads OR-shifted around the on-disk 2-byte offset).
static void repack_q8_0_host(const block_q8_0 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 32;
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const size_t  qs_len   = (size_t) ne1 * nsp * 32;

    memset(dst, 0, qs_len + (size_t) ne1 * nsp * 2);

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q8_0 * b = &blocks[row * n_blocks + blk];
            memcpy(dst + (size_t)(row * nsp + blk) * 32, b->qs, 32);
            memcpy(dst + qs_len + (size_t)(row * nsp + blk) * 2, &b->d, 2);
        }
    }
}

// ---------------------------------------------------------------------
// kernels (GCN only — guarded so non-HIP / non-GCN builds still compile)
// ---------------------------------------------------------------------

// --- MUL_MAT_ID support -----------------------------------------------
// Expert routing comes compacted from ggml_cuda_launch_mm_ids_helper:
// assignment index a in [0, n_assign) is expert-sorted; expert_bounds
// gives each expert's [start, end) range; ids_src1[a] is the flat
// column index into the naturally-ordered activation buffer; ids_dst[a]
// is the flat destination column. Weights for expert e live at
// wbase + e * expert_stride (per-expert repacked slabs, identical
// layout to the 2D case).

// Largest e with expert_bounds[e] <= a.
static __device__ __forceinline__ uint32_t repack_find_expert(
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert, const uint32_t a) {
    uint32_t lo = 0, hi = n_expert;
    while (lo + 1 < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        if ((uint32_t) expert_bounds[mid] <= a) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// tile_off[e] = prefix sum of per-expert token-tile counts (BN-sized
// tiles); single-thread kernel, n_expert <= a few hundred.
template <int BN>
static __global__ void repack_tile_off(
        const int32_t * __restrict__ expert_bounds, int32_t * __restrict__ tile_off,
        const int n_expert) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    int acc = 0;
    tile_off[0] = 0;
    for (int e = 0; e < n_expert; e++) {
        const int cnt = expert_bounds[e + 1] - expert_bounds[e];
        acc += (cnt + BN - 1) / BN;
        tile_off[e + 1] = acc;
    }
}


// Repacked Q4_K matvec. Block = 256 threads = 4 wave64s; each wave
// computes ROWS=2 output rows; lane l streams sub-block l, l+64, ... —
// consecutive lanes read consecutive 16-byte chunks, a fully-coalesced
// sweep of the nibble plane.
//
//   dot = sum_sub [ (d*sc) * dx * <nibbles . q8> - (dmin*m) * sx ]
//
// with (dx, sx) = block_q8_1.ds — sx is dx * sum(q8), which is exactly
// the dequantized sub-block sum the min-term needs (same contract as
// vec_dot_q4_K_q8_1).
template <bool HAS_IDS>
static __global__ void mul_mat_vec_q4k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        // decode (one token): slot a maps directly — expert ids_raw[a],
        // activation column a, dst column a. No compaction kernel needed
        // (ids_src1 carries the RAW ids tensor here).
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        wbase += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    }
    constexpr int ROWS = 2;
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;

    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16);
    const uint32_t * ddp = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 2);
    const uint32_t n_super = n_sub >> 3;

    float acc[ROWS] = {0.0f, 0.0f};

    // ID path (experts) uses 16-weight half-sub-block units: expert
    // tensors are small-K (down-proj K=768 -> 24 sub-blocks for 64
    // lanes) and full units leave most of the wave idle. The -deff*sx
    // min term is applied by the even half only.
    const uint32_t n_unit = HAS_IDS ? n_sub * 2 : n_sub;
    for (uint32_t u = lane; u < n_unit; u += 64) {
        const uint32_t sb   = HAS_IDS ? (u >> 1) : u;
        const uint32_t half = HAS_IDS ? (u & 1)  : 0;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const float sx = __high2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs);

#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int) ne1) {
                continue;
            }

            const uint4    q  = nib[(size_t) row * nsp + sb];
            const uint16_t sm = smp[(size_t) row * nsp + sb];
            const uint32_t dd = ddp[(size_t) row * n_super + (sb >> 3)];
            const uint16_t d_bits    = (uint16_t)(dd & 0xFFFF);
            const uint16_t dmin_bits = (uint16_t)(dd >> 16);
            const float dsc  = __half2float(*reinterpret_cast<const __half *>(&d_bits))
                               * (float)(sm & 0xFFu);
            const float deff = __half2float(*reinterpret_cast<const __half *>(&dmin_bits))
                               * (float)(sm >> 8);

            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
            const int j0 = HAS_IDS ? (int)(half * 2) : 0;
            const int j1 = HAS_IDS ? j0 + 2          : 4;
            int idot = 0;
#pragma unroll
            for (int j = j0; j < j1; j++) {
                idot = ggml_cuda_dp4a((int)( qa[j]       & 0x0F0F0F0Fu), xq32[j],     idot);
                idot = ggml_cuda_dp4a((int)((qa[j] >> 4) & 0x0F0F0F0Fu), xq32[j + 4], idot);
            }
            acc[r] += dsc * dx * (float) idot - (half == 0 ? deff * sx : 0.0f);
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const float a = warp_reduce_sum<64>(acc[r]);
        if (lane == 0 && (row0 + r) < (int) ne1) {
            y[row0 + r] = a;
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Spread 4 bits (b0..b3) to bit 4 of bytes 0..3 — positions the Q5_K
// high bit above the dp4a nibble lanes.
static __device__ __forceinline__ uint32_t repack_spread4(const uint32_t h) {
    return ((h & 1u) << 4) | ((h & 2u) << 11) | ((h & 4u) << 18) | ((h & 8u) << 25);
}

// Spread four 2-bit fields (weight b at bits 2b..2b+1) to bits 4..5 of
// bytes 0..3 — the Q6_K quant's high pair.
static __device__ __forceinline__ uint32_t repack_spread2(const uint32_t h) {
    return ((h & 0x03u) << 4) | ((h & 0x0Cu) << 10)
         | ((h & 0x30u) << 16) | ((h & 0xC0u) << 22);
}

// Q3_K reconstruction: four 2-bit fields to bits 0..1 of bytes 0..3 (the
// low pair) and four 1-bit fields to bit 2 of bytes 0..3 (the high bit).
static __device__ __forceinline__ uint32_t repack_spread2_lo(const uint32_t h) {
    return (h & 0x03u) | ((h & 0x0Cu) << 6)
         | ((h & 0x30u) << 12) | ((h & 0xC0u) << 18);
}
static __device__ __forceinline__ uint32_t repack_spread1_hi(const uint32_t h) {
    return ((h & 1u) << 2) | ((h & 2u) << 9) | ((h & 4u) << 16) | ((h & 8u) << 23);
}

// Q5_K repacked matvec — Q4_K's shape plus the qh plane OR-ed onto the
// nibbles before each dp4a.
template <bool HAS_IDS>
static __global__ void mul_mat_vec_q5k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        // decode (one token): slot a maps directly — expert ids_raw[a],
        // activation column a, dst column a. No compaction kernel needed
        // (ids_src1 carries the RAW ids tensor here).
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        wbase += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    }
    constexpr int ROWS = 2;
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;

    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint32_t * qhp = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 16);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 4);
    const uint32_t * ddp = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 4 + (size_t) ne1 * nsp * 2);

    float acc[ROWS] = {0.0f, 0.0f};

    const uint32_t n_unit = HAS_IDS ? n_sub * 2 : n_sub; // see Q4_K note
    for (uint32_t u = lane; u < n_unit; u += 64) {
        const uint32_t sb   = HAS_IDS ? (u >> 1) : u;
        const uint32_t half = HAS_IDS ? (u & 1)  : 0;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const float sx = __high2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs);

#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int) ne1) {
                continue;
            }
            const size_t   idx = (size_t) row * nsp + sb;
            const uint4    q   = nib[idx];
            const uint32_t qh  = qhp[idx];
            const uint16_t sm  = smp[idx];
            const uint32_t dd  = ddp[(size_t) row * n_super + (sb >> 3)];
            const uint16_t d_bits    = (uint16_t)(dd & 0xFFFF);
            const uint16_t dmin_bits = (uint16_t)(dd >> 16);
            const float dsc  = __half2float(*reinterpret_cast<const __half *>(&d_bits))
                               * (float)(sm & 0xFFu);
            const float deff = __half2float(*reinterpret_cast<const __half *>(&dmin_bits))
                               * (float)(sm >> 8);

            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
            const int j0 = HAS_IDS ? (int)(half * 2) : 0;
            const int j1 = HAS_IDS ? j0 + 2          : 4;
            int idot = 0;
#pragma unroll
            for (int j = j0; j < j1; j++) {
                const uint32_t lo = ( qa[j]       & 0x0F0F0F0Fu)
                    | repack_spread4((qh >> (8 * j))     & 0xFu);
                const uint32_t hi = ((qa[j] >> 4) & 0x0F0F0F0Fu)
                    | repack_spread4((qh >> (8 * j + 4)) & 0xFu);
                idot = ggml_cuda_dp4a((int) lo, xq32[j],     idot);
                idot = ggml_cuda_dp4a((int) hi, xq32[j + 4], idot);
            }
            acc[r] += dsc * dx * (float) idot - (half == 0 ? deff * sx : 0.0f);
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const float a = warp_reduce_sum<64>(acc[r]);
        if (lane == 0 && (row0 + r) < (int) ne1) {
            y[row0 + r] = a;
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q6_K repacked matvec. Symmetric quant (value = q-32); the offset is
// folded out via activation half-sums: sum (q-32)x = sum qx - 32 sum x.
// Two signed scales per sub-block, one per 16 weights.
template <bool HAS_IDS>
static __global__ void mul_mat_vec_q6k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        // decode (one token): slot a maps directly — expert ids_raw[a],
        // activation column a, dst column a. No compaction kernel needed
        // (ids_src1 carries the RAW ids tensor here).
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        wbase += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    }
    constexpr int ROWS = 2;
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;

    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint32_t * h2p = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 16);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 8);
    const uint16_t * ddp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 8 + (size_t) ne1 * nsp * 2);

    float acc[ROWS] = {0.0f, 0.0f};

    const uint32_t n_unit = HAS_IDS ? n_sub * 2 : n_sub; // see Q4_K note
    for (uint32_t u = lane; u < n_unit; u += 64) {
        const uint32_t sb   = HAS_IDS ? (u >> 1) : u;
        const uint32_t half = HAS_IDS ? (u & 1)  : 0;
        const int hj0 = HAS_IDS ? (int)(half * 2) : 0;
        const int hj1 = HAS_IDS ? hj0 + 2         : 4;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs);

        // per-half activation sums: the -32 fold splits with them
        int xis0 = 0, xis1 = 0;
#pragma unroll
        for (int j = hj0; j < hj1; j++) {
            xis0 = ggml_cuda_dp4a(xq32[j],     0x01010101, xis0);
            xis1 = ggml_cuda_dp4a(xq32[j + 4], 0x01010101, xis1);
        }

#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int) ne1) {
                continue;
            }
            const size_t   idx  = (size_t) row * nsp + sb;
            const uint4    q    = nib[idx];
            const uint32_t h2lo = h2p[idx * 2];
            const uint32_t h2hi = h2p[idx * 2 + 1];
            const uint16_t sm     = smp[idx];
            const uint16_t d_bits = ddp[(size_t) row * n_super + (sb >> 3)];
            const float d = __half2float(*reinterpret_cast<const __half *>(&d_bits));
            const float dsc_lo = d * (float)(int)(int8_t)(sm & 0xFFu);
            const float dsc_hi = d * (float)(int)(int8_t)(sm >> 8);

            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
            int idot0 = 0, idot1 = 0;
#pragma unroll
            for (int j = hj0; j < hj1; j++) {
                const uint32_t ge = 2 * j;
                const uint32_t go = 2 * j + 1;
                const uint32_t he = ((ge < 4 ? h2lo : h2hi) >> (8 * (ge & 3))) & 0xFFu;
                const uint32_t ho = ((go < 4 ? h2lo : h2hi) >> (8 * (go & 3))) & 0xFFu;
                const uint32_t q6lo = ( qa[j]       & 0x0F0F0F0Fu) | repack_spread2(he);
                const uint32_t q6hi = ((qa[j] >> 4) & 0x0F0F0F0Fu) | repack_spread2(ho);
                idot0 = ggml_cuda_dp4a((int) q6lo, xq32[j],     idot0);
                idot1 = ggml_cuda_dp4a((int) q6hi, xq32[j + 4], idot1);
            }
            acc[r] += dsc_lo * dx * (float)(idot0 - 32 * xis0)
                    + dsc_hi * dx * (float)(idot1 - 32 * xis1);
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const float a = warp_reduce_sum<64>(acc[r]);
        if (lane == 0 && (row0 + r) < (int) ne1) {
            y[row0 + r] = a;
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q3_K repacked matvec. Like Q6_K but the quant is 2-bit lo + 1-bit hi
// (no 4-bit nibble plane); reconstruct q3 = lo2 | (hbit << 2) per group.
// Symmetric with bias 4.
template <bool HAS_IDS>
static __global__ void mul_mat_vec_q3k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        wbase += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    }
    constexpr int ROWS = 2;
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;

    const uint2    * lo2p = reinterpret_cast<const uint2 *>(wbase);
    const uint32_t * hi1p = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 8);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 8 + (size_t) ne1 * nsp * 4);
    const uint16_t * ddp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 8 + (size_t) ne1 * nsp * 4 + (size_t) ne1 * nsp * 2);

    float acc[ROWS] = {0.0f, 0.0f};

    const uint32_t n_unit = HAS_IDS ? n_sub * 2 : n_sub; // see Q4_K note
    for (uint32_t u = lane; u < n_unit; u += 64) {
        const uint32_t sb   = HAS_IDS ? (u >> 1) : u;
        const uint32_t half = HAS_IDS ? (u & 1)  : 0;
        const int hj0 = HAS_IDS ? (int)(half * 2) : 0;
        const int hj1 = HAS_IDS ? hj0 + 2         : 4;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs);

        int xis0 = 0, xis1 = 0;
#pragma unroll
        for (int j = hj0; j < hj1; j++) {
            xis0 = ggml_cuda_dp4a(xq32[j],     0x01010101, xis0);
            xis1 = ggml_cuda_dp4a(xq32[j + 4], 0x01010101, xis1);
        }

#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int) ne1) {
                continue;
            }
            const size_t   idx    = (size_t) row * nsp + sb;
            const uint2    lo2v   = lo2p[idx];
            const uint32_t qh     = hi1p[idx];
            const uint16_t sm     = smp[idx];
            const uint16_t d_bits = ddp[(size_t) row * n_super + (sb >> 3)];
            const float d = __half2float(*reinterpret_cast<const __half *>(&d_bits));
            const float dsc_lo = d * (float)(int)(int8_t)(sm & 0xFFu);
            const float dsc_hi = d * (float)(int)(int8_t)(sm >> 8);

            const uint32_t lo2lo = lo2v.x, lo2hi = lo2v.y;
            int idot0 = 0, idot1 = 0;
#pragma unroll
            for (int j = hj0; j < hj1; j++) {
                const uint32_t ge = 2 * j;
                const uint32_t go = 2 * j + 1;
                const uint32_t lb = ((ge < 4 ? lo2lo : lo2hi) >> (8 * (ge & 3))) & 0xFFu;
                const uint32_t hb = ((go < 4 ? lo2lo : lo2hi) >> (8 * (go & 3))) & 0xFFu;
                const uint32_t q3lo = repack_spread2_lo(lb) | repack_spread1_hi((qh >> (8 * j))     & 0xFu);
                const uint32_t q3hi = repack_spread2_lo(hb) | repack_spread1_hi((qh >> (8 * j + 4)) & 0xFu);
                idot0 = ggml_cuda_dp4a((int) q3lo, xq32[j],     idot0);
                idot1 = ggml_cuda_dp4a((int) q3hi, xq32[j + 4], idot1);
            }
            acc[r] += dsc_lo * dx * (float)(idot0 - 4 * xis0)
                    + dsc_hi * dx * (float)(idot1 - 4 * xis1);
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const float a = warp_reduce_sum<64>(acc[r]);
        if (lane == 0 && (row0 + r) < (int) ne1) {
            y[row0 + r] = a;
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q8_0 repacked matvec — NWAVES wave64s per block, ROWS output rows
// per wave. The original reinstinct tuning used single-wave blocks
// (NWAVES=1) for its MoE-expert shapes; small dense models want the
// 4-wave shape the K-quant matvecs use (NWAVES=4). ROWS=1 doubles the
// wavefront count and wins at out_dim >= 4096 where ROWS=2 leaves too
// few wavefront generations in flight to sustain HBM bandwidth.
template <int ROWS, int NWAVES, bool HAS_IDS>
static __global__ void mul_mat_vec_q8_0_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        // decode (one token): slot a maps directly — expert ids_raw[a],
        // activation column a, dst column a. No compaction kernel needed
        // (ids_src1 carries the RAW ids tensor here).
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        wbase += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    }
    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t nsp = ((n_blocks & (n_blocks - 1u)) == 0u) ? (n_blocks + 1u) : n_blocks;

    const int      * qs_int  = reinterpret_cast<const int *>(wbase);
    const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 32);

    const int wave = threadIdx.x >> 6;
    const int row0 = blockIdx.x * (ROWS * NWAVES) + wave * ROWS;
    const int lane = threadIdx.x & 63;

    float acc[ROWS] = {};

    // Work unit: a 16-weight half sub-block (4 sdot4s). At small ne0 a
    // full-sub-block unit leaves lanes idle (ne0=1024 -> 32 sub-blocks
    // for 64 lanes); halves keep the wave full down to ne0=1024.
    // (8-weight quarters were tried and regress: 4x the d-plane traffic
    // outweighs the extra balance.)
    const uint32_t n_half = n_blocks * 2;
    for (uint32_t hb = lane; hb < n_half; hb += 64) {
        const uint32_t sb   = hb >> 1;
        const uint32_t half = hb & 1;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs) + half * 4;

#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int) ne1) {
                continue;
            }
            const int      * w_int = qs_int + ((size_t) row * nsp + sb) * 8 + half * 4;
            const uint16_t   db    = d_plane[(size_t) row * nsp + sb];
            const float      dw    = __half2float(*reinterpret_cast<const __half *>(&db));

            int idot = 0;
#pragma unroll
            for (int g = 0; g < 4; g++) {
                idot = ggml_cuda_dp4a(w_int[g], xq32[g], idot);
            }
            acc[r] += dw * dx * (float) idot;
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const float a = warp_reduce_sum<64>(acc[r]);
        if (lane == 0 && (row0 + r) < (int) ne1) {
            y[row0 + r] = a;
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}


// Fused gate+up Q4_K matvec with GLU epilogue. Walks BOTH weight slabs
// in one sub-block loop (one activation read, one launch) and writes
// y[row] = glu(gate_dot) * up_dot — replacing two matvec launches plus
// an elementwise GLU op. This is what canonical mmvq fuses too; without
// it the repacked MoE decode pays ~2x the launches (measured -7% on
// 35B-A3B). Used for both dense MUL_MAT (ids == nullptr) and
// MUL_MAT_ID decode. ID path uses half-sub-block units (small-K expert
// tensors; min term on the even half).
template <bool HAS_IDS>
static __global__ void mul_mat_vec_q4k_repacked_glu(
        const uint8_t * __restrict__ wup, const uint8_t * __restrict__ wgate,
        const block_q8_1 * __restrict__ xq, float * __restrict__ y,
        const uint32_t ne0, const uint32_t ne1, const int glu_op,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        const uint32_t a = blockIdx.y; // slot index; see direct-map note above
        const uint32_t e = (uint32_t) ids_src1[a];
        wup   += e * expert_stride;
        wgate += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    }
    constexpr int ROWS = 2;
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;

    const uint8_t * wb[2] = { wup, wgate };
    float acc[2][ROWS] = {};

    const uint32_t n_unit = HAS_IDS ? n_sub * 2 : n_sub;
    for (uint32_t u = lane; u < n_unit; u += 64) {
        const uint32_t sb   = HAS_IDS ? (u >> 1) : u;
        const uint32_t half = HAS_IDS ? (u & 1)  : 0;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const float sx = __high2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs);

#pragma unroll
        for (int w2 = 0; w2 < 2; w2++) {
            const uint4    * nib = reinterpret_cast<const uint4 *>(wb[w2]);
            const uint16_t * smp = reinterpret_cast<const uint16_t *>(
                wb[w2] + (size_t) ne1 * nsp * 16);
            const uint32_t * ddp = reinterpret_cast<const uint32_t *>(
                wb[w2] + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 2);
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const int row = row0 + r;
                if (row >= (int) ne1) {
                    continue;
                }
                const uint4    q  = nib[(size_t) row * nsp + sb];
                const uint16_t sm = smp[(size_t) row * nsp + sb];
                const uint32_t dd = ddp[(size_t) row * n_super + (sb >> 3)];
                const uint16_t d_bits    = (uint16_t)(dd & 0xFFFF);
                const uint16_t dmin_bits = (uint16_t)(dd >> 16);
                const float dsc  = __half2float(*reinterpret_cast<const __half *>(&d_bits))
                                   * (float)(sm & 0xFFu);
                const float deff = __half2float(*reinterpret_cast<const __half *>(&dmin_bits))
                                   * (float)(sm >> 8);

                const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
                const int j0 = HAS_IDS ? (int)(half * 2) : 0;
                const int j1 = HAS_IDS ? j0 + 2          : 4;
                int idot = 0;
#pragma unroll
                for (int j = j0; j < j1; j++) {
                    idot = ggml_cuda_dp4a((int)( qa[j]       & 0x0F0F0F0Fu), xq32[j],     idot);
                    idot = ggml_cuda_dp4a((int)((qa[j] >> 4) & 0x0F0F0F0Fu), xq32[j + 4], idot);
                }
                acc[w2][r] += dsc * dx * (float) idot - (half == 0 ? deff * sx : 0.0f);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const float up_v   = warp_reduce_sum<64>(acc[0][r]);
        const float gate_v = warp_reduce_sum<64>(acc[1][r]);
        if (lane == 0 && (row0 + r) < (int) ne1) {
            const float g = glu_op == (int) GGML_GLU_OP_SWIGLU
                ? ggml_cuda_op_silu_single(gate_v)
                : ggml_cuda_op_gelu_single(gate_v);
            y[row0 + r] = g * up_v;
        }
    }
#else
    GGML_UNUSED_VARS(wup, wgate, xq, y, ne0, ne1, glu_op, ids_src1, ids_dst,
                     expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// int8 MMQ tile GEMM straight from the repacked planes (prefill path).
// Y[tok, row] = Xq8[tok, :] . W[row, :] without dequantizing W.
//
// A workgroup (256 threads as a 16x16 grid) computes a BM x BN output
// tile (BM = 64 weight rows, BN = 64 tokens), walking the contraction
// in BK = 4 sub-block chunks staged through LDS. Thread (tx,ty) owns a
// strided 4x4 register micro-tile (rows ty, ty+16, ..., tokens tx,
// tx+16, ...) so a wavefront's 16 token reads land on 16 distinct LDS
// banks (block_q8_1 stride is 36 B = 9 words; gcd(9,32)=1).
//
// Tile shape carried from the production kernel in reinstinct, where a
// sweep (BK in {4,8}, TM/TN in {4,8}, occupancy 1/2) found 4x4 at
// occupancy 2 flat-optimal on gfx906.
#define MMQ_RP_BK 4
#define MMQ_RP_TM 4
#define MMQ_RP_TN 4
#define MMQ_RP_BM (16 * MMQ_RP_TM)
#define MMQ_RP_BN (16 * MMQ_RP_TN)

template <bool HAS_IDS, int TN_>
static __global__ void __launch_bounds__(256, 2) mmq_gemm_q4k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok, const uint32_t x_stride,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const uint32_t row0 = blockIdx.x * MMQ_RP_BM;
    uint32_t tok0 = blockIdx.y * (16 * TN_);
    uint32_t a_base = 0, a_end = 0;
    if constexpr (HAS_IDS) {
        if (blockIdx.y >= (uint32_t) tile_off[n_expert]) {
            return;
        }
        const uint32_t e = repack_find_expert(tile_off, n_expert, blockIdx.y);
        const uint32_t local_tile = blockIdx.y - (uint32_t) tile_off[e];
        a_base = (uint32_t) expert_bounds[e] + local_tile * (16 * TN_);
        a_end  = (uint32_t) expert_bounds[e + 1];
        wbase += e * expert_stride;
        tok0   = 0;
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, tile_off, n_expert, expert_stride);
    }

    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;
    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16);
    const uint32_t * ddp = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 2);

    __shared__ uint4      sW [MMQ_RP_BM][MMQ_RP_BK];     // packed nibbles
    __shared__ float2     sWs[MMQ_RP_BM][MMQ_RP_BK];     // (dsc, deff)
    __shared__ block_q8_1 sX [(16 * TN_)][MMQ_RP_BK + 1]; // int8 activations

    float acc[MMQ_RP_TM][TN_] = {};

    constexpr int LDW = MMQ_RP_BM * MMQ_RP_BK / 256; // tile elems per thread

    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_BK) {
#pragma unroll
        for (int i = 0; i < LDW; i++) {
            const int e  = t + i * 256;
            const int lr = e / MMQ_RP_BK, lk = e % MMQ_RP_BK;
            const uint32_t wrow = row0 + lr;
            const uint32_t sb   = sb0 + lk;
            if (wrow < ne1 && sb < n_sub) {
                sW[lr][lk] = nib[(size_t) wrow * nsp + sb];
                const uint16_t sm = smp[(size_t) wrow * nsp + sb];
                const uint32_t dd = ddp[(size_t) wrow * n_super + (sb >> 3)];
                const uint16_t d_bits    = (uint16_t)(dd & 0xFFFF);
                const uint16_t dmin_bits = (uint16_t)(dd >> 16);
                sWs[lr][lk] = make_float2(
                    __half2float(*reinterpret_cast<const __half *>(&d_bits))
                        * (float)(sm & 0xFFu),
                    __half2float(*reinterpret_cast<const __half *>(&dmin_bits))
                        * (float)(sm >> 8));
            } else {
                sWs[lr][lk] = make_float2(0.0f, 0.0f);
            }
        }
        for (int e = t; e < (16 * TN_) * MMQ_RP_BK; e += 256) {
            const int lr = e / MMQ_RP_BK, lk = e % MMQ_RP_BK;
            const uint32_t sb = sb0 + lk;
            uint32_t xcol = tok0 + lr;
            bool     xval = xcol < n_tok;
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + lr;
                xval = a < a_end;
                xcol = xval ? (uint32_t) ids_src1[a] : 0;
            }
            if (xval && sb < n_sub) {
                sX[lr][lk] = xq[(size_t) xcol * x_stride + sb];
            } else {
                sX[lr][lk].ds = make_half2(0.0f, 0.0f);
            }
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < MMQ_RP_BK; kk++) {
            uint4 wq[MMQ_RP_TM];
            float dsc[MMQ_RP_TM], deff[MMQ_RP_TM];
#pragma unroll
            for (int r = 0; r < MMQ_RP_TM; r++) {
                wq[r] = sW[ty + r * 16][kk];
                const float2 s = sWs[ty + r * 16][kk];
                dsc[r]  = s.x;
                deff[r] = s.y;
            }
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const block_q8_1 * xb = &sX[tx + n * 16][kk];
                const int * xq32 = reinterpret_cast<const int *>(xb->qs);
                const float dx = __low2float(xb->ds);
                const float sx = __high2float(xb->ds);
#pragma unroll
                for (int r = 0; r < MMQ_RP_TM; r++) {
                    const uint32_t qa[4] = { wq[r].x, wq[r].y, wq[r].z, wq[r].w };
                    int idot = 0;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        idot = ggml_cuda_dp4a((int)( qa[j]       & 0x0F0F0F0Fu), xq32[j],     idot);
                        idot = ggml_cuda_dp4a((int)((qa[j] >> 4) & 0x0F0F0F0Fu), xq32[j + 4], idot);
                    }
                    acc[r][n] += dsc[r] * dx * (float) idot - deff[r] * sx;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < MMQ_RP_TM; r++) {
        const uint32_t row = row0 + ty + r * 16;
        if (row >= ne1) {
            continue;
        }
#pragma unroll
        for (int n = 0; n < TN_; n++) {
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + tx + n * 16;
                if (a < a_end) {
                    y[(size_t) ids_dst[a] * dst_s1 + row] = acc[r][n];
                }
            } else {
                const uint32_t tok = tok0 + tx + n * 16;
                if (tok < n_tok) {
                    y[(size_t) tok * dst_s1 + row] = acc[r][n];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, x_stride, ids_src1, ids_dst, expert_bounds, tile_off, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q5_K MMQ — Q4_K's tiles plus the qh plane staged alongside.
template <bool HAS_IDS, int TN_>
static __global__ void __launch_bounds__(256, 2) mmq_gemm_q5k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok, const uint32_t x_stride,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const uint32_t row0 = blockIdx.x * MMQ_RP_BM;
    uint32_t tok0 = blockIdx.y * (16 * TN_);
    uint32_t a_base = 0, a_end = 0;
    if constexpr (HAS_IDS) {
        if (blockIdx.y >= (uint32_t) tile_off[n_expert]) {
            return;
        }
        const uint32_t e = repack_find_expert(tile_off, n_expert, blockIdx.y);
        const uint32_t local_tile = blockIdx.y - (uint32_t) tile_off[e];
        a_base = (uint32_t) expert_bounds[e] + local_tile * (16 * TN_);
        a_end  = (uint32_t) expert_bounds[e + 1];
        wbase += e * expert_stride;
        tok0   = 0;
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, tile_off, n_expert, expert_stride);
    }

    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;
    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint32_t * qhp = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 16);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 4);
    const uint32_t * ddp = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 4 + (size_t) ne1 * nsp * 2);

    __shared__ uint4      sW  [MMQ_RP_BM][MMQ_RP_BK];
    __shared__ uint32_t   sWqh[MMQ_RP_BM][MMQ_RP_BK];
    __shared__ float2     sWs [MMQ_RP_BM][MMQ_RP_BK];
    __shared__ block_q8_1 sX  [(16 * TN_)][MMQ_RP_BK + 1];

    float acc[MMQ_RP_TM][TN_] = {};

    const int lr = t >> 2;
    const int lk = t & 3;

    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_BK) {
        const uint32_t sb   = sb0 + lk;
        const uint32_t wrow = row0 + lr;
        if (wrow < ne1 && sb < n_sub) {
            sW  [lr][lk] = nib[(size_t) wrow * nsp + sb];
            sWqh[lr][lk] = qhp[(size_t) wrow * nsp + sb];
            const uint16_t sm = smp[(size_t) wrow * nsp + sb];
            const uint32_t dd = ddp[(size_t) wrow * n_super + (sb >> 3)];
            const uint16_t d_bits    = (uint16_t)(dd & 0xFFFF);
            const uint16_t dmin_bits = (uint16_t)(dd >> 16);
            sWs[lr][lk] = make_float2(
                __half2float(*reinterpret_cast<const __half *>(&d_bits))    * (float)(sm & 0xFFu),
                __half2float(*reinterpret_cast<const __half *>(&dmin_bits)) * (float)(sm >> 8));
        } else {
            sWs[lr][lk] = make_float2(0.0f, 0.0f);
        }
        if (lr < (16 * TN_)) {
            uint32_t xcol = tok0 + lr;
            bool     xval = xcol < n_tok;
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + lr;
                xval = a < a_end;
                xcol = xval ? (uint32_t) ids_src1[a] : 0;
            }
            if (xval && sb < n_sub) {
                sX[lr][lk] = xq[(size_t) xcol * x_stride + sb];
            } else {
                sX[lr][lk].ds = make_half2(0.0f, 0.0f);
            }
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < MMQ_RP_BK; kk++) {
            uint4 wq[MMQ_RP_TM]; uint32_t wqh[MMQ_RP_TM];
            float dsc[MMQ_RP_TM], deff[MMQ_RP_TM];
#pragma unroll
            for (int r = 0; r < MMQ_RP_TM; r++) {
                wq[r]  = sW  [ty + r * 16][kk];
                wqh[r] = sWqh[ty + r * 16][kk];
                const float2 s = sWs[ty + r * 16][kk];
                dsc[r]  = s.x;
                deff[r] = s.y;
            }
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const block_q8_1 * xb = &sX[tx + n * 16][kk];
                const int * xq32 = reinterpret_cast<const int *>(xb->qs);
                const float dx = __low2float(xb->ds);
                const float sx = __high2float(xb->ds);
#pragma unroll
                for (int r = 0; r < MMQ_RP_TM; r++) {
                    const uint32_t qa[4] = { wq[r].x, wq[r].y, wq[r].z, wq[r].w };
                    const uint32_t qh = wqh[r];
                    int idot = 0;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const uint32_t lo = ( qa[j]       & 0x0F0F0F0Fu)
                            | repack_spread4((qh >> (8 * j))     & 0xFu);
                        const uint32_t hi = ((qa[j] >> 4) & 0x0F0F0F0Fu)
                            | repack_spread4((qh >> (8 * j + 4)) & 0xFu);
                        idot = ggml_cuda_dp4a((int) lo, xq32[j],     idot);
                        idot = ggml_cuda_dp4a((int) hi, xq32[j + 4], idot);
                    }
                    acc[r][n] += dsc[r] * dx * (float) idot - deff[r] * sx;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < MMQ_RP_TM; r++) {
        const uint32_t row = row0 + ty + r * 16;
        if (row >= ne1) {
            continue;
        }
#pragma unroll
        for (int n = 0; n < TN_; n++) {
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + tx + n * 16;
                if (a < a_end) {
                    y[(size_t) ids_dst[a] * dst_s1 + row] = acc[r][n];
                }
            } else {
                const uint32_t tok = tok0 + tx + n * 16;
                if (tok < n_tok) {
                    y[(size_t) tok * dst_s1 + row] = acc[r][n];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, x_stride, ids_src1, ids_dst, expert_bounds, tile_off, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q6_K MMQ — h2 plane staged as uint2, signed scale pairs, per-token
// activation half-sums for the symmetric -32 fold.
template <bool HAS_IDS, int TN_>
static __global__ void __launch_bounds__(256, 2) mmq_gemm_q6k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok, const uint32_t x_stride,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const uint32_t row0 = blockIdx.x * MMQ_RP_BM;
    uint32_t tok0 = blockIdx.y * (16 * TN_);
    uint32_t a_base = 0, a_end = 0;
    if constexpr (HAS_IDS) {
        if (blockIdx.y >= (uint32_t) tile_off[n_expert]) {
            return;
        }
        const uint32_t e = repack_find_expert(tile_off, n_expert, blockIdx.y);
        const uint32_t local_tile = blockIdx.y - (uint32_t) tile_off[e];
        a_base = (uint32_t) expert_bounds[e] + local_tile * (16 * TN_);
        a_end  = (uint32_t) expert_bounds[e + 1];
        wbase += e * expert_stride;
        tok0   = 0;
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, tile_off, n_expert, expert_stride);
    }

    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;
    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint2    * h2p = reinterpret_cast<const uint2 *>(
        wbase + (size_t) ne1 * nsp * 16);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 8);
    const uint16_t * ddp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 8 + (size_t) ne1 * nsp * 2);

    __shared__ uint4      sW  [MMQ_RP_BM][MMQ_RP_BK];
    __shared__ uint2      sWh2[MMQ_RP_BM][MMQ_RP_BK];
    __shared__ float2     sWs [MMQ_RP_BM][MMQ_RP_BK];
    __shared__ block_q8_1 sX  [(16 * TN_)][MMQ_RP_BK + 1];

    float acc[MMQ_RP_TM][TN_] = {};

    const int lr = t >> 2;
    const int lk = t & 3;

    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_BK) {
        const uint32_t sb   = sb0 + lk;
        const uint32_t wrow = row0 + lr;
        if (wrow < ne1 && sb < n_sub) {
            sW  [lr][lk] = nib[(size_t) wrow * nsp + sb];
            sWh2[lr][lk] = h2p[(size_t) wrow * nsp + sb];
            const uint16_t sm     = smp[(size_t) wrow * nsp + sb];
            const uint16_t d_bits = ddp[(size_t) wrow * n_super + (sb >> 3)];
            const float d = __half2float(*reinterpret_cast<const __half *>(&d_bits));
            sWs[lr][lk] = make_float2(d * (float)(int)(int8_t)(sm & 0xFFu),
                                      d * (float)(int)(int8_t)(sm >> 8));
        } else {
            sWs[lr][lk] = make_float2(0.0f, 0.0f);
        }
        if (lr < (16 * TN_)) {
            uint32_t xcol = tok0 + lr;
            bool     xval = xcol < n_tok;
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + lr;
                xval = a < a_end;
                xcol = xval ? (uint32_t) ids_src1[a] : 0;
            }
            if (xval && sb < n_sub) {
                sX[lr][lk] = xq[(size_t) xcol * x_stride + sb];
            } else {
                sX[lr][lk].ds = make_half2(0.0f, 0.0f);
            }
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < MMQ_RP_BK; kk++) {
            uint4 wq[MMQ_RP_TM]; uint2 wh2[MMQ_RP_TM];
            float dlo[MMQ_RP_TM], dhi[MMQ_RP_TM];
#pragma unroll
            for (int r = 0; r < MMQ_RP_TM; r++) {
                wq[r]  = sW  [ty + r * 16][kk];
                wh2[r] = sWh2[ty + r * 16][kk];
                const float2 s = sWs[ty + r * 16][kk];
                dlo[r] = s.x;
                dhi[r] = s.y;
            }
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const block_q8_1 * xb = &sX[tx + n * 16][kk];
                const int * xq32 = reinterpret_cast<const int *>(xb->qs);
                const float dx = __low2float(xb->ds);
                int xis0 = 0, xis1 = 0;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    xis0 = ggml_cuda_dp4a(xq32[j],     0x01010101, xis0);
                    xis1 = ggml_cuda_dp4a(xq32[j + 4], 0x01010101, xis1);
                }
#pragma unroll
                for (int r = 0; r < MMQ_RP_TM; r++) {
                    const uint32_t qa[4] = { wq[r].x, wq[r].y, wq[r].z, wq[r].w };
                    const uint32_t h2lo = wh2[r].x, h2hi = wh2[r].y;
                    int idot0 = 0, idot1 = 0;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const uint32_t ge = 2 * j;
                        const uint32_t go = 2 * j + 1;
                        const uint32_t he = ((ge < 4 ? h2lo : h2hi) >> (8 * (ge & 3))) & 0xFFu;
                        const uint32_t ho = ((go < 4 ? h2lo : h2hi) >> (8 * (go & 3))) & 0xFFu;
                        const uint32_t q6lo = ( qa[j]       & 0x0F0F0F0Fu) | repack_spread2(he);
                        const uint32_t q6hi = ((qa[j] >> 4) & 0x0F0F0F0Fu) | repack_spread2(ho);
                        idot0 = ggml_cuda_dp4a((int) q6lo, xq32[j],     idot0);
                        idot1 = ggml_cuda_dp4a((int) q6hi, xq32[j + 4], idot1);
                    }
                    acc[r][n] += dlo[r] * dx * (float)(idot0 - 32 * xis0)
                               + dhi[r] * dx * (float)(idot1 - 32 * xis1);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < MMQ_RP_TM; r++) {
        const uint32_t row = row0 + ty + r * 16;
        if (row >= ne1) {
            continue;
        }
#pragma unroll
        for (int n = 0; n < TN_; n++) {
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + tx + n * 16;
                if (a < a_end) {
                    y[(size_t) ids_dst[a] * dst_s1 + row] = acc[r][n];
                }
            } else {
                const uint32_t tok = tok0 + tx + n * 16;
                if (tok < n_tok) {
                    y[(size_t) tok * dst_s1 + row] = acc[r][n];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, x_stride, ids_src1, ids_dst, expert_bounds, tile_off, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q3_K MMQ — Q6_K's tiles but the quant is 2-bit lo + 1-bit hi (no 4-bit
// nibble plane); reconstruct q3 = lo2 | (hbit << 2). Symmetric, bias 4.
template <bool HAS_IDS, int TN_>
static __global__ void __launch_bounds__(256, 2) mmq_gemm_q3k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok, const uint32_t x_stride,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const uint32_t row0 = blockIdx.x * MMQ_RP_BM;
    uint32_t tok0 = blockIdx.y * (16 * TN_);
    uint32_t a_base = 0, a_end = 0;
    if constexpr (HAS_IDS) {
        if (blockIdx.y >= (uint32_t) tile_off[n_expert]) {
            return;
        }
        const uint32_t e = repack_find_expert(tile_off, n_expert, blockIdx.y);
        const uint32_t local_tile = blockIdx.y - (uint32_t) tile_off[e];
        a_base = (uint32_t) expert_bounds[e] + local_tile * (16 * TN_);
        a_end  = (uint32_t) expert_bounds[e + 1];
        wbase += e * expert_stride;
        tok0   = 0;
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, tile_off, n_expert, expert_stride);
    }

    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;
    const uint2    * lo2p = reinterpret_cast<const uint2 *>(wbase);
    const uint32_t * hi1p = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 8);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 8 + (size_t) ne1 * nsp * 4);
    const uint16_t * ddp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 8 + (size_t) ne1 * nsp * 4 + (size_t) ne1 * nsp * 2);

    __shared__ uint2      sWl[MMQ_RP_BM][MMQ_RP_BK];
    __shared__ uint32_t   sWh[MMQ_RP_BM][MMQ_RP_BK];
    __shared__ float2     sWs[MMQ_RP_BM][MMQ_RP_BK];
    __shared__ block_q8_1 sX [(16 * TN_)][MMQ_RP_BK + 1];

    float acc[MMQ_RP_TM][TN_] = {};

    const int lr = t >> 2;
    const int lk = t & 3;

    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_BK) {
        const uint32_t sb   = sb0 + lk;
        const uint32_t wrow = row0 + lr;
        if (wrow < ne1 && sb < n_sub) {
            sWl[lr][lk] = lo2p[(size_t) wrow * nsp + sb];
            sWh[lr][lk] = hi1p[(size_t) wrow * nsp + sb];
            const uint16_t sm     = smp[(size_t) wrow * nsp + sb];
            const uint16_t d_bits = ddp[(size_t) wrow * n_super + (sb >> 3)];
            const float d = __half2float(*reinterpret_cast<const __half *>(&d_bits));
            sWs[lr][lk] = make_float2(d * (float)(int)(int8_t)(sm & 0xFFu),
                                      d * (float)(int)(int8_t)(sm >> 8));
        } else {
            sWs[lr][lk] = make_float2(0.0f, 0.0f);
        }
        if (lr < (16 * TN_)) {
            uint32_t xcol = tok0 + lr;
            bool     xval = xcol < n_tok;
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + lr;
                xval = a < a_end;
                xcol = xval ? (uint32_t) ids_src1[a] : 0;
            }
            if (xval && sb < n_sub) {
                sX[lr][lk] = xq[(size_t) xcol * x_stride + sb];
            } else {
                sX[lr][lk].ds = make_half2(0.0f, 0.0f);
            }
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < MMQ_RP_BK; kk++) {
            uint2 wl[MMQ_RP_TM]; uint32_t wh[MMQ_RP_TM];
            float dlo[MMQ_RP_TM], dhi[MMQ_RP_TM];
#pragma unroll
            for (int r = 0; r < MMQ_RP_TM; r++) {
                wl[r] = sWl[ty + r * 16][kk];
                wh[r] = sWh[ty + r * 16][kk];
                const float2 s = sWs[ty + r * 16][kk];
                dlo[r] = s.x;
                dhi[r] = s.y;
            }
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const block_q8_1 * xb = &sX[tx + n * 16][kk];
                const int * xq32 = reinterpret_cast<const int *>(xb->qs);
                const float dx = __low2float(xb->ds);
                int xis0 = 0, xis1 = 0;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    xis0 = ggml_cuda_dp4a(xq32[j],     0x01010101, xis0);
                    xis1 = ggml_cuda_dp4a(xq32[j + 4], 0x01010101, xis1);
                }
#pragma unroll
                for (int r = 0; r < MMQ_RP_TM; r++) {
                    const uint32_t lo2lo = wl[r].x, lo2hi = wl[r].y, qh = wh[r];
                    int idot0 = 0, idot1 = 0;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const uint32_t ge = 2 * j;
                        const uint32_t go = 2 * j + 1;
                        const uint32_t lb = ((ge < 4 ? lo2lo : lo2hi) >> (8 * (ge & 3))) & 0xFFu;
                        const uint32_t hb = ((go < 4 ? lo2lo : lo2hi) >> (8 * (go & 3))) & 0xFFu;
                        const uint32_t q3lo = repack_spread2_lo(lb) | repack_spread1_hi((qh >> (8 * j))     & 0xFu);
                        const uint32_t q3hi = repack_spread2_lo(hb) | repack_spread1_hi((qh >> (8 * j + 4)) & 0xFu);
                        idot0 = ggml_cuda_dp4a((int) q3lo, xq32[j],     idot0);
                        idot1 = ggml_cuda_dp4a((int) q3hi, xq32[j + 4], idot1);
                    }
                    acc[r][n] += dlo[r] * dx * (float)(idot0 - 4 * xis0)
                               + dhi[r] * dx * (float)(idot1 - 4 * xis1);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < MMQ_RP_TM; r++) {
        const uint32_t row = row0 + ty + r * 16;
        if (row >= ne1) {
            continue;
        }
#pragma unroll
        for (int n = 0; n < TN_; n++) {
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + tx + n * 16;
                if (a < a_end) {
                    y[(size_t) ids_dst[a] * dst_s1 + row] = acc[r][n];
                }
            } else {
                const uint32_t tok = tok0 + tx + n * 16;
                if (tok < n_tok) {
                    y[(size_t) tok * dst_s1 + row] = acc[r][n];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, x_stride, ids_src1, ids_dst, expert_bounds, tile_off, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q8_0 MMQ — 32 qs bytes per sub-block staged as two uint4s; no offset
// term, so the accumulate is just dsc * dx * idot.
template <bool HAS_IDS, int TN_>
static __global__ void __launch_bounds__(256, 2) mmq_gemm_q8_0_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok, const uint32_t x_stride,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const uint32_t row0 = blockIdx.x * MMQ_RP_BM;
    uint32_t tok0 = blockIdx.y * (16 * TN_);
    uint32_t a_base = 0, a_end = 0;
    if constexpr (HAS_IDS) {
        if (blockIdx.y >= (uint32_t) tile_off[n_expert]) {
            return;
        }
        const uint32_t e = repack_find_expert(tile_off, n_expert, blockIdx.y);
        const uint32_t local_tile = blockIdx.y - (uint32_t) tile_off[e];
        a_base = (uint32_t) expert_bounds[e] + local_tile * (16 * TN_);
        a_end  = (uint32_t) expert_bounds[e + 1];
        wbase += e * expert_stride;
        tok0   = 0;
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, tile_off, n_expert, expert_stride);
    }

    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint4    * qsp = reinterpret_cast<const uint4 *>(wbase);
    const uint16_t * dp  = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 32);

    __shared__ uint4      sW_lo[MMQ_RP_BM][MMQ_RP_BK];
    __shared__ uint4      sW_hi[MMQ_RP_BM][MMQ_RP_BK];
    __shared__ float      sWd  [MMQ_RP_BM][MMQ_RP_BK];
    __shared__ block_q8_1 sX   [(16 * TN_)][MMQ_RP_BK + 1];

    float acc[MMQ_RP_TM][TN_] = {};

    const int lr = t >> 2;
    const int lk = t & 3;

    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_BK) {
        const uint32_t sb   = sb0 + lk;
        const uint32_t wrow = row0 + lr;
        if (wrow < ne1 && sb < n_sub) {
            sW_lo[lr][lk] = qsp[(size_t)(wrow * nsp + sb) * 2];
            sW_hi[lr][lk] = qsp[(size_t)(wrow * nsp + sb) * 2 + 1];
            const uint16_t d_bits = dp[(size_t) wrow * nsp + sb];
            sWd[lr][lk] = __half2float(*reinterpret_cast<const __half *>(&d_bits));
        } else {
            sWd[lr][lk] = 0.0f;
        }
        if (lr < (16 * TN_)) {
            uint32_t xcol = tok0 + lr;
            bool     xval = xcol < n_tok;
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + lr;
                xval = a < a_end;
                xcol = xval ? (uint32_t) ids_src1[a] : 0;
            }
            if (xval && sb < n_sub) {
                sX[lr][lk] = xq[(size_t) xcol * x_stride + sb];
            } else {
                sX[lr][lk].ds = make_half2(0.0f, 0.0f);
            }
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < MMQ_RP_BK; kk++) {
            uint4 wq_lo[MMQ_RP_TM], wq_hi[MMQ_RP_TM];
            float dsc[MMQ_RP_TM];
#pragma unroll
            for (int r = 0; r < MMQ_RP_TM; r++) {
                wq_lo[r] = sW_lo[ty + r * 16][kk];
                wq_hi[r] = sW_hi[ty + r * 16][kk];
                dsc[r]   = sWd  [ty + r * 16][kk];
            }
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const block_q8_1 * xb = &sX[tx + n * 16][kk];
                const int * xq32 = reinterpret_cast<const int *>(xb->qs);
                const float dx = __low2float(xb->ds);
#pragma unroll
                for (int r = 0; r < MMQ_RP_TM; r++) {
                    const uint32_t lo[4] = { wq_lo[r].x, wq_lo[r].y, wq_lo[r].z, wq_lo[r].w };
                    const uint32_t hi[4] = { wq_hi[r].x, wq_hi[r].y, wq_hi[r].z, wq_hi[r].w };
                    int idot = 0;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        idot = ggml_cuda_dp4a((int) lo[j], xq32[j],     idot);
                        idot = ggml_cuda_dp4a((int) hi[j], xq32[j + 4], idot);
                    }
                    acc[r][n] += dsc[r] * dx * (float) idot;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < MMQ_RP_TM; r++) {
        const uint32_t row = row0 + ty + r * 16;
        if (row >= ne1) {
            continue;
        }
#pragma unroll
        for (int n = 0; n < TN_; n++) {
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + tx + n * 16;
                if (a < a_end) {
                    y[(size_t) ids_dst[a] * dst_s1 + row] = acc[r][n];
                }
            } else {
                const uint32_t tok = tok0 + tx + n * 16;
                if (tok < n_tok) {
                    y[(size_t) tok * dst_s1 + row] = acc[r][n];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, x_stride, ids_src1, ids_dst, expert_bounds, tile_off, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// ---------------------------------------------------------------------
// MUL_MAT dispatch
// ---------------------------------------------------------------------

static void ggml_cuda_mul_mat_repacked_slice(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const uint8_t * w, const block_q8_1 * xq,
        float * dst_d, int64_t ne00, int64_t ne01, int64_t ne11,
        int64_t x_stride, cudaStream_t stream);

void ggml_cuda_mul_mat_repacked(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->nb[0] == sizeof(float)); // rows may be strided; dim0 must be dense
    GGML_ASSERT(dst->nb[1]  == (size_t) dst->ne[0] * sizeof(float));

    const int64_t ne00 = src0->ne[0]; // K
    const int64_t ne01 = src0->ne[1]; // M
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1]; // N
    const int64_t ne12 = src1->ne[2]; // broadcast slices (2D weight repeats)
    const int64_t ne13 = src1->ne[3];
    GGML_ASSERT(ne10 == ne00);

    cudaStream_t stream = ctx.stream();
    const uint8_t * w = (const uint8_t *) src0->data;

    // Quantize the whole (possibly 3D/4D) activation once; blocks land
    // contiguously as [ne13][ne12][ne11][ne10_padded/QK8_1].
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1; // q8_1 blocks per column
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(),
        ne13 * ne12 * ne11 * ne10_padded * sizeof(block_q8_1) / QK8_1);
    {
        const int64_t s11 = src1->nb[1] / sizeof(float);
        const int64_t s12 = src1->nb[2] / sizeof(float);
        const int64_t s13 = src1->nb[3] / sizeof(float);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(),
            src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }

    for (int64_t i3 = 0; i3 < ne13; i3++) {
    for (int64_t i2 = 0; i2 < ne12; i2++) {
        const block_q8_1 * xq = (const block_q8_1 *) src1_q8_1.get()
                              + (i3 * ne12 + i2) * ne11 * x_stride;
        float * dst_d = (float *)((char *) dst->data + i3 * dst->nb[3] + i2 * dst->nb[2]);
        ggml_cuda_mul_mat_repacked_slice(ctx, src0, w, xq, dst_d,
            ne00, ne01, ne11, x_stride, stream);
    }
    }
}

static void ggml_cuda_mul_mat_repacked_slice(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const uint8_t * w, const block_q8_1 * xq,
        float * dst_d, const int64_t ne00, const int64_t ne01, const int64_t ne11,
        const int64_t x_stride, cudaStream_t stream) {
    if (ne11 == 1) {
        // decode: dp4a matvec straight from the planes
        switch (src0->type) {
            case GGML_TYPE_Q3_K: {
                const dim3 grid((ne01 + 7) / 8, 1, 1);
                mul_mat_vec_q3k_repacked<false><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, nullptr, nullptr, 0, 0, 0, 0);
            } break;
            case GGML_TYPE_Q4_K: {
                const dim3 grid((ne01 + 7) / 8, 1, 1);
                mul_mat_vec_q4k_repacked<false><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, nullptr, nullptr, 0, 0, 0, 0);
            } break;
            case GGML_TYPE_Q5_K: {
                const dim3 grid((ne01 + 7) / 8, 1, 1);
                mul_mat_vec_q5k_repacked<false><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, nullptr, nullptr, 0, 0, 0, 0);
            } break;
            case GGML_TYPE_Q6_K: {
                const dim3 grid((ne01 + 7) / 8, 1, 1);
                mul_mat_vec_q6k_repacked<false><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, nullptr, nullptr, 0, 0, 0, 0);
            } break;
            case GGML_TYPE_Q8_0: {
                // large ne01: single-wave ROWS=1 blocks maximize the
                // wavefront count (measured on gfx906: ROWS=2 at
                // ne01=4096 stalls ~184 GB/s, ROWS=1 ~2x it). Small
                // ne01: 4-wave ROWS=2 blocks (the K-quant matvec shape).
                if (ne01 >= 4096) {
                    const dim3 grid(ne01, 1, 1);
                    mul_mat_vec_q8_0_repacked<1, 1, false><<<grid, 64, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                        nullptr, nullptr, nullptr, 0, 0, 0, 0);
                } else {
                    // 4-wave ROWS=2 with half-sub-block work units is the
                    // best of the swept variants (gfx906, 0.8B-Q8_0 tg128:
                    // 231.0 vs 222.7 single-wave, 222.2 full-block units,
                    // 219.9 ROWS=4, 214.2 quarter units; canonical mmvq
                    // is 238.0 — the residual ~3% is why Q8_0 stays
                    // behind its own env gate)
                    const dim3 grid((ne01 + 7) / 8, 1, 1);
                    mul_mat_vec_q8_0_repacked<2, 4, false><<<grid, 256, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                        nullptr, nullptr, nullptr, 0, 0, 0, 0);
                }
            } break;
            default: GGML_ABORT("unsupported repack type");
        }
        return;
    }

    // prefill: int8 MMQ tile GEMM straight from the repacked planes
    const dim3 grid((ne01 + MMQ_RP_BM - 1) / MMQ_RP_BM,
                    (ne11 + MMQ_RP_BN - 1) / MMQ_RP_BN, 1);
    switch (src0->type) {
        case GGML_TYPE_Q3_K:
            mmq_gemm_q3k_repacked<false, MMQ_RP_TN><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, (uint32_t) x_stride,
                nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
            break;
        case GGML_TYPE_Q4_K:
            mmq_gemm_q4k_repacked<false, MMQ_RP_TN><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, (uint32_t) x_stride,
                nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
            break;
        case GGML_TYPE_Q5_K:
            mmq_gemm_q5k_repacked<false, MMQ_RP_TN><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, (uint32_t) x_stride,
                nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
            break;
        case GGML_TYPE_Q6_K:
            mmq_gemm_q6k_repacked<false, MMQ_RP_TN><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, (uint32_t) x_stride,
                nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
            break;
        case GGML_TYPE_Q8_0:
            mmq_gemm_q8_0_repacked<false, MMQ_RP_TN><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, (uint32_t) x_stride,
                nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
            break;
        default: GGML_ABORT("unsupported repack type");
    }
    GGML_UNUSED(ctx);
}

// MUL_MAT_ID with src0 in the repack buffer type. The mm_ids_helper
// compacts routing into expert-sorted assignment order; activations are
// quantized once in natural column order and gathered per assignment
// via ids_src1 inside the kernels; outputs scatter via ids_dst.
void ggml_cuda_mul_mat_id_repacked(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ids->type  == GGML_TYPE_I32);
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(ids->nb[0]  == sizeof(int32_t));
    GGML_ASSERT(src1->ne[3] == 1 && dst->ne[3] == 1);
    // column-contiguity: ids_src1/ids_dst are flat column indices
    GGML_ASSERT(src1->nb[2] == src1->nb[1] * src1->ne[1]);
    GGML_ASSERT(dst->nb[2]  == dst->nb[1]  * dst->ne[1]);
    GGML_ASSERT(dst->nb[1]  == (size_t) dst->ne[0] * sizeof(float));

    const int64_t ne00 = src0->ne[0]; // K
    const int64_t ne01 = src0->ne[1]; // rows per expert
    const int64_t ne02 = src0->ne[2]; // experts
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 == ne00);
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = ids->ne[1];
    const int64_t n_assign      = n_expert_used * n_tokens;

    cudaStream_t stream = ctx.stream();
    const uint8_t * w = (const uint8_t *) src0->data;
    float * dst_d = (float *) dst->data;
    const size_t expert_stride = repack_gcn_nbytes(src0->type, ne00, ne01);
    const uint32_t dst_s1 = dst->nb[1] / sizeof(float);

    // routing compaction
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_assign);
    ggml_cuda_pool_alloc<int32_t> ids_dst (ctx.pool(), n_assign);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);
    if (n_tokens > 1) {
        const int si1  = ids->nb[1] / sizeof(int32_t);
        const int sis1 = src1->nb[2] / src1->nb[1];
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data,
            ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, n_tokens, n_expert_used, src1->ne[1], si1, sis1, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    // quantize all activation columns once, natural order
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(),
        src1->ne[2] * src1->ne[1] * ne10_padded * sizeof(block_q8_1) / QK8_1);
    {
        const int64_t s11 = src1->nb[1] / sizeof(float);
        const int64_t s12 = src1->nb[2] / sizeof(float);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(),
            src0->type, ne10, s11, s12, s12 * src1->ne[2], ne10_padded,
            src1->ne[1], src1->ne[2], 1, stream);
    }
    const block_q8_1 * xq = (const block_q8_1 *) src1_q8_1.get();

    if (n_tokens == 1) {
        // decode: one matvec per slot; experts read directly from the
        // raw ids tensor in-kernel (no compaction kernels — launch
        // parity with canonical mmvq-id). Broadcast src1 (ne[1]==1, one
        // shared activation column for all slots) uses x-stride 0.
        const uint32_t xs_eff = src1->ne[1] == 1 ? 0u : (uint32_t) x_stride;
        switch (src0->type) {
            case GGML_TYPE_Q3_K: {
                const dim3 grid((ne01 + 7) / 8, n_assign, 1);
                mul_mat_vec_q3k_repacked<true><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    (const int32_t *) ids->data, nullptr, nullptr,
                    (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
            } break;
            case GGML_TYPE_Q4_K: {
                const dim3 grid((ne01 + 7) / 8, n_assign, 1);
                mul_mat_vec_q4k_repacked<true><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    (const int32_t *) ids->data, nullptr, nullptr,
                    (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
            } break;
            case GGML_TYPE_Q5_K: {
                const dim3 grid((ne01 + 7) / 8, n_assign, 1);
                mul_mat_vec_q5k_repacked<true><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    (const int32_t *) ids->data, nullptr, nullptr,
                    (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
            } break;
            case GGML_TYPE_Q6_K: {
                const dim3 grid((ne01 + 7) / 8, n_assign, 1);
                mul_mat_vec_q6k_repacked<true><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    (const int32_t *) ids->data, nullptr, nullptr,
                    (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
            } break;
            case GGML_TYPE_Q8_0: {
                if (ne01 >= 4096) {
                    const dim3 grid(ne01, n_assign, 1);
                    mul_mat_vec_q8_0_repacked<1, 1, true><<<grid, 64, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                        (const int32_t *) ids->data, nullptr, nullptr,
                        (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
                } else {
                    const dim3 grid((ne01 + 7) / 8, n_assign, 1);
                    mul_mat_vec_q8_0_repacked<2, 4, true><<<grid, 256, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                        (const int32_t *) ids->data, nullptr, nullptr,
                        (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
                }
            } break;
            default: GGML_ABORT("unsupported repack type");
        }
        return;
    }

    // batch: grouped tile GEMM, thin 16-token tiles (MoE routing spreads
    // tokens across experts; a 64-wide tile would be mostly empty)
    constexpr int TN_ID = 1;
    ggml_cuda_pool_alloc<int32_t> tile_off(ctx.pool(), ne02 + 1);
    repack_tile_off<16 * TN_ID><<<1, 1, 0, stream>>>(expert_bounds.get(), tile_off.get(), ne02);
    // over-launch upper bound: every expert can add one partial tile
    const int64_t max_tiles = n_assign / (16 * TN_ID) + ne02;
    const dim3 grid((ne01 + MMQ_RP_BM - 1) / MMQ_RP_BM, max_tiles, 1);

    switch (src0->type) {
        case GGML_TYPE_Q3_K:
            mmq_gemm_q3k_repacked<true, TN_ID><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, 0, (uint32_t) x_stride,
                ids_src1.get(), ids_dst.get(), expert_bounds.get(), tile_off.get(),
                (uint32_t) ne02, expert_stride, dst_s1);
            break;
        case GGML_TYPE_Q4_K:
            mmq_gemm_q4k_repacked<true, TN_ID><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, 0, (uint32_t) x_stride,
                ids_src1.get(), ids_dst.get(), expert_bounds.get(), tile_off.get(),
                (uint32_t) ne02, expert_stride, dst_s1);
            break;
        case GGML_TYPE_Q5_K:
            mmq_gemm_q5k_repacked<true, TN_ID><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, 0, (uint32_t) x_stride,
                ids_src1.get(), ids_dst.get(), expert_bounds.get(), tile_off.get(),
                (uint32_t) ne02, expert_stride, dst_s1);
            break;
        case GGML_TYPE_Q6_K:
            mmq_gemm_q6k_repacked<true, TN_ID><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, 0, (uint32_t) x_stride,
                ids_src1.get(), ids_dst.get(), expert_bounds.get(), tile_off.get(),
                (uint32_t) ne02, expert_stride, dst_s1);
            break;
        case GGML_TYPE_Q8_0:
            mmq_gemm_q8_0_repacked<true, TN_ID><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, 0, (uint32_t) x_stride,
                ids_src1.get(), ids_dst.get(), expert_bounds.get(), tile_off.get(),
                (uint32_t) ne02, expert_stride, dst_s1);
            break;
        default: GGML_ABORT("unsupported repack type");
    }
}

// Eligibility for the fused gate+up GLU path: both weights in the
// repack buffer type, Q4_K, identical shape; decode only (one output
// column per expert slot); SWIGLU or GEGLU.
bool ggml_cuda_repack_should_fuse_glu(const ggml_tensor * up, const ggml_tensor * gate,
        const ggml_tensor * glu) {
    const ggml_tensor * wu = up->src[0];
    const ggml_tensor * wg = gate->src[0];
    if (wu->buffer == nullptr || wg->buffer == nullptr ||
        !ggml_backend_buft_is_cuda_repack(wu->buffer->buft) ||
        !ggml_backend_buft_is_cuda_repack(wg->buffer->buft)) {
        return false;
    }
    if (wu->type != GGML_TYPE_Q4_K || wg->type != GGML_TYPE_Q4_K ||
        !ggml_are_same_shape(wu, wg)) {
        return false;
    }
    const ggml_glu_op op = ggml_get_glu_op(glu);
    if (op != GGML_GLU_OP_SWIGLU && op != GGML_GLU_OP_GEGLU) {
        return false;
    }
    if (up->src[2] != nullptr) { // MUL_MAT_ID: one token
        return up->src[1]->ne[2] == 1 && glu->ne[2] == 1;
    }
    return up->src[1]->ne[1] == 1; // dense: one column
}

void ggml_cuda_mul_mat_repacked_fused_glu(ggml_backend_cuda_context & ctx,
        const ggml_tensor * up_w, const ggml_tensor * gate_w,
        const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const int glu_op) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->nb[0] == sizeof(float));

    const int64_t ne00 = up_w->ne[0];
    const int64_t ne01 = up_w->ne[1];
    cudaStream_t stream = ctx.stream();
    const uint8_t * wu = (const uint8_t *) up_w->data;
    const uint8_t * wg = (const uint8_t *) gate_w->data;
    float * dst_d = (float *) dst->data;

    const int64_t ne10_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1;

    if (ids == nullptr) {
        // dense decode column
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(),
            ne10_padded * sizeof(block_q8_1) / QK8_1);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(),
            up_w->type, ne00, ne00, ne00, ne00, ne10_padded, 1, 1, 1, stream);
        const dim3 grid((ne01 + 7) / 8, 1, 1);
        mul_mat_vec_q4k_repacked_glu<false><<<grid, 256, 0, stream>>>(
            wu, wg, (const block_q8_1 *) src1_q8_1.get(), dst_d,
            (uint32_t) ne00, (uint32_t) ne01, glu_op,
            nullptr, nullptr, nullptr, 0, 0, 0, 0);
        return;
    }

    // MoE decode: same routing machinery as the unfused ID path
    const int64_t ne02 = up_w->ne[2];
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_assign = n_expert_used; // one token
    const size_t expert_stride = repack_gcn_nbytes(up_w->type, ne00, ne01);
    GGML_ASSERT(dst->nb[1] == (size_t) dst->ne[0] * sizeof(float));
    const uint32_t dst_s1 = dst->nb[1] / sizeof(float);

    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(),
        src1->ne[1] * ne10_padded * sizeof(block_q8_1) / QK8_1);
    {
        const int64_t s11 = src1->nb[1] / sizeof(float);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(),
            up_w->type, ne00, s11, s11 * src1->ne[1], s11 * src1->ne[1],
            ne10_padded, src1->ne[1], 1, 1, stream);
    }
    const uint32_t xs_eff = src1->ne[1] == 1 ? 0u : (uint32_t) x_stride;
    const dim3 grid((ne01 + 7) / 8, n_assign, 1);
    mul_mat_vec_q4k_repacked_glu<true><<<grid, 256, 0, stream>>>(
        wu, wg, (const block_q8_1 *) src1_q8_1.get(), dst_d,
        (uint32_t) ne00, (uint32_t) ne01, glu_op,
        (const int32_t *) ids->data, nullptr, nullptr,
        (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
}

// ---------------------------------------------------------------------
// buffer type
// ---------------------------------------------------------------------

struct ggml_backend_cuda_repack_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_repack_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_repack_buffer_type_context * ctx =
        (ggml_backend_cuda_repack_buffer_type_context *) buft->context;
    return ctx->name.c_str();
}

bool ggml_backend_buft_is_cuda_repack(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_repack_buffer_type_get_name;
}

static void ggml_backend_cuda_repack_buffer_set_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor,
        const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_ASSERT(ggml_cuda_repack_tensor_supported(tensor));

    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t ne2 = tensor->ne[2]; // experts (1 for plain 2D weights)

    const size_t src_stride = ggml_nbytes(tensor) / ne2;
    const size_t dst_stride = repack_gcn_nbytes(tensor->type, ne0, ne1);
    std::vector<uint8_t> staged(dst_stride * ne2);
    for (int64_t e = 0; e < ne2; e++) {
        const uint8_t * src_e = (const uint8_t *) data + e * src_stride;
        uint8_t       * dst_e = staged.data() + e * dst_stride;
        switch (tensor->type) {
            case GGML_TYPE_Q3_K: repack_q3k_host ((const block_q3_K *) src_e, dst_e, ne0, ne1); break;
            case GGML_TYPE_Q4_K: repack_q4k_host ((const block_q4_K *) src_e, dst_e, ne0, ne1); break;
            case GGML_TYPE_Q5_K: repack_q5k_host ((const block_q5_K *) src_e, dst_e, ne0, ne1); break;
            case GGML_TYPE_Q6_K: repack_q6k_host ((const block_q6_K *) src_e, dst_e, ne0, ne1); break;
            case GGML_TYPE_Q8_0: repack_q8_0_host((const block_q8_0 *) src_e, dst_e, ne0, ne1); break;
            default:             GGML_ABORT("unsupported repack type");
        }
    }

    ggml_backend_cuda_repack_buffer_type_context * ctx =
        (ggml_backend_cuda_repack_buffer_type_context *) buffer->buft->context;
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(tensor->data, staged.data(), staged.size(),
        cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_repack_buffer_get_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
        void * data, size_t offset, size_t size) {
    GGML_ABORT("repacked tensors cannot be read back (GGML_CUDA_REPACK)");
    GGML_UNUSED_VARS(buffer, tensor, data, offset, size);
}

static ggml_backend_buffer_t ggml_backend_cuda_repack_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_repack_buffer_type_context * ctx =
        (ggml_backend_cuda_repack_buffer_type_context *) buft->context;

    ggml_backend_buffer_t buffer =
        ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(ctx->device), size);
    if (buffer == nullptr) {
        return nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.set_tensor  = ggml_backend_cuda_repack_buffer_set_tensor;
    buffer->iface.get_tensor  = ggml_backend_cuda_repack_buffer_get_tensor;
    buffer->iface.cpy_tensor  = nullptr;
    return buffer;
}

static size_t ggml_backend_cuda_repack_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_repack_buffer_type_get_alloc_size(
        ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    if (ggml_cuda_repack_tensor_supported(tensor)) {
        return repack_gcn_nbytes(tensor->type, tensor->ne[0], tensor->ne[1]) * tensor->ne[2];
    }
    return ggml_nbytes(tensor);
    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_repack_buffer_type_interface = {
    /* .get_name       = */ ggml_backend_cuda_repack_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_cuda_repack_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_cuda_repack_buffer_type_get_alignment,
    /* .get_max_size   = */ nullptr,
    /* .get_alloc_size = */ ggml_backend_cuda_repack_buffer_type_get_alloc_size,
    /* .is_host        = */ nullptr,
};

ggml_backend_buffer_type_t ggml_backend_cuda_repack_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    // Default-on for GCN; GGML_CUDA_REPACK=0 opts out. (Repacked
    // weights cannot be read back: llama-quantize/save from a loaded
    // model needs the opt-out.)
    const char * env = getenv("GGML_CUDA_REPACK");
    if (env != nullptr && env[0] == '0') {
        return nullptr;
    }
    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }
    if (!GGML_CUDA_CC_IS_GCN(ggml_cuda_info().devices[device].cc)) {
        return nullptr;
    }

    static ggml_backend_buffer_type buft_storage[GGML_CUDA_MAX_DEVICES];
    static bool initialized[GGML_CUDA_MAX_DEVICES] = {};

    if (!initialized[device]) {
        buft_storage[device] = {
            /* .iface   = */ ggml_backend_cuda_repack_buffer_type_interface,
            /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
            /* .context = */ new ggml_backend_cuda_repack_buffer_type_context{
                                 device, GGML_CUDA_NAME + std::to_string(device) + "_Repacked"},
        };
        initialized[device] = true;
    }
    return &buft_storage[device];
}
