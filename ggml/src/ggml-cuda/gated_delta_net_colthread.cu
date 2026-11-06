#include "gated_delta_net_colthread.cuh"
#include "ggml-cuda/common.cuh"

// Column-per-thread gated delta net kernels, ported from castagna-veloce.
// lane/row mapping and the reduction reproduce the chunked kernel's summation order,
// so the output is bit-identical to gated_delta_net_chunked_cuda.

#define GDN_CT_TT   16
#define GDN_CT_COLS 32

#if defined(GGML_USE_HIP) && !defined(GGML_GDN_NO_DPP)
template <int CTRL, int BANKS>
static __device__ __forceinline__ int gdn_dpp(const int old, const int v) {
    return __builtin_amdgcn_update_dpp(old, v, CTRL, 0xF, BANKS, false);
}

template <int OFF>
static __device__ __forceinline__ float gdn_xor(const float x) {
    const int v = __float_as_int(x);
    if constexpr (OFF == 8) {
        return __int_as_float(gdn_dpp<0x118, 0xC>(gdn_dpp<0x108, 0x3>(v, v), v)); // row_shl:8 banks 0-1, row_shr:8 banks 2-3
    } else if constexpr (OFF == 4) {
        return __int_as_float(gdn_dpp<0x114, 0xA>(gdn_dpp<0x104, 0x5>(v, v), v)); // row_shl:4 banks 0,2, row_shr:4 banks 1,3
    } else if constexpr (OFF == 2) {
        return __int_as_float(gdn_dpp<0x4E, 0xF>(v, v)); // quad_perm [2,3,0,1]
    } else {
        return __int_as_float(gdn_dpp<0xB1, 0xF>(v, v)); // quad_perm [1,0,3,2]
    }
}
#endif

// Sum over aligned groups of W <= 16 lanes, widening pairing first (xor 1, then 2, 4, ...): the
// same levels of the chunked kernel's warp reduction, applied to the lane-group partials.
template <int W>
static __device__ __forceinline__ float gdn_group_sum(float x) {
#if defined(GGML_USE_HIP) && !defined(GGML_GDN_NO_DPP)
    static_assert(W == 1 || W == 2 || W == 4 || W == 8 || W == 16, "group width");
    if constexpr (W >= 2)  { x += gdn_xor<1>(x); }
    if constexpr (W >= 4)  { x += gdn_xor<2>(x); }
    if constexpr (W >= 8)  { x += gdn_xor<4>(x); }
    if constexpr (W >= 16) { x += gdn_xor<8>(x); }
    return x;
#else
#pragma unroll
    for (int off = 1; off < W; off *= 2) {
        x += __shfl_xor_sync(0xffffffff, x, off, W);
    }
    return x;
#endif
}

// Canonical widening pairwise tree over N partial sums (in place): the chunked kernel's warp
// reduction levels that stay inside one lane group, applied to that lane's leaf partials.
template <int N>
static __device__ __forceinline__ float gdn_pairwise_sum(float (&p)[N]) {
#pragma unroll
    for (int len = N/2; len >= 1; len /= 2) {
#pragma unroll
        for (int i = 0; i < len; ++i) {
            p[i] = p[2*i] + p[2*i + 1];
        }
    }
    return p[0];
}

// Prefill variant: each state column is owned by LPC lanes. Lane `sub` reproduces the B = S_v/(2*LPC)
// chunked-kernel leaf pairs of the warp reduction: rows sub*B + b and sub*B + b + S_v/2, mul + fma in
// the same order, then the in-lane widening tree, then the lane-group widening tree - bit-identical
// arithmetic per column. k, q, g and beta are shared by all columns of a head, so the block stages
// them in shared memory GDN_CT_TT tokens at a time; a block covers COLS columns (COLS*LPC threads).
template <int S_v, bool keep_rs_t, int LPC, int COLS = GDN_CT_COLS, bool KDA = false>
__global__ void __launch_bounds__(COLS * LPC, 1)
gated_delta_net_colthread(const float * q, const float * k, const float * v, const float * g, const float * beta,
                          const float * curr_state, float * dst, float * state,
                          int64_t H, int64_t n_tokens, int64_t n_seqs,
                          int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
                          int64_t sb1, int64_t sb2, int64_t sb3,
                          const uint3 neqk1_magic, const uint3 rq3_magic,
                          float scale, int64_t state_slot_stride, int K) {
    constexpr int R       = S_v / LPC;           // state rows per lane
    constexpr int B        = R / 2;               // chunked-kernel leaf pairs per lane
    constexpr int B4       = B / 4;               // leaf float4s per row half
    constexpr int HI4      = S_v / 8;            // float4 offset of the +S_v/2 row half
    constexpr int NT      = COLS * LPC;   // threads per block
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      sub      = threadIdx.x % LPC; // lane within the column group
    const int      col      = blockIdx.z * COLS + threadIdx.x / LPC;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const float * q_base = q + iq3 * sq3 + iq1 * sq1;
    const float * k_base = k + iq3 * sq3 + iq1 * sq1;
    const float * v_base = v + sequence * sv3 + h_idx * sv1;
    const float * b_base = beta + sequence * sb3 + h_idx * sb1;
    const float * g_base = g    + (sequence * sb3 + h_idx * sb1) * (KDA ? S_v : 1);

    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;
    state += (sequence * H + h_idx) * S_v * S_v;
    const float * s_in = curr_state + (int64_t) sequence * H * S_v * S_v + h_idx * S_v * S_v + (int64_t) col * S_v;

    __shared__ float4 sk [GDN_CT_TT][S_v / 4];
    __shared__ float4 sqs[GDN_CT_TT][S_v / 4];
    __shared__ float4 sge[KDA ? GDN_CT_TT : 1][KDA ? S_v / 4 : 1]; // KDA: exp(g) per row
    __shared__ float  sg[GDN_CT_TT];
    __shared__ float  sb[GDN_CT_TT];
    __shared__ float  sv[GDN_CT_TT][COLS]; // this block's value columns, staged with k/q (off the per-token path)

    float s[R];
    ggml_cuda_pdl_sync();
#pragma unroll
    for (int j = 0; j < B4; ++j) {
        const float4 lo = *(const float4 *) (s_in + sub * B + 4 * j);
        const float4 hi = *(const float4 *) (s_in + sub * B + S_v / 2 + 4 * j);
        s[4*j + 0] = lo.x; s[4*j + 1] = lo.y; s[4*j + 2] = lo.z; s[4*j + 3] = lo.w;
        s[B + 4*j + 0] = hi.x; s[B + 4*j + 1] = hi.y; s[B + 4*j + 2] = hi.z; s[B + 4*j + 3] = hi.w;
    }

    for (int64_t t0 = 0; t0 < n_tokens; t0 += GDN_CT_TT) {
        const int nt = (int) min((int64_t) GDN_CT_TT, n_tokens - t0);

        __syncthreads();
        for (int idx = threadIdx.x; idx < nt * (S_v / 4); idx += NT) {
            const int tt = idx / (S_v / 4);
            const int i4 = idx % (S_v / 4);
            sk [tt][i4] = *(const float4 *) (k_base + (t0 + tt) * sq2 + 4 * i4);
            sqs[tt][i4] = *(const float4 *) (q_base + (t0 + tt) * sq2 + 4 * i4);
            if constexpr (KDA) {
                const float4 gg = *(const float4 *) (g_base + (t0 + tt) * sb2 * S_v + 4 * i4);
                sge[tt][i4] = make_float4(expf(gg.x), expf(gg.y), expf(gg.z), expf(gg.w));
            }
        }
        for (int tt = threadIdx.x; tt < nt; tt += NT) {
            if constexpr (!KDA) {
                sg[tt] = expf(g_base[(t0 + tt) * sb2]);
            }
            sb[tt] = b_base[(t0 + tt) * sb2];
        }
        for (int idx = threadIdx.x; idx < nt * COLS; idx += NT) {
            const int tt = idx / COLS;
            const int c  = idx % COLS;
            sv[tt][c] = v_base[(t0 + tt) * sv2 + blockIdx.z * COLS + c];
        }
        __syncthreads();

        // attn lane partial of the previous token: its group sum runs at the top of the
        // next iteration, where the kv leaf that follows fills its DPP shadow
        float attn_lane = 0.0f;
        for (int tt = 0; tt < nt; ++tt) {
            const int64_t t    = t0 + tt;
            const float   vcol = sv[tt][threadIdx.x / LPC];

            // previous token's attn group sum: dead until the store below, so the kv
            // leaf's LDS reads and muls that follow fill its DPP shadow
            const float attn = gdn_group_sum<LPC>(attn_lane);

            // kv = sum_i S[i][col] * k[i]: one chunked-kernel leaf pair per b (row b, then row b + S_v/2).
            // The intrinsics pin the FMA contraction to the chunked kernel's emitted ops, so the
            // result is bit-identical regardless of how this instantiation gets scheduled.
            float p[B];
#pragma unroll
            for (int j = 0; j < B4; ++j) {
                const float4 klo = sk[tt][sub * B4 + j];
                const float4 khi = sk[tt][sub * B4 + HI4 + j];
                if constexpr (KDA) {
                    const float4 glo = sge[tt][sub * B4 + j];
                    const float4 ghi = sge[tt][sub * B4 + HI4 + j];
                    const float es0 = __fmul_rn(glo.x, s[4*j + 0]);
                    const float es1 = __fmul_rn(glo.y, s[4*j + 1]);
                    const float es2 = __fmul_rn(glo.z, s[4*j + 2]);
                    const float es3 = __fmul_rn(glo.w, s[4*j + 3]);
                    p[4*j + 0] = __fmul_rn(klo.x, es0);
                    p[4*j + 1] = __fmul_rn(klo.y, es1);
                    p[4*j + 2] = __fmul_rn(klo.z, es2);
                    p[4*j + 3] = __fmul_rn(klo.w, es3);
                    p[4*j + 0] = __fmaf_rn(khi.x, __fmul_rn(ghi.x, s[B + 4*j + 0]), p[4*j + 0]);
                    p[4*j + 1] = __fmaf_rn(khi.y, __fmul_rn(ghi.y, s[B + 4*j + 1]), p[4*j + 1]);
                    p[4*j + 2] = __fmaf_rn(khi.z, __fmul_rn(ghi.z, s[B + 4*j + 2]), p[4*j + 2]);
                    p[4*j + 3] = __fmaf_rn(khi.w, __fmul_rn(ghi.w, s[B + 4*j + 3]), p[4*j + 3]);
                } else {
                    p[4*j + 0] = __fmul_rn(s[4*j + 0], klo.x);
                    p[4*j + 1] = __fmul_rn(s[4*j + 1], klo.y);
                    p[4*j + 2] = __fmul_rn(s[4*j + 2], klo.z);
                    p[4*j + 3] = __fmul_rn(s[4*j + 3], klo.w);
                    p[4*j + 0] = __fmaf_rn(s[B + 4*j + 0], khi.x, p[4*j + 0]);
                    p[4*j + 1] = __fmaf_rn(s[B + 4*j + 1], khi.y, p[4*j + 1]);
                    p[4*j + 2] = __fmaf_rn(s[B + 4*j + 2], khi.z, p[4*j + 2]);
                    p[4*j + 3] = __fmaf_rn(s[B + 4*j + 3], khi.w, p[4*j + 3]);
                }
            }
            if (tt > 0 && sub == 0) {
                attn_data[(t - 1) * S_v * H + col] = attn * scale;
            }

            const float kv = gdn_group_sum<LPC>(gdn_pairwise_sum(p));
            // delta = (v - g * kv) * beta, fused the way the chunked kernel compiled it
            const float dm    = KDA ? __fsub_rn(vcol, kv) : __fmaf_rn(-sg[tt], kv, vcol);
            const float delta = __fmul_rn(dm, sb[tt]);

            // S[i][col] = g * S[i][col] + k[i] * delta ; attn = sum_i S[i][col] * q[i] (same leaf order)
            float a[B];
#pragma unroll
            for (int j = 0; j < B4; ++j) {
                const float4 klo = sk[tt][sub * B4 + j];
                const float4 khi = sk[tt][sub * B4 + HI4 + j];
                const float4 qlo = sqs[tt][sub * B4 + j];
                const float4 qhi = sqs[tt][sub * B4 + HI4 + j];
                float gs0, gs1, gs2, gs3, gsB0, gsB1, gsB2, gsB3;
                if constexpr (KDA) {
                    const float4 glo = sge[tt][sub * B4 + j];
                    const float4 ghi = sge[tt][sub * B4 + HI4 + j];
                    gs0 = __fmul_rn(glo.x, s[4*j + 0]);
                    gs1 = __fmul_rn(glo.y, s[4*j + 1]);
                    gs2 = __fmul_rn(glo.z, s[4*j + 2]);
                    gs3 = __fmul_rn(glo.w, s[4*j + 3]);
                    gsB0 = __fmul_rn(ghi.x, s[B + 4*j + 0]);
                    gsB1 = __fmul_rn(ghi.y, s[B + 4*j + 1]);
                    gsB2 = __fmul_rn(ghi.z, s[B + 4*j + 2]);
                    gsB3 = __fmul_rn(ghi.w, s[B + 4*j + 3]);
                } else {
                    gs0 = __fmul_rn(sg[tt], s[4*j + 0]);
                    gs1 = __fmul_rn(sg[tt], s[4*j + 1]);
                    gs2 = __fmul_rn(sg[tt], s[4*j + 2]);
                    gs3 = __fmul_rn(sg[tt], s[4*j + 3]);
                    gsB0 = __fmul_rn(sg[tt], s[B + 4*j + 0]);
                    gsB1 = __fmul_rn(sg[tt], s[B + 4*j + 1]);
                    gsB2 = __fmul_rn(sg[tt], s[B + 4*j + 2]);
                    gsB3 = __fmul_rn(sg[tt], s[B + 4*j + 3]);
                }
                s[4*j + 0] = __fmaf_rn(klo.x, delta, gs0);
                s[4*j + 1] = __fmaf_rn(klo.y, delta, gs1);
                s[4*j + 2] = __fmaf_rn(klo.z, delta, gs2);
                s[4*j + 3] = __fmaf_rn(klo.w, delta, gs3);
                s[B + 4*j + 0] = __fmaf_rn(khi.x, delta, gsB0);
                s[B + 4*j + 1] = __fmaf_rn(khi.y, delta, gsB1);
                s[B + 4*j + 2] = __fmaf_rn(khi.z, delta, gsB2);
                s[B + 4*j + 3] = __fmaf_rn(khi.w, delta, gsB3);
                a[4*j + 0] = __fmul_rn(s[4*j + 0], qlo.x);
                a[4*j + 1] = __fmul_rn(s[4*j + 1], qlo.y);
                a[4*j + 2] = __fmul_rn(s[4*j + 2], qlo.z);
                a[4*j + 3] = __fmul_rn(s[4*j + 3], qlo.w);
                a[4*j + 0] = __fmaf_rn(s[B + 4*j + 0], qhi.x, a[4*j + 0]);
                a[4*j + 1] = __fmaf_rn(s[B + 4*j + 1], qhi.y, a[4*j + 1]);
                a[4*j + 2] = __fmaf_rn(s[B + 4*j + 2], qhi.z, a[4*j + 2]);
                a[4*j + 3] = __fmaf_rn(s[B + 4*j + 3], qhi.w, a[4*j + 3]);
            }
            attn_lane = gdn_pairwise_sum(a);

            if constexpr (keep_rs_t) {
                const int target_slot = (int) (n_tokens - 1 - t);
                if (target_slot >= 0 && target_slot < K) {
                    float * cs = state + target_slot * state_slot_stride + (int64_t) col * S_v;
#pragma unroll
                    for (int j = 0; j < B4; ++j) {
                        *(float4 *) (cs + sub * B + 4 * j) = make_float4(s[4*j + 0], s[4*j + 1], s[4*j + 2], s[4*j + 3]);
                        *(float4 *) (cs + sub * B + S_v / 2 + 4 * j) = make_float4(s[B + 4*j + 0], s[B + 4*j + 1], s[B + 4*j + 2], s[B + 4*j + 3]);
                    }
                }
            }
        }

        // last token of the tile: no next leaf to pipeline with
        const float attn = gdn_group_sum<LPC>(attn_lane);
        if (sub == 0) {
            attn_data[(t0 + nt - 1) * S_v * H + col] = attn * scale;
        }
    }

    if constexpr (!keep_rs_t) {
        float * cs = state + (int64_t) col * S_v;
#pragma unroll
        for (int j = 0; j < B4; ++j) {
            *(float4 *) (cs + sub * B + 4 * j) = make_float4(s[4*j + 0], s[4*j + 1], s[4*j + 2], s[4*j + 3]);
            *(float4 *) (cs + sub * B + S_v / 2 + 4 * j) = make_float4(s[B + 4*j + 0], s[B + 4*j + 1], s[B + 4*j + 2], s[B + 4*j + 3]);
        }
    }
}

// Single-token decode (S_v 128): the column-per-thread kernel above spends its time on serialized
// memory round trips (state loads -> barrier -> k/q staging -> g/beta staging -> v staging, each
// loop waiting on its own loads) for ~3 MB of traffic. Here every lane loads its own state chunks
// and the k/q chunks straight into registers, all issued before the first use, with no LDS and no
// barriers. Same lane mapping and the same arithmetic in the same order as
// gated_delta_net_colthread<128, keep_rs_t, 16>, so bit-identical.
template <bool keep_rs_t, int COLS, bool KDA = false>
__global__ void __launch_bounds__(COLS * 16, 1)
gated_delta_net_decode(const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
                       const float * __restrict__ g, const float * __restrict__ beta,
                       const float * __restrict__ curr_state, float * __restrict__ dst, float * __restrict__ state,
                       int64_t H, int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
                       int64_t sb1, int64_t sb2, int64_t sb3,
                       const uint3 neqk1_magic, const uint3 rq3_magic,
                       float scale, int64_t state_slot_stride, int K) {
    constexpr int S_v = 128, LPC = 16, R = S_v / LPC;
    constexpr int B = R / 2, B4 = B / 4, HI4 = S_v / 8;
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      sub      = threadIdx.x % LPC;
    const int      col      = blockIdx.z * COLS + threadIdx.x / LPC;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const float * q_base = q + iq3 * sq3 + iq1 * sq1;
    const float * k_base = k + iq3 * sq3 + iq1 * sq1;
    const float * v_base = v + sequence * sv3 + h_idx * sv1;
    const float * b_base = beta + sequence * sb3 + h_idx * sb1;
    const float * g_base = g    + (sequence * sb3 + h_idx * sb1) * (KDA ? S_v : 1);

    float * attn_data = dst + (sequence * H + h_idx) * S_v;
    state += (sequence * H + h_idx) * S_v * S_v;
    const float * s_in = curr_state + (int64_t) sequence * H * S_v * S_v + h_idx * S_v * S_v + (int64_t) col * S_v;

    ggml_cuda_pdl_sync();
    float  s[R];
    float4 klo[B4], khi[B4], qlo[B4], qhi[B4];
    float4 glo[KDA ? B4 : 1], ghi[KDA ? B4 : 1];
    float  vcol, gl, bt;
#pragma unroll
    for (int j = 0; j < B4; ++j) {
        const float4 lo = *(const float4 *) (s_in + sub * B + 4 * j);
        const float4 hi = *(const float4 *) (s_in + sub * B + S_v / 2 + 4 * j);
        s[4*j + 0] = lo.x; s[4*j + 1] = lo.y; s[4*j + 2] = lo.z; s[4*j + 3] = lo.w;
        s[B + 4*j + 0] = hi.x; s[B + 4*j + 1] = hi.y; s[B + 4*j + 2] = hi.z; s[B + 4*j + 3] = hi.w;

        klo[j] = *(const float4 *) (k_base + sub * B + 4 * j);
        khi[j] = *(const float4 *) (k_base + sub * B + S_v / 2 + 4 * j);
        qlo[j] = *(const float4 *) (q_base + sub * B + 4 * j);
        qhi[j] = *(const float4 *) (q_base + sub * B + S_v / 2 + 4 * j);
        if constexpr (KDA) {
            glo[j] = *(const float4 *) (g_base + sub * B + 4 * j);
            ghi[j] = *(const float4 *) (g_base + sub * B + S_v / 2 + 4 * j);
        }
    }
    vcol = v_base[col];
    gl   = KDA ? 0.0f : g_base[0];
    bt   = b_base[0];

    // kv = sum_i S[i][col] * k[i]: one chunked-kernel leaf pair per b (row b, then row b + S_v/2)
    float p[B];
#pragma unroll
    for (int j = 0; j < B4; ++j) {
        if constexpr (KDA) {
            const float eg0 = expf(glo[j].x);
            const float eg1 = expf(glo[j].y);
            const float eg2 = expf(glo[j].z);
            const float eg3 = expf(glo[j].w);
            const float es0 = __fmul_rn(eg0, s[4*j + 0]);
            const float es1 = __fmul_rn(eg1, s[4*j + 1]);
            const float es2 = __fmul_rn(eg2, s[4*j + 2]);
            const float es3 = __fmul_rn(eg3, s[4*j + 3]);
            p[4*j + 0] = __fmul_rn(klo[j].x, es0);
            p[4*j + 1] = __fmul_rn(klo[j].y, es1);
            p[4*j + 2] = __fmul_rn(klo[j].z, es2);
            p[4*j + 3] = __fmul_rn(klo[j].w, es3);
            p[4*j + 0] = __fmaf_rn(khi[j].x, __fmul_rn(expf(ghi[j].x), s[B + 4*j + 0]), p[4*j + 0]);
            p[4*j + 1] = __fmaf_rn(khi[j].y, __fmul_rn(expf(ghi[j].y), s[B + 4*j + 1]), p[4*j + 1]);
            p[4*j + 2] = __fmaf_rn(khi[j].z, __fmul_rn(expf(ghi[j].z), s[B + 4*j + 2]), p[4*j + 2]);
            p[4*j + 3] = __fmaf_rn(khi[j].w, __fmul_rn(expf(ghi[j].w), s[B + 4*j + 3]), p[4*j + 3]);
        } else {
            p[4*j + 0] = __fmul_rn(s[4*j + 0], klo[j].x);
            p[4*j + 1] = __fmul_rn(s[4*j + 1], klo[j].y);
            p[4*j + 2] = __fmul_rn(s[4*j + 2], klo[j].z);
            p[4*j + 3] = __fmul_rn(s[4*j + 3], klo[j].w);
            p[4*j + 0] = __fmaf_rn(s[B + 4*j + 0], khi[j].x, p[4*j + 0]);
            p[4*j + 1] = __fmaf_rn(s[B + 4*j + 1], khi[j].y, p[4*j + 1]);
            p[4*j + 2] = __fmaf_rn(s[B + 4*j + 2], khi[j].z, p[4*j + 2]);
            p[4*j + 3] = __fmaf_rn(s[B + 4*j + 3], khi[j].w, p[4*j + 3]);
        }
    }
    const float kv = gdn_group_sum<LPC>(gdn_pairwise_sum(p));
    // delta = (v - g * kv) * beta, fused the way the chunked kernel compiled it
    const float gv    = KDA ? 1.0f : expf(gl);
    const float dm    = KDA ? __fsub_rn(vcol, kv) : __fmaf_rn(-gv, kv, vcol);
    const float delta = __fmul_rn(dm, bt);

    // S[i][col] = g * S[i][col] + k[i] * delta ; attn = sum_i S[i][col] * q[i] (same leaf order)
    float a[B];
#pragma unroll
    for (int j = 0; j < B4; ++j) {
        float gs0, gs1, gs2, gs3, gsB0, gsB1, gsB2, gsB3;
        if constexpr (KDA) {
            gs0 = __fmul_rn(expf(glo[j].x), s[4*j + 0]);
            gs1 = __fmul_rn(expf(glo[j].y), s[4*j + 1]);
            gs2 = __fmul_rn(expf(glo[j].z), s[4*j + 2]);
            gs3 = __fmul_rn(expf(glo[j].w), s[4*j + 3]);
            gsB0 = __fmul_rn(expf(ghi[j].x), s[B + 4*j + 0]);
            gsB1 = __fmul_rn(expf(ghi[j].y), s[B + 4*j + 1]);
            gsB2 = __fmul_rn(expf(ghi[j].z), s[B + 4*j + 2]);
            gsB3 = __fmul_rn(expf(ghi[j].w), s[B + 4*j + 3]);
        } else {
            gs0 = __fmul_rn(gv, s[4*j + 0]);
            gs1 = __fmul_rn(gv, s[4*j + 1]);
            gs2 = __fmul_rn(gv, s[4*j + 2]);
            gs3 = __fmul_rn(gv, s[4*j + 3]);
            gsB0 = __fmul_rn(gv, s[B + 4*j + 0]);
            gsB1 = __fmul_rn(gv, s[B + 4*j + 1]);
            gsB2 = __fmul_rn(gv, s[B + 4*j + 2]);
            gsB3 = __fmul_rn(gv, s[B + 4*j + 3]);
        }
        s[4*j + 0] = __fmaf_rn(klo[j].x, delta, gs0);
        s[4*j + 1] = __fmaf_rn(klo[j].y, delta, gs1);
        s[4*j + 2] = __fmaf_rn(klo[j].z, delta, gs2);
        s[4*j + 3] = __fmaf_rn(klo[j].w, delta, gs3);
        s[B + 4*j + 0] = __fmaf_rn(khi[j].x, delta, gsB0);
        s[B + 4*j + 1] = __fmaf_rn(khi[j].y, delta, gsB1);
        s[B + 4*j + 2] = __fmaf_rn(khi[j].z, delta, gsB2);
        s[B + 4*j + 3] = __fmaf_rn(khi[j].w, delta, gsB3);
        a[4*j + 0] = __fmul_rn(s[4*j + 0], qlo[j].x);
        a[4*j + 1] = __fmul_rn(s[4*j + 1], qlo[j].y);
        a[4*j + 2] = __fmul_rn(s[4*j + 2], qlo[j].z);
        a[4*j + 3] = __fmul_rn(s[4*j + 3], qlo[j].w);
        a[4*j + 0] = __fmaf_rn(s[B + 4*j + 0], qhi[j].x, a[4*j + 0]);
        a[4*j + 1] = __fmaf_rn(s[B + 4*j + 1], qhi[j].y, a[4*j + 1]);
        a[4*j + 2] = __fmaf_rn(s[B + 4*j + 2], qhi[j].z, a[4*j + 2]);
        a[4*j + 3] = __fmaf_rn(s[B + 4*j + 3], qhi[j].w, a[4*j + 3]);
    }
    const float attn = gdn_group_sum<LPC>(gdn_pairwise_sum(a));

    if (sub == 0) {
        attn_data[col] = attn * scale;
    }

    if constexpr (keep_rs_t) {
        const int target_slot = 0;
        if (target_slot < K) {
            float * cs = state + target_slot * state_slot_stride + (int64_t) col * S_v;
#pragma unroll
            for (int j = 0; j < B4; ++j) {
                *(float4 *) (cs + sub * B + 4 * j) = make_float4(s[4*j + 0], s[4*j + 1], s[4*j + 2], s[4*j + 3]);
                *(float4 *) (cs + sub * B + S_v / 2 + 4 * j) = make_float4(s[B + 4*j + 0], s[B + 4*j + 1], s[B + 4*j + 2], s[B + 4*j + 3]);
            }
        }
    } else {
        float * cs = state + (int64_t) col * S_v;
#pragma unroll
        for (int j = 0; j < B4; ++j) {
            *(float4 *) (cs + sub * B + 4 * j) = make_float4(s[4*j + 0], s[4*j + 1], s[4*j + 2], s[4*j + 3]);
            *(float4 *) (cs + sub * B + S_v / 2 + 4 * j) = make_float4(s[B + 4*j + 0], s[B + 4*j + 1], s[B + 4*j + 2], s[B + 4*j + 3]);
        }
    }
}

template <bool KDA, bool keep_rs_t>
bool launch_gated_delta_net_colthread(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    // GGML_GDN_NO_COLTHREAD=1 keeps gated_delta_net_cuda / the chunked kernel
    static const bool no_colthread = [] { const char * e = getenv("GGML_GDN_NO_COLTHREAD"); return e && atoi(e) != 0; }();
    if (no_colthread || S_v != 128 || (sq2 % 4) != 0 ||
            ((uintptr_t) q_d) % 16 != 0 || ((uintptr_t) k_d) % 16 != 0 || ((uintptr_t) s_d) % 16 != 0 ||
            (KDA && ((uintptr_t) g_d) % 16 != 0)) {
        return false;
    }
    // the leaf mapping reproduces the chunked kernel's warp-64 reduction
    if (ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size != 64) {
        return false;
    }

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    // single-token decode only (gfx906: 35.7 -> 16 us against the generic kernel); GGML_GDN_DECODE=0 disables
    static const bool decode = [] { const char * e = getenv("GGML_GDN_DECODE"); return !e || atoi(e) != 0; }();
    if (n_tokens == 1 && decode) {
        constexpr int C = 32;
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(H, n_seqs, 128 / C), C * 16, 0, stream);
        ggml_cuda_kernel_launch(gated_delta_net_decode<keep_rs_t, C, KDA>, lp,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
        return true;
    }

    if (n_tokens < 8) {
        constexpr int C = 32;
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(H, n_seqs, 128 / C), C * 16, 0, stream);
        ggml_cuda_kernel_launch(gated_delta_net_colthread<128, keep_rs_t, 16, C, KDA>, lp,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
        return true;
    }

    // lanes per column: 8 on GCN (gfx906, 384 tokens: 317 -> 290 ms per run), 4 elsewhere
    static const int lpc = [] {
        const char * e = getenv("GGML_GDN_LPC");
        return e ? atoi(e) : (GGML_CUDA_CC_IS_GCN(ggml_cuda_info().devices[ggml_cuda_get_device()].cc) ? 8 : 4);
    }();
    const dim3 grid(H, n_seqs, 128 / GDN_CT_COLS);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(grid, GDN_CT_COLS * lpc, 0, stream);
    if (lpc == 16) {
        ggml_cuda_kernel_launch(gated_delta_net_colthread<128, keep_rs_t, 16, GDN_CT_COLS, KDA>, lp,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
    } else if (lpc == 8) {
        ggml_cuda_kernel_launch(gated_delta_net_colthread<128, keep_rs_t, 8, GDN_CT_COLS, KDA>, lp,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
    } else if (lpc == 2) {
        ggml_cuda_kernel_launch(gated_delta_net_colthread<128, keep_rs_t, 2, GDN_CT_COLS, KDA>, lp,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
    } else {
        ggml_cuda_kernel_launch(gated_delta_net_colthread<128, keep_rs_t, 4, GDN_CT_COLS, KDA>, lp,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
    }
    return true;
}

// instantiated here, called from gated_delta_net.cu
template bool launch_gated_delta_net_colthread<true, false>(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream);
template bool launch_gated_delta_net_colthread<false, false>(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream);
template bool launch_gated_delta_net_colthread<true, true>(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream);
template bool launch_gated_delta_net_colthread<false, true>(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream);
