#include "gated_delta_net_chunk.cuh"
#include "ggml-cuda/common.cuh"

#if defined(GGML_USE_HIP) && defined(__gfx906__)

#define GDN_GCN_DPP_ADD(name, barrier, dpp_ctrl)                                \
    static __device__ __forceinline__ float name(const float x) {               \
        float r;                                                                \
        asm volatile(                                                           \
            barrier                                                             \
            "v_add_f32_dpp %0, %1, %1 " dpp_ctrl " row_mask:0xf bank_mask:0xf"  \
            : "=v"(r) : "v"(x) : "memory");                                     \
        return r;                                                               \
    }

GDN_GCN_DPP_ADD(gdn_dpp_add_xor1, "s_nop 4\n", "quad_perm:[1,0,3,2]")
GDN_GCN_DPP_ADD(gdn_dpp_add_xor2, "s_nop 1\n", "quad_perm:[2,3,0,1]")
GDN_GCN_DPP_ADD(gdn_dpp_add_xor8, "s_nop 1\n", "row_ror:8")

#undef GDN_GCN_DPP_ADD

static __device__ __forceinline__ float gdn_dpp_xfer_xor4(const float x) {
    int d;
    asm volatile("v_mov_b32 %0, %1\n"
                 "s_nop 1\n"
                 "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
                 "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
                 : "=v"(d) : "v"(__float_as_int(x)) : "memory");
    return __int_as_float(d);
}

static __device__ __forceinline__ float gdn_dpp_xfer_xor16(const float x) {
    int d;
    asm volatile("ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n"
                 "s_waitcnt lgkmcnt(0)\n"
                 : "=v"(d) : "v"(__float_as_int(x)) : "memory");
    return __int_as_float(d);
}

template <int width>
static __device__ __forceinline__ float gdn_warp_reduce_sum(const float x) {
    static_assert(width >= 1 && (width & (width - 1)) == 0, "width must be a power of 2");
    float r = x;
    if constexpr (width >=  2) { r = gdn_dpp_add_xor1(r); }
    if constexpr (width >=  4) { r = gdn_dpp_add_xor2(r); }
    if constexpr (width >=  8) { r += gdn_dpp_xfer_xor4(r); }
    if constexpr (width >= 16) { r = gdn_dpp_add_xor8(r); }
    if constexpr (width >= 32) { r += gdn_dpp_xfer_xor16(r); }
    if constexpr (width >= 64) { r += __shfl_xor_sync(0xffffffff, r, 32, 64); }
    return r;
}

#else

template <int width>
static __device__ __forceinline__ float gdn_warp_reduce_sum(const float x) {
    return warp_reduce_sum<width>(x);
}

#endif


// Chunked Gated DeltaNet kernel.
//
// One thread-block owns a single (head, sequence) and processes it by looping over chunks of CS tokens.
// The recurrent state S_v x S_v stays in registers across chunks (it is never written back to HBM),
// which makes it a long prefill a single kernel launch with no state round-trips.
// CS = 64 for the non-KDA path and CS = 16 for the KDA (per-feature gate) path.

template <int S_v, bool KDA, int CS, int Tc>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_chunked_cuda(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ g,
        const float * __restrict__ beta,
        const float * __restrict__ curr_state,
        float *       __restrict__ dst,
        float *       __restrict__ state,
        int64_t       H,
        int64_t       n_tokens,
        int64_t       n_seqs,
        int64_t       sq1,
        int64_t       sq2,
        int64_t       sq3,
        int64_t       sv1,
        int64_t       sv2,
        int64_t       sv3,
        int64_t       sb1,
        int64_t       sb2,
        int64_t       sb3,
        const uint3   neqk1_magic,
        const uint3   rq3_magic,
        float         scale,
        int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      warp_id  = threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       state_out = state;

    const int64_t state_in_offset  = sequence * K * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state_out += state_out_offset;
    const float * state_in_base = curr_state + state_in_offset;

    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;

    const int base_col = (blockIdx.z * blockDim.y + warp_id) * Tc;

    float s_shard[rows_per_lane][Tc];
    #pragma unroll
    for (int c = 0; c < Tc; c++) {
        const int col = base_col + c;
        #pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            s_shard[r][c] = (col < S_v && i < S_v) ? state_in_base[col * S_v + i] : 0.0f;
        }
    }

        ggml_cuda_pdl_sync();

    const int n_chunks = (n_tokens + CS - 1) / CS;
    for (int c = 0; c < n_chunks; c++) {
        const int t_base = c * CS;
        const int n_local = (t_base + CS <= n_tokens) ? CS : (n_tokens - t_base);

        for (int p = 0; p < n_local; p++) {
            const int t = t_base + p;
            const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;
            const int64_t gb = sequence * sb3 + t * sb2 + h_idx * sb1;

            float k_shard[rows_per_lane];
            float q_shard[rows_per_lane];
            #pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                k_shard[r] = k_t[i];
                q_shard[r] = q_t[i];
            }
            const float beta_val = beta[gb];

            if constexpr (!KDA) {
                const float g_val = expf(g[gb]);

                for (int cc = 0; cc < Tc; cc++) {
                    const int col = base_col + cc;
                    if (col >= S_v) break;
                    const float v_col = v_t[col];

                    float kv_shard = 0.0f;
                    #pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        kv_shard += s_shard[r][cc] * k_shard[r];
                    }
                    float kv_col = gdn_warp_reduce_sum<warp_size>(kv_shard);

                    float delta_col = (v_col - g_val * kv_col) * beta_val;

                    float attn_partial = 0.0f;
                    #pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        s_shard[r][cc] = g_val * s_shard[r][cc] + k_shard[r] * delta_col;
                        attn_partial += s_shard[r][cc] * q_shard[r];
                    }

                    float attn_col = gdn_warp_reduce_sum<warp_size>(attn_partial);

                    if (lane == 0) {
                        attn_data[col] = attn_col * scale;
                    }
                }
            } else {

                const float * g_t = g + gb * S_v;
                float eg_shard[rows_per_lane];
                #pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    eg_shard[r] = (i < S_v) ? expf(g_t[i]) : 1.0f;
                }

                for (int cc = 0; cc < Tc; cc++) {
                    const int col = base_col + cc;
                    if (col >= S_v) break;
                    const float v_col = v_t[col];

                    float kv_shard = 0.0f;
                    #pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        kv_shard += eg_shard[r] * s_shard[r][cc] * k_shard[r];
                    }
                    float kv_col = gdn_warp_reduce_sum<warp_size>(kv_shard);

                    float delta_col = (v_col - kv_col) * beta_val;

                    float attn_partial = 0.0f;
                    #pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        s_shard[r][cc] = eg_shard[r] * s_shard[r][cc] + k_shard[r] * delta_col;
                        attn_partial += s_shard[r][cc] * q_shard[r];
                    }

                    float attn_col = gdn_warp_reduce_sum<warp_size>(attn_partial);

                    if (lane == 0) {
                        attn_data[col] = attn_col * scale;
                    }
                }
            }

            attn_data += (int64_t) S_v * H;
        }
    }

    #pragma unroll
    for (int c = 0; c < Tc; c++) {
        const int col = base_col + c;
        if (col >= S_v) break;
        #pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            if (i < S_v) {
                state_out[col * S_v + i] = s_shard[r][c];
            }
        }
    }
}

template <bool KDA, bool /*keep_rs_t*/>
void launch_gated_delta_net_chunk(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int K, cudaStream_t stream) {
    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const int cc = ggml_cuda_info().devices[device].cc;
    const int CS = KDA ? 16 : 64;
    const int num_warps = cc == GGML_CUDA_CC_VEGA20 ? 2 : 4;
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    switch (S_v) {
        case 16: {
            constexpr int Tc = 4;
            dim3 grid_dims(H, n_seqs, (S_v + num_warps * Tc - 1) / (num_warps * Tc));
            const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
            ggml_cuda_kernel_launch(gated_delta_net_chunked_cuda<16, KDA, KDA ? 16 : 64, Tc>, lp,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, K);
            break;
        }
        case 32: {
            constexpr int Tc = 4;
            dim3 grid_dims(H, n_seqs, (S_v + num_warps * Tc - 1) / (num_warps * Tc));
            const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
            ggml_cuda_kernel_launch(gated_delta_net_chunked_cuda<32, KDA, KDA ? 16 : 64, Tc>, lp,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, K);
            break;
        }
        case 64: {
            constexpr int Tc = 4;
            dim3 grid_dims(H, n_seqs, (S_v + num_warps * Tc - 1) / (num_warps * Tc));
            const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
            ggml_cuda_kernel_launch(gated_delta_net_chunked_cuda<64, KDA, KDA ? 16 : 64, Tc>, lp,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, K);
            break;
        }
        case 128: {
            constexpr int Tc = 2;
            dim3 grid_dims(H, n_seqs, (S_v + num_warps * Tc - 1) / (num_warps * Tc));
            const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
            ggml_cuda_kernel_launch(gated_delta_net_chunked_cuda<128, KDA, KDA ? 16 : 64, Tc>, lp,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// instantiated here, called from gated_delta_net.cu
template void launch_gated_delta_net_chunk<true, false>(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int K, cudaStream_t stream);
template void launch_gated_delta_net_chunk<false, false>(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int K, cudaStream_t stream);
