#include "common.cuh"
#include "fattn-common.cuh"

#ifdef GGML_USE_HIP
static bool ggml_cuda_fattn_gqa6_mtp_supported(const int cc, const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (cc != GGML_CUDA_CC_VEGA20 || Q->ne[0] != 256 ||
            Q->ne[1] < 2 || Q->ne[1] > 5 || Q->ne[2] != 24 || Q->ne[3] != 1 ||
            K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_Q8_0 ||
            K->ne[0] != 256 || V->ne[0] != 256 || K->ne[2] != 4 || V->ne[2] != 4 ||
            K->ne[1] < (Q->ne[1] == 5 ? 65536 : Q->ne[1] == 4 ? 5120 : 2816) || K->ne[1] % 256 != 0 || dst->src[4] != nullptr) {
        return false;
    }
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    static const bool enabled = []() {
        const char * gqa2 = getenv("GGML_CUDA_GFX906_Q8_GQA2");
        const char * gqa6 = getenv("GGML_CUDA_GFX906_Q8_GQA6");
        const char * mtp = getenv("GGML_CUDA_GFX906_Q8_GQA6_MTP");
        return (gqa2 == nullptr || atoi(gqa2) != 0) &&
               (gqa6 == nullptr || atoi(gqa6) != 0) &&
               (mtp == nullptr || atoi(mtp) != 0);
    }();
    return enabled;
}
#endif

static int ggml_cuda_fattn_vec_get_nthreads_host(const int cc) {
    return 128;
    GGML_UNUSED(cc);
}

static constexpr __device__ int ggml_cuda_fattn_vec_get_nthreads_device() {
    return 128;
}

// Currently llvm with the amdgcn target does not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap> // D == head size
__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device(), 1)
static __global__ void flash_attn_ext_vec(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    ggml_cuda_pdl_lc();
#ifdef FLASH_ATTN_AVAILABLE
    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks    = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

#ifdef GGML_USE_HIP
#ifdef RDNA
    constexpr int nthreads_KQ_q = 2;
#else
    constexpr int nthreads_KQ_q = 4;
#endif // RDNA
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#else
    constexpr int nthreads_KQ_q = (D/4 < 32 ? D/4 : 32);
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#endif // GGML_USE_HIP

    constexpr int nthreads    = ggml_cuda_fattn_vec_get_nthreads_device();
    constexpr int nthreads_KQ = (type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_BF16) ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr int nthreads_V  = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) ? 128 / cpy_nb : nthreads_V_q;

    static_assert(WARP_SIZE % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(WARP_SIZE % nthreads_V  == 0, "bad nthreads_V");

    constexpr int V_rows_per_thread = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) ? 2*cpy_ne : 4;
    constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ>();
    constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16 && type_K != GGML_TYPE_BF16;
#ifdef V_DOT2_F32_F16_AVAILABLE
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, half,  V_rows_per_thread>();
#else
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, float, V_rows_per_thread>();
#endif // V_DOT2_F32_F16_AVAILABLE

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const half * maskh  = (const half  *) (mask + nb33*(sequence % ne33) + nb31*ic0);
    // Mask row stride in halves. Must come from nb31, not ne11: under the
    // sequence-split fattn path ne11 is the KV SLICE length while the mask
    // keeps its full-width rows. Identical to ne11 for contiguous masks.
    const int s31 = nb31 / (int) sizeof(half);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = nthreads / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    constexpr int ne_KQ      = ncols*D;
    constexpr int ne_combine = nwarps*V_cols_per_iter*D;
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2            VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ half   KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#else
    float2           VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ float  KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#endif // V_DOT2_F32_F16_AVAILABLE

    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2  Q_reg[ncols][(D/2)/nthreads_KQ]; // Will be initialized completely.
#else
    __align__(16) float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // V_DOT2_F32_F16_AVAILABLE
    int    Q_i32[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    float2  Q_ds[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];

    ggml_cuda_pdl_sync();
    if constexpr (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 1 && ic0 + j >= int(ne01.z)) {
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    if (i0 + WARP_SIZE <= int(D/sizeof(int)) || i < int(D/sizeof(int))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
                }
            } else {
                const float * Q_f = (const float *) (Q + j*nb01);
                constexpr int nthreads_quantize = D/sizeof(int) < WARP_SIZE ? D/sizeof(int) : WARP_SIZE;
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<float2, nthreads_quantize>
                        (Q_f + i0*sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
#ifdef V_DOT2_F32_F16_AVAILABLE
        const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;

                __align__(16) float2 tmp[cpy_ne] = {{0.0f, 0.0f}};
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0/nthreads_KQ + i1] = make_half2(tmp[i1].x, tmp[i1].y);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#else
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x *= scale;
                Q_reg[j][k].y *= scale;
            }
        }
#endif // V_DOT2_F32_F16_AVAILABLE
    }

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    K     += blockIdx.y*nthreads * nb11;
    V     += blockIdx.y*nthreads * nb21;
    maskh += blockIdx.y*nthreads;
    for (int k_VKQ_0 = blockIdx.y*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads,
             // Increment pointers after each loop:
             K += gridDim.y*nthreads*nb11, V += gridDim.y*nthreads*nb21, maskh += gridDim.y*nthreads) {

        // Calculate KQ tile and keep track of new maximum KQ values:
        float KQ_reg[ncols]; // KQ in registers.

        float KQ_max_new[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE + (nthreads_KQ == WARP_SIZE ? 0 : (threadIdx.x & ~(nthreads_KQ-1))) + i_KQ_0;

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum<nthreads_KQ>(sum);

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                if (mask && (ncols == 1 || ic0 + j < int(ne01.z))) {
                    sum += slope*__half2float(maskh[j*s31 + i_KQ]);
                }

                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum + FATTN_KQ_MAX_OFFSET);

                if ((nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ) == uint32_t(i_KQ_0)) {
                    KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
            const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }

#ifndef GGML_USE_HIP
        __syncwarp();
#endif // GGML_USE_HIP

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V);

#ifdef V_DOT2_F32_F16_AVAILABLE
            half2 KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = __half2half2(KQ[j*nthreads + k]);
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                half2 tmp[V_rows_per_thread/2];
                if constexpr (type_V == GGML_TYPE_BF16) {
                    float2 tmp_f[V_rows_per_thread/2];
                    dequantize_V(V + k*nb21, tmp_f,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                    for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
                        tmp[i_VKQ_1] = __float22half2_rn(tmp_f[i_VKQ_1]);
                    }
                } else {
                    dequantize_V(V + k*nb21, tmp,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
                }
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#else
            float KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                float2 tmp[V_rows_per_thread/2];
                dequantize_V(V + k*nb21, tmp,
                    2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x += tmp[i_VKQ_1].x*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y += tmp[i_VKQ_1].y*KQ_k[j];
                    }
                }
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    if (sinks && blockIdx.y == 0) {
        const float sink = ((const float *) sinks)[head];

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            const float kqmax_new_j = fmaxf(sink, KQ_max[j]);
            const float KQ_max_scale = expf(KQ_max[j] - kqmax_new_j);
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + (threadIdx.x == 0 ? expf(sink - KQ_max[j]) : 0.0f);

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 1 && ic0 + j_VKQ >= int(ne01.z)) {
            break;
        }

        float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = expf(KQ_max[j_VKQ] - kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

#ifdef V_DOT2_F32_F16_AVAILABLE
        half2 * VKQ_tmp = (half2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

        const half2 kqmax_scale_h2 = make_half2(kqmax_scale, kqmax_scale);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread*sizeof(half)>(VKQ_tmp + i_VKQ, &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
        }
#else
        float2 * VKQ_tmp = (float2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y *= kqmax_scale;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#endif // V_DOT2_F32_F16_AVAILABLE

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
        if (threadIdx.x == 0) {
            KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ];
        }

        __syncthreads();

        if (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
            KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += float(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                if (gridDim.y == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                dst[(((sequence*int(ne01.z) + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D + i0 + tid] = dst_val;
            }
        }

        if (j_VKQ < ncols-1) {
            __syncthreads();
        }

    }

    if (gridDim.y != 1 && tid < ncols && (ncols == 1 || ic0 + tid < int(ne01.z))) {
        dst_meta[((sequence*int(ne01.z) + ic0 + tid)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

// Adapted from mistrjirka/llama.cpp 8d600f5f2860aa4f09752898b1bca33d4cb986fb.
// Two query heads share Q8_0 K/V loads; Q8_1 queries stay in shared memory.
// P.V uses FP32 accumulation, unlike the stock gfx906 vector kernel.
template <int D, int NH>
__launch_bounds__(D, 1)
static __global__ void flash_attn_ext_vec_q8_gqa(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3 ne01, const int32_t ne02, const int32_t ne03,
                           const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                           const int32_t nb11, const int32_t nb12, const int64_t nb13,
                           const int32_t nb21, const int32_t nb22, const int64_t nb23,
                           const int32_t ne31, const int32_t ne32, const int32_t ne33,
                           const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(GGML_USE_HIP) && defined(__gfx906__)
    static_assert(D == 256, "Q8 GQA kernel is specialized for D=256");
    static_assert(NH == 2, "only GQA2 is currently instantiated");
    static_assert(D % WARP_SIZE == 0, "bad D");

    GGML_UNUSED(logit_softcap);
    GGML_UNUSED(ne00);
    GGML_UNUSED(ne03);
    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
    GGML_UNUSED(ne31);
    GGML_UNUSED(ne32);
    GGML_UNUSED(nb32);

    constexpr int nwarps = D / WARP_SIZE;
    constexpr int n_q_i32 = D / sizeof(int);
    constexpr int n_q_ds  = D / QK8_1;

    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    __builtin_assume(tid < D);

    const int ic0       = blockIdx.x;
    const int ne02g     = ne02 / NH;
    const int sequence  = blockIdx.z / ne02g;
    const int head0     = (blockIdx.z - sequence * ne02g) * NH;
    const int gqa_ratio = ne02 / ne12;

    Q += (size_t) nb03 * sequence + (size_t) nb02 * head0 + (size_t) nb01 * ic0;
    K += (size_t) nb13 * sequence + (size_t) nb12 * (head0 / gqa_ratio);
    V += (size_t) nb23 * sequence + (size_t) nb22 * (head0 / gqa_ratio);

    const half * maskh = mask
        ? (const half *) (mask + (size_t) nb33 * (sequence % ne33) + (size_t) nb31 * ic0)
        : nullptr;
    const float * sinksf = (const float *) sinks;

    float slope[NH];
#pragma unroll
    for (int j = 0; j < NH; ++j) {
        slope[j] = get_alibi_slope(max_bias, head0 + j, n_head_log2, m0, m1);
    }

    __shared__ int    Q_i32[NH][n_q_i32];
    __shared__ float2 Q_ds [NH][n_q_ds];
    __shared__ float  KQ   [NH][D];
    __shared__ float  kqmax_shared[NH][WARP_SIZE];
    __shared__ float  kqsum_shared[NH][WARP_SIZE];

    if (threadIdx.y == 0) {
#pragma unroll
        for (int j = 0; j < NH; ++j) {
            const float * Q_f = (const float *) (Q + (size_t) j * nb02);
#pragma unroll
            for (int i0 = 0; i0 < n_q_i32; i0 += WARP_SIZE) {
                quantize_q8_1_to_shared<float2, WARP_SIZE>(
                    Q_f + i0 * sizeof(int), scale,
                    &Q_i32[j][i0], &Q_ds[j][i0 / QI8_1]);
            }
        }
    }

#pragma unroll
    for (int j = 0; j < NH; ++j) {
        if (threadIdx.y == 0) {
            kqmax_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            kqsum_shared[j][threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

    float kqmax[NH];
    float kqsum[NH];
    float VKQ[NH];
#pragma unroll
    for (int j = 0; j < NH; ++j) {
        kqmax[j] = -FLT_MAX/2.0f;
        kqsum[j] = 0.0f;
        VKQ[j]   = 0.0f;
    }

    const int k_VKQ_max = KV_max ? KV_max[sequence * gridDim.x + blockIdx.x] : ne11;

    K     += (size_t) blockIdx.y * D * nb11;
    V     += (size_t) blockIdx.y * D * nb21;
    if (maskh) {
        maskh += blockIdx.y * D;
    }

    for (int k_VKQ_0 = blockIdx.y * D; k_VKQ_0 < k_VKQ_max;
         k_VKQ_0 += gridDim.y * D,
         K += (size_t) gridDim.y * D * nb11,
         V += (size_t) gridDim.y * D * nb21) {

        float kqmax_new[NH];
#pragma unroll
        for (int j = 0; j < NH; ++j) {
            kqmax_new[j] = kqmax[j];
        }

        // One K-row load feeds both Q heads.
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;
            const block_q8_0 * K_row = (const block_q8_0 *) (K + (size_t) i_KQ * nb11);

            float sums[NH] = {0.0f, 0.0f};
#pragma unroll
            for (int k0 = 0; k0 < n_q_i32; k0 += WARP_SIZE) {
                const int k = k0 + threadIdx.x;
                const int ib  = k / QI8_0;
                const int iqs = k % QI8_0;

                int v;
                ggml_cuda_memcpy_1<sizeof(v), 2>(&v, K_row[ib].qs + 4 * iqs);
                const float kd = __half2float(K_row[ib].d);

#pragma unroll
                for (int j = 0; j < NH; ++j) {
                    const float qd = Q_ds[j][k / QI8_1].x;
                    sums[j] += vec_dot_q8_0_q8_1_impl<float, 1>(
                        &v, &Q_i32[j][k], kd, qd);
                }
            }

#pragma unroll
            for (int j = 0; j < NH; ++j) {
                float sum = warp_reduce_sum(sums[j]);
                if (maskh) {
                    sum += slope[j] * __half2float(maskh[i_KQ]);
                }
                kqmax_new[j] = fmaxf(kqmax_new[j], sum + FATTN_KQ_MAX_OFFSET);
                if (threadIdx.x == 0) {
                    KQ[j][i_KQ] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < NH; ++j) {
            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = kqmax_new[j];
            }
        }
        __syncthreads();

#pragma unroll
        for (int j = 0; j < NH; ++j) {
            float km = kqmax_shared[j][threadIdx.x];
            km = warp_reduce_max(km);
            const float rescale = expf(kqmax[j] - km);
            kqmax[j] = km;

            const float val = expf(KQ[j][tid] - km);
            kqsum[j] = kqsum[j] * rescale + val;
            KQ[j][tid] = val;
            VKQ[j] *= rescale;
        }
        __syncthreads();

        // One V dequantization feeds both output heads.
#pragma unroll 4
        for (int k = 0; k < D; ++k) {
            const block_q8_0 * V_row = (const block_q8_0 *) (V + (size_t) k * nb21);
            const int ib  = tid / QK8_0;
            const int iqs = tid % QK8_0;
            const float V_ki = __half2float(V_row[ib].d) * V_row[ib].qs[iqs];
#pragma unroll
            for (int j = 0; j < NH; ++j) {
                VKQ[j] += V_ki * KQ[j][k];
            }
        }
        __syncthreads();

        if (maskh) {
            maskh += gridDim.y * D;
        }
    }

    if (sinksf && blockIdx.y == 0) {
#pragma unroll
        for (int j = 0; j < NH; ++j) {
            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = fmaxf(kqmax[j], sinksf[head0 + j]);
            }
        }
        __syncthreads();

#pragma unroll
        for (int j = 0; j < NH; ++j) {
            float km = kqmax_shared[j][threadIdx.x];
            km = warp_reduce_max(km);
            const float rescale = expf(kqmax[j] - km);
            kqmax[j] = km;
            kqsum[j] *= rescale;
            if (tid == 0) {
                kqsum[j] += expf(sinksf[head0 + j] - km);
            }
            VKQ[j] *= rescale;
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < NH; ++j) {
        float s = warp_reduce_sum(kqsum[j]);
        if (threadIdx.x == 0) {
            kqsum_shared[j][threadIdx.y] = s;
        }
    }
    __syncthreads();

#pragma unroll
    for (int j = 0; j < NH; ++j) {
        float denom = kqsum_shared[j][threadIdx.x];
        denom = warp_reduce_sum(denom);

        float out = VKQ[j];
        if (gridDim.y == 1) {
            out /= denom;
        }
        dst[(((size_t) (sequence * int(ne01.z) + ic0) * ne02 + head0 + j) *
             gridDim.y + blockIdx.y) * D + tid] = out;

        if (gridDim.y != 1 && tid == j) {
            dst_meta[((size_t) (sequence * int(ne01.z) + ic0) * ne02 + head0 + j) *
                     gridDim.y + blockIdx.y] = make_float2(kqmax[j], denom);
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03,
        ne10, ne11, ne12, ne13, nb11, nb12, nb13,
        nb21, nb22, nb23, ne31, ne32, ne33, nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif
}

// Six query heads share each Q8_0 K/V load. The guarded decode shape has
// unit mask slope and no sinks. Eight-lane QK reductions and two adjacent
// P.V outputs per thread retain FP32 accumulation. On HIP, the second
// launch bound requests two waves per execution unit from the compiler.
template <int D>
__launch_bounds__(128, 2)
static __global__ void flash_attn_ext_vec_q8_gqa6(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3 ne01, const int32_t ne02, const int32_t ne03,
                           const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                           const int32_t nb11, const int32_t nb12, const int64_t nb13,
                           const int32_t nb21, const int32_t nb22, const int64_t nb23,
                           const int32_t ne31, const int32_t ne32, const int32_t ne33,
                           const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(GGML_USE_HIP) && defined(__gfx906__)
    static_assert(D == 256, "Q8 GQA6 kernel is specialized for D=256");
    constexpr int NH = 6;
    constexpr int REDUCE_HEADS = 6;
    constexpr int QK_LANES = 8;
    constexpr int NT = 128;
    static_assert(NT % WARP_SIZE == 0 && D % NT == 0, "bad output partition");

    GGML_UNUSED_VARS(sinks, max_bias, m0, m1, n_head_log2);
    GGML_UNUSED(logit_softcap);
    GGML_UNUSED(ne00);
    GGML_UNUSED(ne03);
    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
    GGML_UNUSED(ne31);
    GGML_UNUSED(ne32);
    GGML_UNUSED(nb32);

    constexpr int NE = D / NT;
    constexpr int n_q_i32 = D / sizeof(int);
    constexpr int n_q_ds  = D / QK8_1;

    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    __builtin_assume(tid < NT);

    const int ic0       = blockIdx.x;
    const int ne02g     = ne02 / NH;
    const int sequence  = blockIdx.z / ne02g;
    const int head0     = (blockIdx.z - sequence * ne02g) * NH;
    const int gqa_ratio = ne02 / ne12;

    Q += (size_t) nb03 * sequence + (size_t) nb02 * head0 + (size_t) nb01 * ic0;
    K += (size_t) nb13 * sequence + (size_t) nb12 * (head0 / gqa_ratio);
    V += (size_t) nb23 * sequence + (size_t) nb22 * (head0 / gqa_ratio);

    const half * maskh = mask
        ? (const half *) (mask + (size_t) nb33 * (sequence % ne33) + (size_t) nb31 * ic0)
        : nullptr;
    __shared__ int    Q_i32[NH][n_q_i32];
    __shared__ float2 Q_ds [NH][n_q_ds];
    __shared__ float  KQ   [NH][D];
    __shared__ float  kqmax_shared[NH][WARP_SIZE];
    __shared__ float  kqsum_shared[NH][WARP_SIZE];

    if (threadIdx.y == 0) {
#pragma unroll
        for (int j = 0; j < NH; ++j) {
            const float * Q_f = (const float *) (Q + (size_t) j * nb02);
#pragma unroll
            for (int i0 = 0; i0 < n_q_i32; i0 += WARP_SIZE) {
                quantize_q8_1_to_shared<float2, WARP_SIZE>(
                    Q_f + i0 * sizeof(int), scale,
                    &Q_i32[j][i0], &Q_ds[j][i0 / QI8_1]);
            }
        }
    }

#pragma unroll
    for (int j = 0; j < NH; ++j) {
        if (threadIdx.y == 0) {
            kqmax_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            kqsum_shared[j][threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

    float kqmax[NH];
    float kqsum[NH];
    float VKQ[NH][NE];
#pragma unroll
    for (int j = 0; j < NH; ++j) {
        kqmax[j] = -FLT_MAX/2.0f;
        kqsum[j] = 0.0f;
#pragma unroll
        for (int component = 0; component < NE; ++component) {
            VKQ[j][component] = 0.0f;
        }
    }

    const int k_VKQ_max = KV_max ? KV_max[sequence * gridDim.x + blockIdx.x] : ne11;

    K     += (size_t) blockIdx.y * D * nb11;
    V     += (size_t) blockIdx.y * D * nb21;
    if (maskh) {
        maskh += blockIdx.y * D;
    }

    for (int k_VKQ_0 = blockIdx.y * D; k_VKQ_0 < k_VKQ_max;
         k_VKQ_0 += gridDim.y * D,
         K += (size_t) gridDim.y * D * nb11,
         V += (size_t) gridDim.y * D * nb21) {

        float kqmax_new[NH];
#pragma unroll
        for (int j = 0; j < NH; ++j) {
            kqmax_new[j] = kqmax[j];
        }

        {
            // Each eight-lane subgroup covers one K row; each lane handles
            // one complete Q8 block.
            // Accumulate packed integer dots before applying the block scales.
            constexpr int words_per_lane = n_q_i32 / QK_LANES;
            constexpr int lanes_per_q8_block = QI8_0 / words_per_lane;
            constexpr int rows_per_iter = NT / QK_LANES;
            static_assert(QI8_0 % words_per_lane == 0, "subgroup must split Q8 blocks exactly");
            const int qk_lane = tid % QK_LANES;
            const int ib = qk_lane / lanes_per_q8_block;
            const int word_offset = (qk_lane % lanes_per_q8_block) * words_per_lane;
            const int row_group = tid / QK_LANES;

#pragma unroll 1
            for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += rows_per_iter) {
                const int i_KQ = i_KQ_0 + row_group;
                const block_q8_0 * K_row = (const block_q8_0 *) (K + (size_t) i_KQ * nb11);
                int v[words_per_lane];
#pragma unroll
                for (int iw = 0; iw < words_per_lane; iw += 4) {
                    // Q8 block payloads are only two-byte aligned.
                    ggml_cuda_memcpy_1<16, 2>(&v[iw], K_row[ib].qs + 4 * (word_offset + iw));
                }
                const float kd = __half2float(K_row[ib].d);
                float sums[NH] = {};
                // All query heads in this group use the same mask row and unit slope.
                const float mask_value = maskh ? __half2float(maskh[i_KQ]) : 0.0f;
#pragma unroll
                for (int j = 0; j < NH; ++j) {
                    const float qd = Q_ds[j][ib].x;
                    sums[j] = vec_dot_q8_0_q8_1_impl<float, words_per_lane>(
                        v, &Q_i32[j][ib * QI8_1 + word_offset], kd, qd);
                }
                // Keep each head's xor4/xor2/xor1 reduction order, while
                // exposing independent head exchanges before their sums.
#pragma unroll
                for (int j0 = 0; j0 < NH; j0 += REDUCE_HEADS) {
                    float reduced[REDUCE_HEADS];
#pragma unroll
                    for (int r = 0; r < REDUCE_HEADS; ++r) {
                        reduced[r] = sums[j0 + r];
                    }
#pragma unroll
                    for (int offset = QK_LANES / 2; offset > 0; offset >>= 1) {
                        float exchanged[REDUCE_HEADS];
#pragma unroll
                        for (int r = 0; r < REDUCE_HEADS; ++r) {
                            exchanged[r] = __shfl_xor_sync(
                                0xffffffff, reduced[r], offset, QK_LANES);
                        }
#pragma unroll
                        for (int r = 0; r < REDUCE_HEADS; ++r) {
                            reduced[r] += exchanged[r];
                        }
                    }
#pragma unroll
                    for (int r = 0; r < REDUCE_HEADS; ++r) {
                        const int j = j0 + r;
                        float sum = reduced[r];
                        if (maskh) {
                            sum += mask_value;
                        }
                        kqmax_new[j] = fmaxf(kqmax_new[j], sum + FATTN_KQ_MAX_OFFSET);
                        if (qk_lane == 0) {
                            KQ[j][i_KQ] = sum;
                        }
                    }
                }
            }

            // Merge the independent subgroups' maxima within each logical warp
            // before the cross-warp stage (xor 8, then xor 16).
#pragma unroll
            for (int j = 0; j < NH; ++j) {
#pragma unroll
                for (int offset = QK_LANES; offset < WARP_SIZE; offset <<= 1) {
                    kqmax_new[j] = fmaxf(kqmax_new[j], __shfl_xor_sync(
                        0xffffffff, kqmax_new[j], offset, WARP_SIZE));
                }
            }
        }

#pragma unroll
        for (int j = 0; j < NH; ++j) {
            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = kqmax_new[j];
            }
        }
        __syncthreads();

#pragma unroll
        for (int j = 0; j < NH; ++j) {
            float km = kqmax_shared[j][threadIdx.x];
            km = warp_reduce_max(km);
            const float rescale = expf(kqmax[j] - km);
            kqmax[j] = km;

            const float val = expf(KQ[j][tid] - km);
            kqsum[j] = kqsum[j] * rescale + val;
            KQ[j][tid] = val;
            // NT128 owns two probability positions, but rescales its old sum once.
#pragma unroll
            for (int component = 1; component < NE; ++component) {
                const int position = tid + component * NT;
                const float val_next = expf(KQ[j][position] - km);
                kqsum[j] += val_next;
                KQ[j][position] = val_next;
            }
#pragma unroll
            for (int component = 0; component < NE; ++component) {
                VKQ[j][component] *= rescale;
            }
        }
        __syncthreads();

        // One V dequantization feeds all output heads.
#pragma unroll 4
        for (int k = 0; k < D; ++k) {
            const block_q8_0 * V_row = (const block_q8_0 *) (V + (size_t) k * nb21);
            const int ib  = (NE * tid) / QK8_0;
            const int iqs = (NE * tid) % QK8_0;
            {
                // Two adjacent signed Q8 values always share their block scale.
                // iqs is even and at most 30, so the two-byte load stays in qs[].
                uint16_t packed;
                ggml_cuda_memcpy_1<2, 2>(&packed, V_row[ib].qs + iqs);
                const float vd = __half2float(V_row[ib].d);
                const float V_ki0 = vd * (int8_t) (packed & 0xff);
                const float V_ki1 = vd * (int8_t) (packed >> 8);
#pragma unroll
                for (int j = 0; j < NH; ++j) {
                    const float probability = KQ[j][k];
                    VKQ[j][0] += V_ki0 * probability;
                    VKQ[j][1] += V_ki1 * probability;
                }
            }
        }
        __syncthreads();

        if (maskh) {
            maskh += gridDim.y * D;
        }
    }

#pragma unroll
    for (int j = 0; j < NH; ++j) {
        float s = warp_reduce_sum(kqsum[j]);
        if (threadIdx.x == 0) {
            kqsum_shared[j][threadIdx.y] = s;
        }
    }
    __syncthreads();

#pragma unroll
    for (int j = 0; j < NH; ++j) {
        float denom = kqsum_shared[j][threadIdx.x];
        denom = warp_reduce_sum(denom);

#pragma unroll
        for (int component = 0; component < NE; ++component) {
            float out = VKQ[j][component];
            if (gridDim.y == 1) {
                out /= denom;
            }
            const int output_dim = NE * tid + component;
            dst[(((size_t) (sequence * int(ne01.z) + ic0) * ne02 + head0 + j) *
                 gridDim.y + blockIdx.y) * D + output_dim] = out;
        }

        if (gridDim.y != 1 && tid == j) {
            dst_meta[((size_t) (sequence * int(ne01.z) + ic0) * ne02 + head0 + j) *
                     gridDim.y + blockIdx.y] = make_float2(kqmax[j], denom);
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03,
        ne10, ne11, ne12, ne13, nb11, nb12, nb13,
        nb21, nb22, nb23, ne31, ne32, ne33, nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif
}

template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const int nthreads = ggml_cuda_fattn_vec_get_nthreads_host(cc);
    const int nwarps   = nthreads / WARP_SIZE;
#ifdef GGML_USE_HIP
    if constexpr (D == 256 && cols_per_block == 1 &&
            type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0 && !use_logit_softcap) {
        const ggml_tensor * Q = dst->src[0];
        const ggml_tensor * K = dst->src[1];
        float max_bias = 0.0f;
        memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
        static const bool enabled = []() {
            const char * e = getenv("GGML_CUDA_GFX906_Q8_GQA2");
            return e == nullptr || atoi(e) != 0;
        }();
        // Full six-head sharing is restricted to this decode shape.
        if (enabled && cc == GGML_CUDA_CC_VEGA20 && K->ne[1] >= 2816 &&
                Q->ne[1] == 1 && Q->ne[2] == 24 && Q->ne[3] == 1 && K->ne[2] == 4 &&
                dst->src[4] == nullptr && max_bias == 0.0f) {
            static const bool gqa6_enabled = []() {
                const char * e = getenv("GGML_CUDA_GFX906_Q8_GQA6");
                return e == nullptr || atoi(e) != 0;
            }();
            if (gqa6_enabled && K->ne[1] % D == 0) {
                launch_fattn<D, 1, 6>(ctx, dst, flash_attn_ext_vec_q8_gqa6<D>,
                    128 / WARP_SIZE, 0, D, false, false, false);
                return;
            }
        }
        // Shorter contexts have measured regressions at split boundaries.
        if (enabled && cc == GGML_CUDA_CC_VEGA20 && K->ne[1] >= 11776 &&
                Q->ne[1] == 1 && Q->ne[2] == 24 && Q->ne[3] == 1 && K->ne[2] == 4 &&
                dst->src[4] == nullptr && max_bias == 0.0f) {
            launch_fattn<D, 1, 2>(ctx, dst, flash_attn_ext_vec_q8_gqa<D, 2>,
                D / WARP_SIZE, 0, D, false, false, false);
            return;
        }
    }
#endif // GGML_USE_HIP
    fattn_kernel_t fattn_kernel = flash_attn_ext_vec<D, cols_per_block, type_K, type_V, use_logit_softcap>;
    const bool need_f16_K = type_K == GGML_TYPE_F16;
    const bool need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;
    launch_fattn<D, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

#ifdef GGML_USE_HIP
    if constexpr (D == 256 && type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        if (ggml_cuda_fattn_gqa6_mtp_supported(cc, dst)) {
            launch_fattn<D, 1, 6>(ctx, dst, flash_attn_ext_vec_q8_gqa6<D>,
                128 / WARP_SIZE, 0, D, false, false, false);
            return;
        }
    }
#endif

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block = 1;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 2;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    template void ggml_cuda_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_BF16); \

EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_BF16)
