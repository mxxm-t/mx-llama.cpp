#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <cstdint>

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_MXFP4:
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool fallback = ne01 % 128 != 0;

    const bool use_native_fp4 = blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);
    const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;

    if (!ids) {
        const auto workspace_nbytes = [&](const int64_t ncols) {
            return ne13*ne12 * ncols*ne10_padded * y_block_size/y_values_per_block +
                ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ncols) * sizeof(block_q8_1_mmq);
        };

        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
        ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());

        // The fp4 path also produces a separate scale buffer, so only the plain
        // q8_1 layouts are shared.
        char * src1_q8_1_ptr = nullptr;
        bool   src1_q8_1_hit = false;
        bool   cache_pressure = false;
        int64_t chunk_ne11 = ctx.mmq_workspace_cols_cap > 0 ?
            std::min(ne11, ctx.mmq_workspace_cols_cap) : ne11;
        {
            const int64_t qs11 = src1->nb[1] / ts_src1;
            const int64_t qs12 = src1->nb[2] / ts_src1;
            const int64_t qs13 = src1->nb[3] / ts_src1;
            if (!use_native_fp4 && ctx.mmq_workspace_cols_cap == 0) {
                const int variant = 1 + (int) mmq_get_q8_1_ds_layout(src0->type);
                src1_q8_1_ptr = ggml_cuda_q8_1_cache_acquire(ctx, src1, variant, ne10_padded,
                                                             qs11, qs12, qs13, workspace_nbytes(ne11),
                                                             src1_q8_1_hit, &cache_pressure);
            }
            if (!src1_q8_1_ptr) {
                if (cache_pressure && chunk_ne11 > 1) {
                    chunk_ne11 = (chunk_ne11 + 1) / 2;
                }
                while (src1_q8_1.try_alloc(workspace_nbytes(chunk_ne11)) == nullptr) {
                    if (chunk_ne11 == 1) {
                        GGML_ABORT("CUDA MMQ: failed to allocate the minimum quantization workspace");
                    }
                    chunk_ne11 = (chunk_ne11 + 1) / 2;
                }
                src1_q8_1_ptr = src1_q8_1.get();
            }
        }

        if (chunk_ne11 != ne11) {
            if (ctx.mmq_workspace_cols_cap == 0 || chunk_ne11 < ctx.mmq_workspace_cols_cap) {
                ctx.mmq_workspace_cols_cap = chunk_ne11;
            }
            if (!ctx.mmq_workspace_cols_cap_logged) {
                GGML_LOG_WARN("CUDA MMQ: device %d workspace pressure, splitting %lld columns into chunks of %lld\n",
                              ctx.device, (long long) ne11, (long long) chunk_ne11);
                ctx.mmq_workspace_cols_cap_logged = true;
            }
        }

        if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
            src1_scale.alloc(ne13*ne12*chunk_ne11);
        }

        const int64_t qs11 = src1->nb[1] / ts_src1;
        const int64_t qs12 = src1->nb[2] / ts_src1;
        const int64_t qs13 = src1->nb[3] / ts_src1;
        for (int64_t col = 0; col < ne11; col += chunk_ne11) {
            const int64_t iter_ne11 = std::min(chunk_ne11, ne11 - col);

            if (!src1_q8_1_hit) {
                if (use_native_fp4) {
                    static constexpr size_t align_float8 = 32;
                    const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
                    static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                    quantize_mmq_fp4_cuda(src1_d + col*qs11, nullptr, src1_q8_1_ptr, src1_scale.ptr,
                                          src0->type, use_aligned_float8, ne10, qs11, qs12, qs13,
                                          ne10_padded, iter_ne11, ne12, ne13, stream);
                } else {
                    quantize_mmq_q8_1_cuda(src1_d + col*qs11, nullptr, src1_q8_1_ptr, src0->type,
                                           ne10, qs11, qs12, qs13, ne10_padded,
                                           iter_ne11, ne12, ne13, stream);
                }
                CUDA_CHECK(cudaGetLastError());
            }

            // The quantized workspace is packed to the current column chunk;
            // destination channel/sample strides keep their original tensor layout.
            const int64_t ys12 = use_native_fp4 ?
                iter_ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                iter_ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
            const int64_t ys13 = ne12*ys12;

            const mmq_args args = {
                src0_d, src0->type, (const int *) src1_q8_1_ptr, nullptr, nullptr, dst_d + col*s1,
                src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? src1_scale.ptr : nullptr,
                ne00, ne01, iter_ne11, s01, iter_ne11, s1,
                ne02, ne12, s02, ys12, s2,
                ne03, ne13, s03, ys13, s3,
                iter_ne11};
            ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        }
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    GGML_ASSERT(ne1 == n_expert_used);

    // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once and
    // scatter to its slots. ids_src1 then holds the inverse map (token slot -> compact row).
    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;
    const auto quant_workspace_nbytes = [&](const int64_t n_tokens) {
        return n_tokens*n_expert_used*ne10_padded * y_block_size/y_values_per_block +
            ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
    };

    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool());

    // Allocate the largest buffer first. Otherwise the best-fit pool can hand
    // a large reusable block to a tiny ID array and strand it until this op
    // completes. If the complete set does not fit, halve the independent token
    // dimension and reuse the bounded workspaces for each chunk.
    int64_t chunk_ne12 = ctx.mmq_id_workspace_tokens_cap > 0 ?
        std::min(ne12, ctx.mmq_id_workspace_tokens_cap) : ne12;
    auto reset_workspaces = [&]() {
        expert_bounds.reset();
        ids_dst.reset();
        ids_src1.reset();
        src1_scale.reset();
        src1_q8_1.reset();
    };
    auto try_workspaces = [&]() {
        const int64_t chunk_rows = chunk_ne12*n_expert_used;
        bool ok = src1_q8_1.try_alloc(quant_workspace_nbytes(chunk_ne12)) != nullptr;
        if (ok && src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
            ok = src1_scale.try_alloc(chunk_rows) != nullptr;
        }
        if (ok) ok = ids_src1.try_alloc(chunk_rows) != nullptr;
        if (ok) ok = ids_dst.try_alloc(chunk_rows) != nullptr;
        if (ok) ok = expert_bounds.try_alloc(ne02 + 1) != nullptr;
        if (!ok) {
            reset_workspaces();
        }
        return ok;
    };

    bool workspaces_fit = try_workspaces();
    if (!workspaces_fit && !ctx.q8_1_cache.empty()) {
        ctx.q8_1_cache.clear();
        if (!ctx.q8_1_cache_pressure_logged) {
            GGML_LOG_WARN("CUDA q8_1 cache[%d]: required MMQ workspace did not fit; releasing cached activations\n",
                          ctx.device);
            ctx.q8_1_cache_pressure_logged = true;
        }
        workspaces_fit = try_workspaces();
    }
    while (!workspaces_fit) {
        if (chunk_ne12 == 1) {
            GGML_ABORT("CUDA MMQ ID: failed to allocate the minimum token workspace");
        }
        chunk_ne12 = (chunk_ne12 + 1) / 2;
        workspaces_fit = try_workspaces();
    }

    if (chunk_ne12 != ne12) {
        if (ctx.mmq_id_workspace_tokens_cap == 0 || chunk_ne12 < ctx.mmq_id_workspace_tokens_cap) {
            ctx.mmq_id_workspace_tokens_cap = chunk_ne12;
        }
        if (!ctx.mmq_id_workspace_tokens_cap_logged) {
            GGML_LOG_WARN("CUDA MMQ ID: device %d workspace pressure, splitting %lld tokens into chunks of %lld\n",
                          ctx.device, (long long) ne12, (long long) chunk_ne12);
            ctx.mmq_id_workspace_tokens_cap_logged = true;
        }
    }

    GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
    const int ids_s1 = ids->nb[1] / ggml_element_size(ids);
    const int src1_ids_s1 = nb12 / nb11;
    const int64_t src_s11 = src1->nb[1] / ts_src1;
    const int64_t src_s12 = src1->nb[2] / ts_src1;
    const int64_t src_s13 = src1->nb[3] / ts_src1;
    static_assert(QK_FP4_MMQ == 8 * QK_MXFP4, "QK_FP4_MMQ needs to be 8 * QK_MXFP4");
    const int64_t workspace_s12 = use_native_fp4 ?
        ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
        ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));

    for (int64_t token = 0; token < ne12; token += chunk_ne12) {
        const int64_t iter_ne12 = std::min(chunk_ne12, ne12 - token);
        const int64_t iter_rows = iter_ne12*n_expert_used;
        const int32_t * ids_chunk = (const int32_t *) ids->data + token*ids_s1;
        const float * src1_chunk = src1_d + token*src_s12;
        float * dst_chunk = dst_d + token*s2;

        ggml_cuda_launch_mm_ids_helper(ids_chunk, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, iter_ne12, n_expert_used, ne11, ids_s1, src1_ids_s1,
            /*write_inverse =*/ dedup_bcast, stream);
        CUDA_CHECK(cudaGetLastError());

        if (use_native_fp4) {
            static constexpr size_t align_float8 = 32;
            const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
            if (dedup_bcast) {
                quantize_scatter_mmq_fp4_cuda(src1_chunk, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr,
                    src0->type, use_aligned_float8, ne10, src_s12, ne10_padded, iter_ne12,
                    iter_rows, n_expert_used, stream);
            } else {
                quantize_mmq_fp4_cuda(src1_chunk, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr,
                    src0->type, use_aligned_float8, ne10, src_s11, src_s12, src_s13,
                    ne10_padded, iter_rows, 1, 1, stream);
            }
        } else if (dedup_bcast) {
            quantize_scatter_mmq_q8_1_cuda(src1_chunk, ids_src1.get(), src1_q8_1.get(), src0->type,
                ne10, src_s12, ne10_padded, iter_ne12, iter_rows, n_expert_used, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_chunk, ids_src1.get(), src1_q8_1.get(), src0->type,
                ne10, src_s11, src_s12, src_s13, ne10_padded, iter_rows, 1, 1, stream);
        }
        CUDA_CHECK(cudaGetLastError());

        const int64_t workspace_s13 = iter_ne12*workspace_s12;
        // ne02 is used instead of iter_ne12 because the number of y channels
        // determines the z dimension of the CUDA grid.
        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_chunk,
            src1_scale.ptr,
            ne00, ne01, iter_rows, s01, iter_rows, s1,
            ne02, ne02, s02, workspace_s12, s2,
            ne03, ne13, s03, workspace_s13, s3,
            iter_ne12};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
    }
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
// -------------------------------------------------
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
// -------------------------------------------------
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
// -------------------------------------------------
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    // MMQ tiles require at least 48 KiB per-block shared memory; fall back to BLAS otherwise.
    {
        const int    id    = ggml_cuda_get_device();
        const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
        if (smpbo < 48 * 1024) {
            return false;
        }
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    // gfx900 (Vega 10) lacks native dp4a, loses to dequant + hipBLAS
    // for dense matrices; keep MMQ only for MoE, where the
    // hipBLAS path is much slower.
    if (cc == GGML_CUDA_CC_VEGA) {
        return n_experts > 0;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
