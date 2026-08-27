// MoE entry point (mul_mat_id) for repacked Q8_0 weights: route tokens to their
// experts, then run the mat-vec (single token) or tiled GEMM (multi-token) path.
#include "repack.cuh"
#include "repack-common.cuh"
#include "repack-kernels.cuh"
#include "../quantize.cuh"
#include "../mmq.cuh"
#include "../mmid.cuh"

void ggml_cuda_mul_mat_id_repacked(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ids->type  == GGML_TYPE_I32);
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(ids->nb[0]  == sizeof(int32_t));
    GGML_ASSERT(src1->ne[3] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(src1->nb[2] == src1->nb[1] * src1->ne[1]);
    GGML_ASSERT(dst->nb[2]  == dst->nb[1]  * dst->ne[1]);
    GGML_ASSERT(dst->nb[1]  == (size_t) dst->ne[0] * sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 == ne00);
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = ids->ne[1];
    const int64_t n_assign      = n_expert_used * n_tokens;

    cudaStream_t stream = ctx.stream();

    const uint8_t * w;
    if (src0->view_src != nullptr && ggml_cuda_repack_tensor_supported(src0->view_src)) {
        w = repack_view_get_cached(src0, src0->view_src, stream);
    } else {
        w = (const uint8_t *) src0->data;
    }
    float * dst_d = (float *) dst->data;
    const size_t expert_stride = repack_gcn_nbytes(src0->type, ne00, ne01);
    const uint32_t dst_s1 = dst->nb[1] / sizeof(float);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_assign);
    ggml_cuda_pool_alloc<int32_t> ids_dst (ctx.pool(), n_assign);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);
    // The GEMM consumes ids_src1/ids_dst/expert_bounds, the mat-vec reads ids->data
    // directly. Both types now have a repack mat-vec, so the helper is multi-token
    // only again. (History: routing MXFP4 decode through the GEMM without widening
    // this condition handed repack_tile_off uninitialized pool memory as expert
    // bounds - if a type is ever GEMM-routed at one token, widen this with it.)
    if (n_tokens > 1) {
        const int si1  = ids->nb[1] / sizeof(int32_t);
        const int sis1 = src1->nb[2] / src1->nb[1];
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data,
            ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, n_tokens, n_expert_used, src1->ne[1], si1, sis1, false, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1;
    const int64_t n_cols      = src1->ne[1] * src1->ne[2];

    const int64_t s11 = src1->nb[1] / sizeof(float);

    // Narrow batch: one mat-vec per assignment in a single launch, instead of a
    // 32-wide tile per active expert. Weight loads go through rp_traits, so
    // every repacked type takes this path.
    if (n_tokens > 1 && n_tokens <= MMQ_RP_Q8_MOE_MMV_MAX_TOKENS) {
        ggml_cuda_pool_alloc<block_q8_1> src1_q8_1_own;
        block_q8_1 * src1_q8_1_d = repack_quantize_src1_q8_1(ctx, src0, src1, ne10,
            ne10_padded, s11, s11 * n_cols, s11 * n_cols, n_cols, 1, 1,
            (size_t) (n_cols * x_stride), src1_q8_1_own, stream);

        // Four rows per block, two waves. Five geometries were measured at
        // the verify widths and all landed within five percent; this was best.
        const dim3 grid((ne01 + 3) / 4, (unsigned) n_assign, 1);
        switch (src0->type) {
            case GGML_TYPE_Q8_0:
                mul_mat_vec_repacked_id1<4, 2, 2><<<grid, 128, 0, stream>>>(
                    w, src1_q8_1_d, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    ids_src1.get(), ids_dst.get(), expert_bounds.get(), (uint32_t) ne02,
                    expert_stride, (uint32_t) x_stride, dst_s1);
                break;
            case GGML_TYPE_MXFP4:
                mul_mat_vec_repacked_id1<4, 2, 2, GGML_TYPE_MXFP4><<<grid, 128, 0, stream>>>(
                    w, src1_q8_1_d, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    ids_src1.get(), ids_dst.get(), expert_bounds.get(), (uint32_t) ne02,
                    expert_stride, (uint32_t) x_stride, dst_s1);
                break;
            default: GGML_ABORT("unsupported repack type");
        }
        return;
    }

    if (n_tokens == 1) {
        ggml_cuda_pool_alloc<block_q8_1> src1_q8_1_own;
        const block_q8_1 * xq = repack_quantize_src1_q8_1(ctx, src0, src1, ne10,
            ne10_padded, s11, s11 * n_cols, s11 * n_cols, n_cols, 1, 1,
            (size_t) (n_cols * x_stride), src1_q8_1_own, stream);
        const uint32_t nchannels_y = (uint32_t) src1->ne[1];
        const uint32_t xs_id       = (uint32_t) x_stride;
        switch (src0->type) {
            case GGML_TYPE_Q8_0: {
                const dim3 grid((ne01 + 63) / 64, n_assign, 1);
                static const bool generic_mmv = getenv("GGML_RP_GENERIC_MMV") != nullptr;
                if (generic_mmv) {
                    mul_mat_vec_rp<GGML_TYPE_Q8_0, 64, 16, true, 16><<<grid, 1024, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                        (const int32_t *) ids->data, nchannels_y, expert_stride,
                        xs_id, dst_s1);
                    break;
                }
                mul_mat_vec_q8_0_repacked<64, 16, true, 16><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    (const int32_t *) ids->data, nullptr, nullptr,
                    (uint32_t) ne02, nchannels_y, expert_stride, xs_id, dst_s1,
                    nullptr, nullptr, nullptr, GGML_GLU_OP_REGLU);
            } break;
            case GGML_TYPE_MXFP4: {
                const dim3 grid((ne01 + 63) / 64, n_assign, 1);
                mul_mat_vec_rp<GGML_TYPE_MXFP4, 64, 16, true, 16><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    (const int32_t *) ids->data, nchannels_y, expert_stride,
                    xs_id, dst_s1);
            } break;
            default: GGML_ABORT("unsupported repack type");
        }
        return;
    }

    const uint64_t n_groups   = (uint64_t) ne10_padded / (4 * QK8_1);

    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(),
        (size_t) n_cols * n_groups * sizeof(block_q8_1_mmq_h));
    {
        quantize_mmq_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(),
            src0->type, ne10, s11, s11 * n_cols, s11 * n_cols, ne10_padded,
            n_cols, 1, 1, stream);
    }
    const block_q8_1 * xq = (const block_q8_1 *) src1_q8_1.get();

    // The tile token width is baked into repack_tile_off<BN>, so each path gets its
    // own prefix-sum + tile-meta buffers.
    constexpr int BN_ID   = 64 * MMQ_RP_Q8_TN;      // 64-wide tile
    constexpr int BN_W32  = 32 * 1;                 // 32-wide tile, TN==1
    // The 64-wide tile spills 5 VGPR (occupancy 2 against the 32-wide's 3), so it
    // only pays once every expert can fill it twice. Columns come per expert, not
    // per ubatch, which is why the crossover moves with the expert count.
    const bool use_w32 = n_assign < 2 * BN_ID * ne02;

    switch (src0->type) {
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4: {
            const bool is_mx = src0->type == GGML_TYPE_MXFP4;
            if (use_w32) {
                const int64_t max_tiles_w32 = n_assign / BN_W32 + ne02;
                ggml_cuda_pool_alloc<int32_t>          tile_off_w32 (ctx.pool(), ne02 + 1);
                ggml_cuda_pool_alloc<repack_tile_meta> tile_meta_w32(ctx.pool(), max_tiles_w32);
                repack_tile_off<BN_W32><<<1, 1, 0, stream>>>(expert_bounds.get(), tile_off_w32.get(), tile_meta_w32.get(), ne02);
                const dim3 grid((ne01 + MMQ_RP_Q8_BM - 1) / MMQ_RP_Q8_BM, max_tiles_w32, 1);
                if (is_mx) {
                mmq_gemm_repacked_w32<true, 1, MMQ_RP_Q8_NROW_LANES * 2, GGML_TYPE_MXFP4><<<grid, dim3(32, MMQ_RP_Q8_NROW_LANES * 2), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) n_cols,
                    ids_src1.get(), ids_dst.get(), expert_bounds.get(), tile_off_w32.get(), tile_meta_w32.get(),
                    (uint32_t) ne02, expert_stride, dst_s1);
                } else {
                mmq_gemm_repacked_w32<true, 1, MMQ_RP_Q8_NROW_LANES * 2, GGML_TYPE_Q8_0><<<grid, dim3(32, MMQ_RP_Q8_NROW_LANES * 2), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) n_cols,
                    ids_src1.get(), ids_dst.get(), expert_bounds.get(), tile_off_w32.get(), tile_meta_w32.get(),
                    (uint32_t) ne02, expert_stride, dst_s1);
                }
            } else {
                const int64_t max_tiles = n_assign / BN_ID + ne02;
                ggml_cuda_pool_alloc<int32_t>           tile_off (ctx.pool(), ne02 + 1);
                ggml_cuda_pool_alloc<repack_tile_meta>  tile_meta(ctx.pool(), max_tiles);
                repack_tile_off<BN_ID><<<1, 1, 0, stream>>>(expert_bounds.get(), tile_off.get(), tile_meta.get(), ne02);
                // Loose grid.y bound; the kernel's blockIdx.y >= tile_off[n_expert] guard skips slack.
                const dim3 grid((ne01 + MMQ_RP_Q8_BM - 1) / MMQ_RP_Q8_BM, max_tiles, 1);
                if (is_mx) {
                mmq_gemm_repacked<true, MMQ_RP_Q8_TN, MMQ_RP_Q8_NROW_LANES, GGML_TYPE_MXFP4><<<grid, dim3(64, MMQ_RP_Q8_NROW_LANES), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) n_cols,
                    ids_src1.get(), ids_dst.get(), expert_bounds.get(), tile_off.get(), tile_meta.get(),
                    (uint32_t) ne02, expert_stride, dst_s1);
                } else {
                mmq_gemm_repacked<true, MMQ_RP_Q8_TN, MMQ_RP_Q8_NROW_LANES, GGML_TYPE_Q8_0><<<grid, dim3(64, MMQ_RP_Q8_NROW_LANES), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) n_cols,
                    ids_src1.get(), ids_dst.get(), expert_bounds.get(), tile_off.get(), tile_meta.get(),
                    (uint32_t) ne02, expert_stride, dst_s1);
                }
            }
        } break;
        default: GGML_ABORT("unsupported repack type");
    }
}

// Fused MMV entry for repacked weights (MoE): single-token only, fuses the up
// and gate dot products (shared quantized input, shared expert lookup) plus
// optional biases and the GLU op.
// Dispatch: ggml_cuda_try_fuse `{op,op,GLU}` repack branch.
void ggml_cuda_mul_mat_id_vec_repacked_fused(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ids->type  == GGML_TYPE_I32);
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(ids->nb[0]  == sizeof(int32_t));
    GGML_ASSERT(dst->nb[1]  == (size_t) dst->ne[0] * sizeof(float));
    GGML_ASSERT(dst->ne[3] == 1);
    // Narrow batches take the per-assignment mat-vec; wider ones never reach here.
    GGML_ASSERT(ggml_cuda_repack_mmv_fusion_width_ok(ids->ne[1], true, src0->type));
    GGML_ASSERT(fusion != nullptr && fusion->gate != nullptr);
    GGML_ASSERT(fusion->gate->type == src0->type &&
                (src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_MXFP4));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 == ne00);
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = ids->ne[1];
    const int64_t n_assign      = n_expert_used * n_tokens;

    cudaStream_t stream = ctx.stream();

    const uint8_t * w;
    if (src0->view_src != nullptr && ggml_cuda_repack_tensor_supported(src0->view_src)) {
        w = repack_view_get_cached(src0, src0->view_src, stream);
    } else {
        w = (const uint8_t *) src0->data;
    }

    const ggml_tensor * gate = fusion->gate;
    const uint8_t * w_gate;
    if (gate->view_src != nullptr && ggml_cuda_repack_tensor_supported(gate->view_src)) {
        w_gate = repack_view_get_cached(gate, gate->view_src, stream);
    } else {
        w_gate = (const uint8_t *) gate->data;
    }

    // ADD_ID bias is [ne01, n_expert]; the kernel indexes it as expert*ne01 + row.
    const float * x_bias    = fusion->x_bias    ? (const float *) fusion->x_bias->data    : nullptr;
    const float * gate_bias = fusion->gate_bias ? (const float *) fusion->gate_bias->data : nullptr;

    const size_t   expert_stride = repack_gcn_nbytes(src0->type, ne00, ne01);
    const uint32_t dst_s1        = dst->nb[1] / sizeof(float);
    const int64_t  ne10_padded   = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t  x_stride      = ne10_padded / QK8_1;
    const int64_t  n_cols        = src1->ne[1] * src1->ne[2];
    const int64_t  s11           = src1->nb[1] / sizeof(float);

    ggml_cuda_pool_alloc<block_q8_1> src1_q8_1_own;
    block_q8_1 * src1_q8_1_d = repack_quantize_src1_q8_1(ctx, src0, src1, ne10,
        ne10_padded, s11, s11 * n_cols, s11 * n_cols, n_cols, 1, 1,
        (size_t) (n_cols * x_stride), src1_q8_1_own, stream);

    // Eight rows per block, four waves, 32 lanes to a row. Seven geometries
    // were traced at decode and lane width dominates: a 1024-thread workgroup
    // is 16 waves, 4 per SIMD, which caps registers at 64 per lane, and fusion
    // doubles the live weight state. Against up and gate as separate kernels,
    // MoE mat-vec time per token:
    //   <64,16,16> 1024thr  +3.5%      <16,16,64> 1024thr  -6.2%
    //   < 8, 2,16>  128thr -10.1%      < 8, 4,32>  256thr -15.9%
    if (n_tokens > 1) {
        // Narrow batch: one fused mat-vec per assignment in a single launch,
        // routed exactly as the unfused narrow path routes it.
        ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_assign);
        ggml_cuda_pool_alloc<int32_t> ids_dst (ctx.pool(), n_assign);
        ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);
        const int si1  = ids->nb[1] / sizeof(int32_t);
        const int sis1 = src1->nb[2] / src1->nb[1];
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data,
            ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, n_tokens, n_expert_used, src1->ne[1], si1, sis1, false, stream);
        CUDA_CHECK(cudaGetLastError());

        const dim3 grid_nc((ne01 + 3) / 4, (unsigned) n_assign, 1);
        // One row per lane, four waves. Two rows per lane costs a spill and
        // drops occupancy from 6 to 4 once the gate matrix shares the loop.
        switch (src0->type) {
            case GGML_TYPE_Q8_0:
                mul_mat_vec_repacked_id1_fused<4, 4, 1><<<grid_nc, 256, 0, stream>>>(
                    w, src1_q8_1_d, (float *) dst->data, (uint32_t) ne00, (uint32_t) ne01,
                    ids_src1.get(), ids_dst.get(), expert_bounds.get(), (uint32_t) ne02,
                    expert_stride, (uint32_t) x_stride, dst_s1,
                    w_gate, x_bias, gate_bias, fusion->glu_op);
                break;
            case GGML_TYPE_MXFP4:
                mul_mat_vec_repacked_id1_fused<4, 4, 1, GGML_TYPE_MXFP4><<<grid_nc, 256, 0, stream>>>(
                    w, src1_q8_1_d, (float *) dst->data, (uint32_t) ne00, (uint32_t) ne01,
                    ids_src1.get(), ids_dst.get(), expert_bounds.get(), (uint32_t) ne02,
                    expert_stride, (uint32_t) x_stride, dst_s1,
                    w_gate, x_bias, gate_bias, fusion->glu_op);
                break;
            default: GGML_ABORT("unsupported repack type");
        }
        return;
    }

    const dim3 grid((ne01 + 7) / 8, (unsigned) n_assign, 1);
    if (src0->type == GGML_TYPE_MXFP4) {
        mul_mat_vec_rp<GGML_TYPE_MXFP4, 8, 4, true, 32, true><<<grid, 256, 0, stream>>>(
            w, src1_q8_1_d, (float *) dst->data, (uint32_t) ne00, (uint32_t) ne01,
            (const int32_t *) ids->data, (uint32_t) src1->ne[1], expert_stride,
            (uint32_t) x_stride, dst_s1, w_gate, x_bias, gate_bias, fusion->glu_op);
        return;
    }
    mul_mat_vec_q8_0_repacked<8, 4, true, 32, true><<<grid, 256, 0, stream>>>(
        w, src1_q8_1_d, (float *) dst->data, (uint32_t) ne00, (uint32_t) ne01,
        (const int32_t *) ids->data, nullptr, nullptr,
        (uint32_t) ne02, (uint32_t) src1->ne[1], expert_stride,
        (uint32_t) x_stride, dst_s1,
        w_gate, x_bias, gate_bias, fusion->glu_op);
}
