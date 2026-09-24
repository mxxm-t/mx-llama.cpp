#include "llama-hparams.h"
#include "models.h"

#include "llama-kv-cache-dsv4.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

static float dsv4_rope_attn_factor(float freq_scale, float ext_factor) {
    if (ext_factor == 0.0f) {
        return 1.0f;
    }

    return 1.0f / (1.0f + 0.1f*logf(1.0f/freq_scale));
}

void llama_model_deepseek4::load_arch_hparams(llama_model_loader & ml) {
    if (hparams.n_layer_nextn > 0) {
        const uint32_t n_layer_main = hparams.n_layer_all - hparams.n_layer_nextn;
        const std::string mtp_probe = "blk." + std::to_string(n_layer_main) + ".nextn.eh_proj.weight";
        if (ml.get_weight(mtp_probe.c_str()) == nullptr) {
            hparams.n_layer_nextn = 0;
        }
    }

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,     hparams.swiglu_clamp_exp,   hparams.n_layer_all);
    if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP,   hparams.swiglu_clamp_shexp, hparams.n_layer_all, 0)) {
        hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
    }

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

    ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         hparams.dsv4_o_group_count);
    ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           hparams.dsv4_o_lora_rank);
    ml.get_key(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,    hparams.dsv4_compress_rope_base);
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
    ml.get_key(LLM_KV_HASH_LAYER_COUNT,                     hparams.dsv4_hash_layer_count);

    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    uint32_t n_compress_ratios = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_COMPRESS_RATIOS, n_compress_ratios);
    if (n_compress_ratios < hparams.n_layer_all) {
        throw std::runtime_error("DeepSeek-V4 compress_ratios is shorter than block_count");
    }
    GGML_ASSERT(n_compress_ratios <= LLAMA_MAX_LAYERS);
    ml.get_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios);

    ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func);
    if (hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
        throw std::runtime_error("DeepSeek-V4 loader currently expects sqrtsoftplus MoE scoring");
    }
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.set_swa_pattern(0);
    // tokens of an image span attend bidirectionally to the whole span, the window only applies to older tokens
    // ref: get_window_topk_idxs_visible in the reference impl
    hparams.non_causal_type = LLAMA_NON_CAUSAL_TYPE_SWA_FULL;
    for (uint32_t il = hparams.n_layer(); il < hparams.n_layer_all; ++il) {
        hparams.is_swa_impl[il] = true;
    }

    switch (hparams.n_layer()) {
        case 43: type = LLM_TYPE_UNKNOWN; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_deepseek4::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t o_groups    = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult     = hparams.dsv4_hc_mult;
    const int64_t hc_dim      = hc_mult * n_embd;
    const int64_t hc_mix_dim  = (2 + hc_mult) * hc_mult;

    const bool mtp_only = (n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    const int trunk_flags = mtp_only    ? TENSOR_NOT_REQUIRED : 0;
    const int mtp_flags   = ml.load_mtp ? 0 : TENSOR_SKIP;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN, "weight"),    {hc_dim, hc_mult}, head_hc_flags);
    hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE, "weight"),  {hc_mult}, head_hc_flags);
    hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1}, head_hc_flags);

    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];
        const int flags = i < n_layer ? trunk_flags : mtp_flags;

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, flags);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, flags);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, flags);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, flags);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, flags);
        layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, flags);
        layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, flags);
        // for wo_a, the shape in the file is (n_head * n_embd_head / o_groups, o_lora_rank*o_groups)
        // so we reshape here, to avoid reshaping the tensor in the graph
        layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank, o_groups}, flags | TENSOR_ALLOW_RESHAPE);
        layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, flags);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, flags);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, flags);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, flags);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, flags);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, flags);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, flags);

        const int64_t ratio = hparams.dsv4_compress_ratios[i];
        if (ratio != 0) {
            const int64_t coff = ratio == 4 ? 2 : 1;

            layer.attn_comp_wkv   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV,   "weight", i), {n_embd, coff * n_embd_head}, flags);
            layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", i), {n_embd, coff * n_embd_head}, flags);
            layer.attn_comp_ape   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_APE,   "weight", i), {coff * n_embd_head, ratio}, flags);
            layer.attn_comp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM,  "weight", i), {n_embd_head}, flags);

            if (ratio == 4) {
                const int64_t n_embd_indexer = hparams.indexer_head_size;

                layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, flags);
                layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * n_embd_indexer}, flags);

                layer.indexer_comp_wkv   = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_WKV,   "weight", i), {n_embd, 2 * n_embd_indexer}, flags);
                layer.indexer_comp_wgate = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_WGATE, "weight", i), {n_embd, 2 * n_embd_indexer}, flags);
                layer.indexer_comp_ape   = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_APE,   "weight", i), {2 * n_embd_indexer, ratio}, flags);
                layer.indexer_comp_norm  = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_NORM,  "weight", i), {n_embd_indexer}, flags);
            } else if (ratio != 128) {
                throw std::runtime_error("DeepSeek-V4 loader only supports compression ratios 0, 4, and 128");
            }
        }

        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, flags);
        if ((uint32_t) i < hparams.dsv4_hash_layer_count) {
            layer.ffn_gate_tid2eid = create_tensor(tn(LLM_TENSOR_FFN_GATE_TID2EID, "weight", i), {n_expert_used, n_vocab}, flags);
        } else {
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, flags);
        }
        // vision variant only: routing bias for image tokens
        layer.ffn_exp_probs_b_vl = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B_VL, "bias", i), {n_expert}, flags | TENSOR_NOT_REQUIRED);
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, flags);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, flags);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, flags);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    }, flags);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, flags);

        if (i >= n_layer) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2 * n_embd, n_embd}, flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd},             flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd},             flags);
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), {n_embd, n_vocab},    TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), {n_embd, n_vocab},    TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd},             TENSOR_NOT_REQUIRED | flags);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek4::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

static size_t dsv4_elem_offset(const ggml_tensor * t, int64_t i) {
    return ggml_row_size(t->type, i);
}

static ggml_tensor * dsv4_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, dsv4_elem_offset(t, i0));
}

static ggml_tensor * dsv4_view_2d(
        ggml_context * ctx,
        ggml_tensor  * t,
        int64_t        ne0,
        int64_t        ne1,
        int64_t        i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], dsv4_elem_offset(t, i0));
}

static ggml_tensor * dsv4_append_zero_row(ggml_context * ctx, ggml_tensor * t, bool neg_inf) {
    ggml_tensor * row = ggml_view_1d(ctx, t, t->ne[0], 0);
    row = neg_inf ? ggml_scale_bias(ctx, row, 0.0f, -INFINITY) : ggml_scale(ctx, row, 0.0f);
    row = ggml_reshape_2d(ctx, row, t->ne[0], 1);

    return ggml_concat(ctx, t, row, 1);
}

struct dsv4_state_tensors {
    ggml_tensor * kv;
    ggml_tensor * score;
};

static dsv4_state_tensors dsv4_build_state_restore(
        ggml_context * ctx,
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_dsv4_comp_state * state,
        int32_t il) {
    dsv4_state_tensors restored = {
        state->get_kv_all(ctx, il),
        state->get_score_all(ctx, il),
    };

    if (inp.state_restore_src_idxs == nullptr || inp.state_restore_dst_idxs == nullptr) {
        return restored;
    }

    ggml_tensor * kv_rows = ggml_get_rows(ctx, restored.kv, inp.state_restore_src_idxs);
    restored.kv = state->cpy_kv(ctx, kv_rows, inp.state_restore_dst_idxs, il);

    ggml_tensor * score_rows = ggml_get_rows(ctx, restored.score, inp.state_restore_src_idxs);
    restored.score = state->cpy_score(ctx, score_rows, inp.state_restore_dst_idxs, il);

    return restored;
}

static dsv4_state_tensors dsv4_build_state_snapshot(
        ggml_context * ctx,
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_dsv4_comp_state * state,
        ggml_tensor * source_kv,
        ggml_tensor * source_score,
        int32_t il) {
    if (inp.state_snapshot_src_idxs == nullptr || inp.state_snapshot_dst_idxs == nullptr ||
            source_kv == nullptr || source_score == nullptr) {
        return {};
    }

    ggml_tensor * kv_rows = ggml_get_rows(ctx, source_kv, inp.state_snapshot_src_idxs);
    ggml_tensor * kv = state->cpy_kv(ctx, kv_rows, inp.state_snapshot_dst_idxs, il);

    ggml_tensor * score_rows = ggml_get_rows(ctx, source_score, inp.state_snapshot_src_idxs);
    ggml_tensor * score = state->cpy_score(ctx, score_rows, inp.state_snapshot_dst_idxs, il);

    return { kv, score };
}


// mean over the hyper-connection streams: [n_embd, hc, n_tokens] -> [n_embd, n_tokens]
static ggml_tensor * dsv4_hc_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t hc = x->ne[1];

    ggml_tensor * acc = ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx, acc, ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
    }
    return ggml_scale(ctx, acc, 1.0f/hc);
}

static ggml_tensor * dsv4_hc_affine(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * scale,
        ggml_tensor  * base) {
    x = ggml_mul(ctx, x, scale);
    x = ggml_add(ctx, x, base);
    return x;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * weights,
        int           il) const {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[1] == hparams.dsv4_hc_mult);

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    if (cparams.fused_dsv4_hc_pre && il >= 0) {
        ggml_tensor * result = ggml_dsv4_hc_pre(ctx0, x, weights);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, result, il});
        return result;
    }

    ggml_tensor * result = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih*x->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx0, weights, 1, nt, weights->nb[1], ih*weights->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        result = result ? ggml_add(ctx0, result, cur) : cur;
    }

    return result;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_sinkhorn(
        ggml_tensor * comb,
        int           il) const {
    GGML_UNUSED(il);

    // comb is [dst_hc, src_hc, n_tokens]. Sinkhorn follows the reference:
    // row softmax over dst, one column normalization, then repeated row/column normalization.
    comb = ggml_soft_max(ctx0, comb);

    ggml_tensor * eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx0, eps, hparams.dsv4_hc_eps);

    comb = ggml_add(ctx0, comb, eps);

    auto norm_cols = [&]() {
        ggml_tensor * comb_src_dst = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
        ggml_tensor * col_sum = ggml_sum_rows(ctx0, comb_src_dst);
        col_sum = ggml_add(ctx0, col_sum, eps);
        col_sum = ggml_permute(ctx0, col_sum, 1, 0, 2, 3);
        comb = ggml_div(ctx0, comb, col_sum);
    };

    auto norm_rows = [&]() {
        ggml_tensor * row_sum = ggml_sum_rows(ctx0, comb);
        row_sum = ggml_add(ctx0, row_sum, eps);
        comb = ggml_div(ctx0, comb, row_sum);
    };

    norm_cols();
    for (uint32_t i = 1; i < hparams.dsv4_hc_sinkhorn_iters; ++i) {
        norm_rows();
        norm_cols();
    }

    return comb;
}

ggml_tensor * llama_model_deepseek4::graph::build_head_fold(const llama_model & model, ggml_tensor * x) const {
    if (model.hc_head_fn == nullptr) {
        // V4.1: fold with the mix the last layer computed from its pre-FFN input.
        // Recomputing one here from the post-FFN residual would not be the same value.
        GGML_ASSERT(last_ffn_pre != nullptr);
        return build_hc_pre(x, last_ffn_pre, -1);
    }

    return build_hc_head(x, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        ggml_tensor ** post,
        ggml_tensor ** comb,
        int il,
        ggml_tensor ** pre_out,
        ggml_tensor  * mix_in,
        bool           collapse) const {
    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc*n_embd;
    const int64_t hc_mix_dim = (2 + hc)*hc;
    const int64_t nt         = x->ne[2];

    GGML_ASSERT(hc == 4);
    GGML_ASSERT(hc_fn->ne[1] == hc_mix_dim);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_mixes", il);

    ggml_tensor * scale_pre  = dsv4_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = dsv4_view_1d(ctx0, hc_scale, 1, 1);

    ggml_tensor * base_pre  = dsv4_view_1d(ctx0, hc_base, hc, 0);
    ggml_tensor * base_post = dsv4_view_1d(ctx0, hc_base, hc, hc);

    ggml_tensor * pre = dsv4_view_2d(ctx0, mixes, hc, nt, 0);
    pre = dsv4_hc_affine(ctx0, pre, scale_pre, base_pre);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_pre", il);

    if (pre_out) {
        *pre_out = pre;
    }

    *post = dsv4_view_2d(ctx0, mixes, hc, nt, hc);
    *post = dsv4_hc_affine(ctx0, *post, scale_post, base_post);
    *post = ggml_sigmoid(ctx0, *post);
    *post = ggml_scale(ctx0, *post, 2.0f);
    cb(*post, "hc_post", il);

    if (cparams.fused_dsv4_hc_comb) {
        *comb = ggml_dsv4_hc_comb(ctx0, mixes, hc_scale, hc_base, hparams.dsv4_hc_eps,
                (int32_t) hparams.dsv4_hc_sinkhorn_iters);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_COMB, *comb, il});
    } else {
        ggml_tensor * scale_comb = dsv4_view_1d(ctx0, hc_scale, 1, 2);
        ggml_tensor * base_comb  = dsv4_view_1d(ctx0, hc_base, hc*hc, 2*hc);

        *comb = dsv4_view_2d(ctx0, mixes, hc*hc, nt, 2*hc);
        *comb = dsv4_hc_affine(ctx0, *comb, scale_comb, base_comb);
        *comb = ggml_reshape_3d(ctx0, *comb, hc, hc, nt);
        *comb = build_hc_sinkhorn(*comb, il);
    }
    cb(*comb, "hc_comb", il);

    // V4 collapses with the mix this sublayer just computed.
    // V4.1 hands its coefficients to the NEXT sublayer instead - attention uses the previous block's ffn mix, the FFN uses this block's attention mix - so the caller passes the carried one.
    // A caller that replaces the collapse with its own selection (V4.1 layer 0, which has no carried mix yet) must not get a mix built here.
    // A built but unused mix registers a fused node that no backend is ever assigned.
    // The fusion probe then reads that as missing support and disables the fused hc_pre for every layer.
    if (!collapse) {
        return nullptr;
    }
    ggml_tensor * result = build_hc_pre(x, mix_in ? mix_in : pre, il);
    return result;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_post(
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        ggml_tensor * comb,
        int il) const {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == hparams.dsv4_hc_mult);

    if (cparams.fused_dsv4_hc_post) {
        ggml_tensor * result = ggml_dsv4_hc_post(ctx0, x, residual, post, comb);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, result, il});
        return result;
    }

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[1];

    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * post_dst = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst*post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, x, post_dst);

        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * res_src = ggml_view_2d(ctx0, residual, n_embd, nt, residual->nb[2], src*residual->nb[1]);
            ggml_tensor * comb_src_dst = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2],
                    dst*comb->nb[0] + src*comb->nb[1]);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, comb_src_dst));
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_head(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base) const {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;
    const int64_t nt     = x->ne[2];

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_head_mixes", -1);

    ggml_tensor * pre = dsv4_hc_affine(ctx0, mixes, hc_scale, hc_base);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_head_pre", -1);

    return build_hc_pre(x, pre, -1);
}

ggml_tensor * llama_model_deepseek4::graph::build_hca_compressed_kv_from_state(
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t ratio,
        int64_t n_embd_head,
        const char * name,
        int il) const {
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_blocks         = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs);
    GGML_ASSERT(state_read_idxs->ne[0] == ratio*n_blocks);
    GGML_ASSERT(n_embd_head >= n_embd_head_rope);

    ggml_tensor * kv = ggml_get_rows(ctx0, kv_state, state_read_idxs);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, ratio, n_blocks);
    cb(kv, name, il);

    // at ratio 1 the softmax below runs over a single element and is the identity, which is why a ratio 1 source layer needs no gate at all and the file carries no attn_comp_wgate for it
    ggml_tensor * score = ggml_get_rows(ctx0, score_state, state_read_idxs);
    score = ggml_reshape_3d(ctx0, score, n_embd_head, ratio, n_blocks);
    cb(score, name, il);

    ggml_tensor * values = ggml_cont(ctx0, ggml_permute(ctx0, kv, 1, 0, 2, 3));
    ggml_tensor * scores = ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3));

    ggml_tensor * weights = ggml_soft_max(ctx0, scores);
    ggml_tensor * comp = ggml_mul(ctx0, values, weights);
    comp = ggml_sum_rows(ctx0, comp);
    comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
    cb(comp, name, il);

    comp = build_norm(comp, norm, nullptr, LLM_NORM_RMS, il);
    cb(comp, name, il);

    comp = ggml_rope_ext(ctx0, comp, comp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    comp = ggml_rope_set_offset(comp, n_embd_head_nope);
    cb(comp, name, il);

    return comp;
}

ggml_tensor * llama_model_deepseek4::graph::build_overlap_compressed_kv_from_state(
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t ratio,
        int64_t n_embd_head,
        const char * name,
        int il) const {
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_blocks         = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs);
    GGML_ASSERT(state_read_idxs->ne[0] == 2*ratio*n_blocks);
    GGML_ASSERT(kv_state->ne[0] == 2*n_embd_head);
    GGML_ASSERT(score_state->ne[0] == 2*n_embd_head);
    GGML_ASSERT(n_embd_head >= n_embd_head_rope);

    kv_state    = dsv4_append_zero_row(ctx0, kv_state,    false);
    score_state = dsv4_append_zero_row(ctx0, score_state, true);

    const int64_t n_read = ratio*n_blocks;

    ggml_tensor * kv_rows = ggml_get_rows(ctx0, kv_state, state_read_idxs);
    ggml_tensor * score_rows = ggml_get_rows(ctx0, score_state, state_read_idxs);

    ggml_tensor * kv_prev = ggml_cont(ctx0,
            ggml_view_2d(ctx0, kv_rows, n_embd_head, n_read, kv_rows->nb[1], 0));
    kv_prev = ggml_reshape_3d(ctx0, kv_prev, n_embd_head, ratio, n_blocks);
    cb(kv_prev, name, il);

    ggml_tensor * score_prev = ggml_cont(ctx0,
            ggml_view_2d(ctx0, score_rows, n_embd_head, n_read, score_rows->nb[1], 0));
    score_prev = ggml_reshape_3d(ctx0, score_prev, n_embd_head, ratio, n_blocks);
    cb(score_prev, name, il);

    ggml_tensor * kv_cur = ggml_cont(ctx0,
            ggml_view_2d(ctx0, kv_rows, n_embd_head, n_read, kv_rows->nb[1],
                n_read*kv_rows->nb[1] + ggml_row_size(kv_rows->type, n_embd_head)));
    kv_cur = ggml_reshape_3d(ctx0, kv_cur, n_embd_head, ratio, n_blocks);

    ggml_tensor * score_cur = ggml_cont(ctx0,
            ggml_view_2d(ctx0, score_rows, n_embd_head, n_read, score_rows->nb[1],
                n_read*score_rows->nb[1] + ggml_row_size(score_rows->type, n_embd_head)));
    score_cur = ggml_reshape_3d(ctx0, score_cur, n_embd_head, ratio, n_blocks);

    ggml_tensor * values = ggml_concat(ctx0, kv_prev, kv_cur, 1);
    ggml_tensor * scores = ggml_concat(ctx0, score_prev, score_cur, 1);

    values = ggml_cont(ctx0, ggml_permute(ctx0, values, 1, 0, 2, 3));
    scores = ggml_cont(ctx0, ggml_permute(ctx0, scores, 1, 0, 2, 3));

    ggml_tensor * weights = ggml_soft_max(ctx0, scores);
    ggml_tensor * comp = ggml_mul(ctx0, values, weights);
    comp = ggml_sum_rows(ctx0, comp);
    comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
    cb(comp, name, il);

    comp = build_norm(comp, norm, nullptr, LLM_NORM_RMS, il);
    cb(comp, name, il);

    comp = ggml_rope_ext(ctx0, comp, comp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    comp = ggml_rope_set_offset(comp, n_embd_head_nope);
    cb(comp, name, il);

    return comp;
}

ggml_tensor * llama_model_deepseek4::graph::build_lid_top_k(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];
    const auto & inp_lid = inp_dsv4->get_lid();
    const int64_t n_embd_indexer_head      = hparams.indexer_head_size;
    const int64_t n_embd_indexer_head_rope = hparams.n_rot();
    const int64_t n_embd_indexer_head_nope = n_embd_indexer_head - n_embd_indexer_head_rope;
    const int64_t n_indexer_head           = hparams.indexer_n_head;
    const int64_t nt                       = cur->ne[1];

    GGML_ASSERT(inp_lid.kq_mask);
    GGML_ASSERT(inp_lid.k_rot);
    GGML_ASSERT(n_embd_indexer_head >= n_embd_indexer_head_rope);

    ggml_tensor * indexer_q = build_lora_mm(layer.indexer_attn_q_b, qr);
    indexer_q = ggml_reshape_3d(ctx0, indexer_q, n_embd_indexer_head, n_indexer_head, nt);
    cb(indexer_q, "lid_q", il);

    indexer_q = ggml_rope_ext(ctx0, indexer_q, inp_pos, nullptr, n_embd_indexer_head_rope,
            rope_type, n_ctx_orig, hparams.dsv4_compress_rope_base, freq_scale,
            ext_factor, dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    indexer_q = ggml_rope_set_offset(indexer_q, n_embd_indexer_head_nope);
    cb(indexer_q, "lid_q_rope", il);

    indexer_q = llama_mul_mat_hadamard(ctx0, indexer_q, inp_lid.k_rot);
    cb(indexer_q, "lid_q_rot", il);

    ggml_tensor * indexer_weights = build_lora_mm(layer.indexer_proj, cur);
    indexer_weights = ggml_scale(ctx0, indexer_weights, 1.0f/sqrtf(float(n_embd_indexer_head*n_indexer_head)));
    cb(indexer_weights, "lid_weights", il);

    ggml_tensor * indexer_k = inp_dsv4->mctx->get_lid()->get_k(ctx0, il);
    const int64_t n_lid = inp_lid.kq_mask->ne[0];
    GGML_ASSERT(n_lid > 0);
    GGML_ASSERT(n_lid <= indexer_k->ne[2]);

    indexer_k = ggml_view_4d(ctx0, indexer_k,
            indexer_k->ne[0], indexer_k->ne[1], n_lid, indexer_k->ne[3],
            indexer_k->nb[1], indexer_k->nb[2], indexer_k->nb[3], 0);
    cb(indexer_k, "lid_k", il);

    const int64_t n_stream = indexer_k->ne[3];
    indexer_q = ggml_view_4d(ctx0, indexer_q,
            indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2]/n_stream, n_stream,
            indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3]/n_stream, 0);
    indexer_weights = ggml_view_4d(ctx0, indexer_weights,
            indexer_weights->ne[0], indexer_weights->ne[1]/n_stream, indexer_weights->ne[2], n_stream,
            indexer_weights->nb[1], indexer_weights->nb[2]/n_stream, indexer_weights->nb[3]/n_stream, 0);

    ggml_tensor * indexer_score = nullptr;
    if (cparams.fused_lid) {
        indexer_score = ggml_lightning_indexer(ctx0, indexer_q, indexer_k, indexer_weights, inp_lid.kq_mask);
        cb(indexer_score, "lid_score_masked", il);
        res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, indexer_score, il});
    } else {
        indexer_q = ggml_permute(ctx0, indexer_q, 0, 2, 1, 3);
        cb(indexer_q, "lid_q", il);
        indexer_k = ggml_permute(ctx0, indexer_k, 0, 2, 1, 3);
        cb(indexer_k, "lid_k", il);

        ggml_tensor * indexer_kq = ggml_mul_mat(ctx0, indexer_k, indexer_q);
        cb(indexer_kq, "lid_kq", il);

        indexer_kq = ggml_cont(ctx0, ggml_permute(ctx0, indexer_kq, 2, 1, 0, 3));
        cb(indexer_kq, "lid_kq", il);

        indexer_score = ggml_relu(ctx0, indexer_kq);
        indexer_score = ggml_mul(ctx0, indexer_score, indexer_weights);
        indexer_score = ggml_sum_rows(ctx0, indexer_score);
        indexer_score = ggml_cont(ctx0, ggml_permute(ctx0, indexer_score, 2, 1, 0, 3));
        cb(indexer_score, "lid_score", il);

        indexer_score = ggml_add(ctx0, indexer_score, inp_lid.kq_mask);
        cb(indexer_score, "lid_score_masked", il);
    }

    const uint32_t n_top_k = indexer_score->ne[0] < hparams.indexer_top_k ? indexer_score->ne[0] : hparams.indexer_top_k;
    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, indexer_score, n_top_k));
    cb(top_k, "lid_top_k", il);

    return top_k;
}

ggml_tensor * llama_model_deepseek4::graph::build_top_k_mask(
        ggml_tensor * kq_mask,
        ggml_tensor * top_k,
        const char * name,
        int il) const {
    GGML_ASSERT(kq_mask);
    GGML_ASSERT(top_k);

    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3],
            kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1,
            top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k,
            kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3],
            kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);
    cb(kq_mask_top_k, name, il);

    return kq_mask_top_k;
}

ggml_tensor * llama_model_deepseek4::graph::build_csa_lid_attention(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        llm_graph_input_dsv4_raw * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    const auto & inp_csa = inp_dsv4->get_csa();
    GGML_ASSERT(inp_csa.kq_mask);

    ggml_tensor * top_k = build_lid_top_k(model, inp_dsv4, qr, cur, inp_pos, il);

    ggml_tensor * k_rot = inp_attn->self_k_rot;
    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);
    cb(raw_k, "csa_raw_k", il);

    ggml_tensor * csa_k = inp_dsv4->mctx->get_csa()->get_k(ctx0, il);
    const int64_t n_csa = inp_csa.kq_mask->ne[0];
    GGML_ASSERT(n_csa > 0);
    GGML_ASSERT(n_csa <= csa_k->ne[2]);

    csa_k = ggml_view_4d(ctx0, csa_k,
            csa_k->ne[0], csa_k->ne[1], n_csa, csa_k->ne[3],
            csa_k->nb[1], csa_k->nb[2], csa_k->nb[3], 0);
    cb(csa_k, "csa_comp_k", il);

    ggml_tensor * k_all = ggml_concat(ctx0, raw_k, csa_k, 2);
    cb(k_all, "csa_k_all", il);

    ggml_tensor * raw_mask = inp_attn->get_kq_mask();
    ggml_tensor * csa_mask = build_top_k_mask(inp_csa.kq_mask, top_k, "csa_top_k_mask", il);

    ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, csa_mask, 0);
    cb(kq_mask, "csa_lid_kq_mask", il);

    const int64_t n_kv_max = std::min<int64_t>(raw_mask->ne[0], hparams.n_swa) + top_k->ne[0];
    ggml_tensor * out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, sinks, nullptr, n_kv_max, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_csa_lid", il);

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_hca_attention(
        llm_graph_input_dsv4 * inp_dsv4,
        llm_graph_input_dsv4_raw * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * sinks,
        float kq_scale,
        int il,
        bool idx_tier) const {
    // V4.1 layers after a source reuse the source's compressed cache, which the cache resolves from il.
    const auto & inp_hca = idx_tier ? inp_dsv4->get_csa() : inp_dsv4->get_hca();
    GGML_ASSERT(inp_hca.kq_mask);

    ggml_tensor * k_rot = inp_attn->self_k_rot;
    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);
    cb(raw_k, "hca_raw_k", il);

    ggml_tensor * hca_k = build_get_k(idx_tier ? inp_dsv4->mctx->get_csa() : inp_dsv4->mctx->get_hca(), il);
    const int64_t n_hca = inp_hca.kq_mask->ne[0];
    GGML_ASSERT(n_hca > 0);
    GGML_ASSERT(n_hca <= hca_k->ne[2]);

    hca_k = ggml_view_4d(ctx0, hca_k,
            hca_k->ne[0], hca_k->ne[1], n_hca, hca_k->ne[3],
            hca_k->nb[1], hca_k->nb[2], hca_k->nb[3], 0);
    cb(hca_k, "hca_comp_k", il);

    ggml_tensor * k_all = ggml_concat(ctx0, raw_k, hca_k, 2);
    cb(k_all, "hca_k_all", il);

    ggml_tensor * raw_mask = inp_attn->get_kq_mask();
    ggml_tensor * hca_mask = inp_hca.kq_mask;

    ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, hca_mask, 0);
    cb(kq_mask, "hca_kq_mask", il);

    ggml_tensor * out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, sinks, nullptr, 0, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_hca", il);

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_raw_attention(
        llm_graph_input_dsv4_raw * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    GGML_ASSERT(hparams.is_swa(il));

    ggml_tensor * k_rot = inp_attn->self_k_rot;

    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_cur = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * kq_mask = inp_attn->get_kq_mask();

    ggml_tensor * k = mctx_cur->get_k(ctx0, il);

    ggml_tensor * out = build_attn_mha(q, k, k, nullptr, kq_mask, sinks, nullptr, 0, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_raw", il);

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_attention(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    return build_attention_impl(model, inp_dsv4, nullptr, cur, inp_pos, il);
}

ggml_tensor * llama_model_deepseek4::graph::build_attention(
        const llama_model & model,
        llm_graph_input_attn_k_iswa * inp_mtp,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    return build_attention_impl(model, nullptr, inp_mtp, cur, inp_pos, il);
}

ggml_tensor * llama_model_deepseek4::graph::build_attention_impl(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        llm_graph_input_attn_k_iswa * inp_mtp,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    GGML_ASSERT((inp_dsv4 == nullptr) != (inp_mtp == nullptr));

    const auto & layer = model.layers[il];
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4 ? inp_dsv4->get_raw() : nullptr;

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t n_heads_group    = n_head / n_groups;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim      = n_heads_group*n_embd_head;
    const int64_t nt               = cur->ne[1];

    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % n_groups == 0);

    const bool use_compress_rope = hparams.dsv4_compress_ratios[il] != 0;
    const float freq_base_l      = use_compress_rope ? hparams.dsv4_compress_rope_base : freq_base;
    const float freq_scale_l     = use_compress_rope ? freq_scale : 1.0f;
    const float ext_factor_l     = use_compress_rope ? ext_factor : 0.0f;
    const float attn_factor_l    = dsv4_rope_attn_factor(freq_scale_l, ext_factor_l);
    const float beta_fast_l      = use_compress_rope ? beta_fast : 0.0f;
    const float beta_slow_l      = use_compress_rope ? beta_slow : 0.0f;
    const int32_t n_ctx_orig_l   = use_compress_rope ? n_ctx_orig : 0;

    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    cb(qr, "qr", il);

    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "qr_norm", il);

    ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, nt);
    // V4 renormalizes q per head after wq_b.
    // V4.1 does not: its only query norm is the one on wq_a's output, and this second one rescales every head's attention scores.
    // LLAMA_DSV41_QNORM=1 puts it back for A/B.
    static const bool q_norm_off = [] {
        const char * e = getenv("LLAMA_DSV41_QNORM");
        return !(e != nullptr && atoi(e) != 0);
    }();
    const bool dspark_v41 = inp_mtp && model.arch == LLM_ARCH_DFLASH && model.dspark_markov_w1 && model.hc_head_fn == nullptr;
    if (!(hparams.dsv41_n_kv_source > 0 && q_norm_off) && !dspark_v41) {
        q = ggml_rms_norm(ctx0, q, norm_rms_eps);
    }
    cb(q, "q_norm", il);

    q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    q = ggml_rope_set_offset(q, n_embd_head_nope);
    cb(q, "q", il);

    ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
    kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);
    cb(kv, "kv_norm", il);

    kv = ggml_rope_ext(ctx0, kv, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    kv = ggml_rope_set_offset(kv, n_embd_head_nope);
    cb(kv, "kv", il);

    const int64_t ratio = hparams.dsv4_compress_ratios[il];
    GGML_ASSERT(inp_dsv4 || ratio == 0);

    // Which compressed tier this layer belongs to, and who owns its KV.
    // V4 compresses on every ratio layer, so each is its own source and both tiers keep their V4 shapes.
    // V4.1 pools non-overlapping groups on both tiers and writes only on a source layer, which the layers after it read.
    const bool v41        = hparams.dsv41_n_kv_source > 0;
    const bool tier_idx   = ratio != 0 && (uint32_t) ratio == hparams.dsv4_ratio_idx;
    const bool tier_plain = ratio != 0 && (uint32_t) ratio == hparams.dsv4_ratio_plain;
    const int  kv_src     = (v41 && ratio != 0) ? hparams.dsv41_kv_source_for(il) : il;
    const bool owns_kv    = ratio != 0 && kv_src == il;

    GGML_ASSERT(!(v41 && ratio != 0) || kv_src >= 0);

    // LLAMA_DSV41_NO_COMPRESS=1 drops the compressed tier and leaves every layer on its 128-token sliding window, to tell a broken compressed path from a broken stack.
    // RoPE is left alone so only one thing changes.
    static const bool compress_off = [] {
        const char * e = getenv("LLAMA_DSV41_NO_COMPRESS");
        const bool off = e != nullptr && atoi(e) != 0;
        if (off) {
            LLAMA_LOG_WARN("deepseek41: LLAMA_DSV41_NO_COMPRESS=1, compressed tier disabled\n");
        }
        return off;
    }();

    // the pooled compressor serves V4's plain tier, and both of V4.1's
    const bool pooled   = (v41 ? owns_kv : tier_plain) && !(v41 && compress_off);
    const llm_graph_input_dsv4::comp_input no_tier = {};
    const auto & tier   = inp_dsv4 ? (v41 && tier_idx ? inp_dsv4->get_csa() : inp_dsv4->get_hca())
                                   : no_tier;
    const int64_t tier_ratio = ratio;

    ggml_tensor * hca_state_kv    = nullptr;
    ggml_tensor * hca_state_score = nullptr;
    ggml_tensor * hca_source_kv   = nullptr;
    ggml_tensor * hca_source_score = nullptr;
    if (pooled && tier.state_pos) {
        hca_state_kv = build_lora_mm(layer.attn_comp_wkv, cur);
        cb(hca_state_kv, "comp_state_kv", il);

        // A ratio 1 tier pools one token, so its softmax is the identity and any score gives weight 1.
        // V4.1's ratio 1 source layer ships no gate at all, hence the zeros.
        hca_state_score = layer.attn_comp_wgate
            ? build_lora_mm(layer.attn_comp_wgate, cur)
            : ggml_scale(ctx0, hca_state_kv, 0.0f);
        cb(hca_state_score, "comp_state_score", il);

        // V4 adds a learned position embedding to the gate. V4.1 ships none.
        if (layer.attn_comp_ape) {
            ggml_tensor * ape_rows = ggml_get_rows(ctx0, layer.attn_comp_ape, tier.state_pos);
            hca_state_score = ggml_add(ctx0, hca_state_score, ape_rows);
            cb(hca_state_score, "comp_state_score_ape", il);
        }
    }

    if (!v41 && tier_idx && inp_dsv4->get_csa().state_pos) {
        ggml_tensor * csa_state_kv = build_lora_mm(layer.attn_comp_wkv, cur);
        cb(csa_state_kv, "csa_state_kv", il);

        ggml_tensor * csa_state_score = build_lora_mm(layer.attn_comp_wgate, cur);
        cb(csa_state_score, "csa_state_score", il);

        ggml_tensor * csa_ape = layer.attn_comp_ape;

        ggml_tensor * csa_ape_rows = ggml_get_rows(ctx0, csa_ape, inp_dsv4->get_csa().state_pos);
        csa_state_score = ggml_add(ctx0, csa_state_score, csa_ape_rows);
        cb(csa_state_score, "csa_state_score_ape", il);

        GGML_ASSERT(inp_dsv4->get_csa().state_write_idxs);

        const auto * csa_state = inp_dsv4->mctx->get_csa_state();
        const dsv4_state_tensors csa_restored = dsv4_build_state_restore(
                ctx0, inp_dsv4->get_csa(), csa_state, il);
        ggml_tensor * csa_base_kv = dsv4_view_2d(
                ctx0, csa_restored.kv, csa_restored.kv->ne[0], csa_state->get_n_rows(), 0);
        ggml_tensor * csa_base_score = dsv4_view_2d(
                ctx0, csa_restored.score, csa_restored.score->ne[0], csa_state->get_n_rows(), 0);

        ggml_tensor * csa_source_kv = ggml_concat(ctx0, csa_base_kv, csa_state_kv, 1);
        ggml_tensor * csa_source_score = ggml_concat(ctx0, csa_base_score, csa_state_score, 1);

        ggml_tensor * kv_comp_csa_state = build_overlap_compressed_kv_from_state(
                csa_source_kv,
                csa_source_score,
                inp_dsv4->get_csa().state_read_idxs,
                inp_dsv4->get_csa().state_write_pos,
                layer.attn_comp_norm,
                hparams.dsv4_ratio_idx,
                n_embd_head,
                "csa_state_compress",
                il);

        if (inp_dsv4->get_csa().k_rot) {
            kv_comp_csa_state = llama_mul_mat_hadamard(ctx0, kv_comp_csa_state, inp_dsv4->get_csa().k_rot);
            cb(kv_comp_csa_state, "csa_state_compress_rot", il);
        }

        ggml_build_forward_expand(gf, inp_dsv4->mctx->get_csa()->cpy_k(ctx0,
                    kv_comp_csa_state, inp_dsv4->get_csa().state_write_idxs, il));

        ggml_tensor * csa_snapshot_source_kv = ggml_concat(ctx0,
                csa_restored.kv, csa_state_kv, 1);
        ggml_tensor * csa_snapshot_source_score = ggml_concat(ctx0,
                csa_restored.score, csa_state_score, 1);

        const dsv4_state_tensors csa_snapshot = dsv4_build_state_snapshot(
                ctx0, inp_dsv4->get_csa(), csa_state, csa_snapshot_source_kv, csa_snapshot_source_score, il);
        if (csa_snapshot.kv != nullptr) {
            ggml_build_forward_expand(gf, csa_snapshot.kv);
        }
        if (csa_snapshot.score != nullptr) {
            ggml_build_forward_expand(gf, csa_snapshot.score);
        }

        ggml_tensor * csa_persist_kv = ggml_get_rows(ctx0, csa_state_kv, inp_dsv4->get_csa().state_persist_src_idxs);
        ggml_tensor * csa_persist_score = ggml_get_rows(ctx0, csa_state_score, inp_dsv4->get_csa().state_persist_src_idxs);

        csa_state_kv = inp_dsv4->mctx->get_csa_state()->cpy_kv(ctx0,
                csa_persist_kv, inp_dsv4->get_csa().state_persist_dst_idxs, il);
        csa_state_score = inp_dsv4->mctx->get_csa_state()->cpy_score(ctx0,
                csa_persist_score, inp_dsv4->get_csa().state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, csa_state_kv);
        ggml_build_forward_expand(gf, csa_state_score);

        ggml_tensor * lid_state_kv = build_lora_mm(layer.indexer_comp_wkv, cur);
        cb(lid_state_kv, "lid_state_kv", il);

        ggml_tensor * lid_state_score = build_lora_mm(layer.indexer_comp_wgate, cur);
        cb(lid_state_score, "lid_state_score", il);

        ggml_tensor * lid_ape = layer.indexer_comp_ape;

        ggml_tensor * lid_ape_rows = ggml_get_rows(ctx0, lid_ape, inp_dsv4->get_lid().state_pos);
        lid_state_score = ggml_add(ctx0, lid_state_score, lid_ape_rows);
        cb(lid_state_score, "lid_state_score_ape", il);

        GGML_ASSERT(inp_dsv4->get_lid().state_write_idxs);

        const auto * lid_state = inp_dsv4->mctx->get_lid_state();
        const dsv4_state_tensors lid_restored = dsv4_build_state_restore(
                ctx0, inp_dsv4->get_lid(), lid_state, il);
        ggml_tensor * lid_base_kv = dsv4_view_2d(
                ctx0, lid_restored.kv, lid_restored.kv->ne[0], lid_state->get_n_rows(), 0);
        ggml_tensor * lid_base_score = dsv4_view_2d(
                ctx0, lid_restored.score, lid_restored.score->ne[0], lid_state->get_n_rows(), 0);

        ggml_tensor * lid_source_kv = ggml_concat(ctx0, lid_base_kv, lid_state_kv, 1);
        ggml_tensor * lid_source_score = ggml_concat(ctx0, lid_base_score, lid_state_score, 1);

        ggml_tensor * kv_comp_lid_state = build_overlap_compressed_kv_from_state(
                lid_source_kv,
                lid_source_score,
                inp_dsv4->get_lid().state_read_idxs,
                inp_dsv4->get_lid().state_write_pos,
                layer.indexer_comp_norm,
                hparams.dsv4_ratio_idx,
                hparams.indexer_head_size,
                "lid_state_compress",
                il);

        if (inp_dsv4->get_lid().k_rot) {
            kv_comp_lid_state = llama_mul_mat_hadamard(ctx0, kv_comp_lid_state, inp_dsv4->get_lid().k_rot);
            cb(kv_comp_lid_state, "lid_state_compress_rot", il);
        }

        ggml_build_forward_expand(gf, inp_dsv4->mctx->get_lid()->cpy_k(ctx0,
                    kv_comp_lid_state, inp_dsv4->get_lid().state_write_idxs, il));

        ggml_tensor * lid_snapshot_source_kv = ggml_concat(ctx0,
                lid_restored.kv, lid_state_kv, 1);
        ggml_tensor * lid_snapshot_source_score = ggml_concat(ctx0,
                lid_restored.score, lid_state_score, 1);

        const dsv4_state_tensors lid_snapshot = dsv4_build_state_snapshot(
                ctx0, inp_dsv4->get_lid(), lid_state, lid_snapshot_source_kv, lid_snapshot_source_score, il);
        if (lid_snapshot.kv != nullptr) {
            ggml_build_forward_expand(gf, lid_snapshot.kv);
        }
        if (lid_snapshot.score != nullptr) {
            ggml_build_forward_expand(gf, lid_snapshot.score);
        }

        ggml_tensor * lid_persist_kv = ggml_get_rows(ctx0, lid_state_kv, inp_dsv4->get_lid().state_persist_src_idxs);
        ggml_tensor * lid_persist_score = ggml_get_rows(ctx0, lid_state_score, inp_dsv4->get_lid().state_persist_src_idxs);

        lid_state_kv = inp_dsv4->mctx->get_lid_state()->cpy_kv(ctx0,
                lid_persist_kv, inp_dsv4->get_lid().state_persist_dst_idxs, il);
        lid_state_score = inp_dsv4->mctx->get_lid_state()->cpy_score(ctx0,
                lid_persist_score, inp_dsv4->get_lid().state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, lid_state_kv);
        ggml_build_forward_expand(gf, lid_state_score);
    }

    const llama_dsv4_comp_state * hca_state = nullptr;
    dsv4_state_tensors hca_restored = {};
    if (pooled && tier.state_write_idxs) {
        GGML_ASSERT(hca_state_kv);
        GGML_ASSERT(hca_state_score);

        hca_state = (v41 && tier_idx) ? inp_dsv4->mctx->get_csa_state() : inp_dsv4->mctx->get_hca_state();
        hca_restored = dsv4_build_state_restore(ctx0, tier, hca_state, il);
        ggml_tensor * hca_base_kv = dsv4_view_2d(
                ctx0, hca_restored.kv, hca_restored.kv->ne[0], hca_state->get_n_rows(), 0);
        ggml_tensor * hca_base_score = dsv4_view_2d(
                ctx0, hca_restored.score, hca_restored.score->ne[0], hca_state->get_n_rows(), 0);

        hca_source_kv = ggml_concat(ctx0, hca_base_kv, hca_state_kv, 1);
        hca_source_score = ggml_concat(ctx0, hca_base_score, hca_state_score, 1);

        ggml_tensor * kv_comp_hca = build_hca_compressed_kv_from_state(
                hca_source_kv,
                hca_source_score,
                tier.state_read_idxs,
                tier.state_write_pos,
                layer.attn_comp_norm,
                tier_ratio,
                n_embd_head,
                "comp_state_compress",
                il);

        if (tier.k_rot) {
            kv_comp_hca = llama_mul_mat_hadamard(ctx0, kv_comp_hca, tier.k_rot);
            cb(kv_comp_hca, "comp_state_compress_rot", il);
        }

        const auto * tier_kv = (v41 && tier_idx) ? inp_dsv4->mctx->get_csa() : inp_dsv4->mctx->get_hca();
        build_cpy_k(tier_kv, kv_comp_hca, tier.state_write_idxs, il);
    }

    if (pooled && tier.state_pos) {
        GGML_ASSERT(hca_state_kv);
        GGML_ASSERT(hca_state_score);

        if (hca_state == nullptr) {
            hca_state = (v41 && tier_idx) ? inp_dsv4->mctx->get_csa_state() : inp_dsv4->mctx->get_hca_state();
        }
        if (hca_restored.kv == nullptr) {
            hca_restored = dsv4_build_state_restore(ctx0, tier, hca_state, il);
        }
        if (hca_source_kv == nullptr || hca_source_score == nullptr) {
            ggml_tensor * hca_base_kv = dsv4_view_2d(
                    ctx0, hca_restored.kv, hca_restored.kv->ne[0], hca_state->get_n_rows(), 0);
            ggml_tensor * hca_base_score = dsv4_view_2d(
                    ctx0, hca_restored.score, hca_restored.score->ne[0], hca_state->get_n_rows(), 0);

            hca_source_kv = ggml_concat(ctx0, hca_base_kv, hca_state_kv, 1);
            hca_source_score = ggml_concat(ctx0, hca_base_score, hca_state_score, 1);
        }

        ggml_tensor * hca_snapshot_source_kv = ggml_concat(ctx0,
                hca_restored.kv, hca_state_kv, 1);
        ggml_tensor * hca_snapshot_source_score = ggml_concat(ctx0,
                hca_restored.score, hca_state_score, 1);

        const dsv4_state_tensors hca_snapshot = dsv4_build_state_snapshot(
                ctx0, tier, hca_state, hca_snapshot_source_kv, hca_snapshot_source_score, il);
        if (hca_snapshot.kv != nullptr) {
            ggml_build_forward_expand(gf, hca_snapshot.kv);
        }
        if (hca_snapshot.score != nullptr) {
            ggml_build_forward_expand(gf, hca_snapshot.score);
        }

        ggml_tensor * hca_persist_kv = ggml_get_rows(ctx0, hca_state_kv, tier.state_persist_src_idxs);
        ggml_tensor * hca_persist_score = ggml_get_rows(ctx0, hca_state_score, tier.state_persist_src_idxs);

        hca_state_kv = hca_state->cpy_kv(ctx0,
                hca_persist_kv, tier.state_persist_dst_idxs, il);
        hca_state_score = hca_state->cpy_score(ctx0,
                hca_persist_score, tier.state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, hca_state_kv);
        ggml_build_forward_expand(gf, hca_state_score);
    }

    ggml_tensor * out = nullptr;
    if (inp_mtp) {
        out = build_attn(inp_mtp,
                nullptr, nullptr, nullptr,
                q, kv, kv,
                nullptr, layer.attn_sinks, nullptr,
                1.0f/sqrtf(float(n_embd_head)), il);
        cb(out, "attn_raw", il);
    } else if (!v41 && tier_idx &&
            inp_dsv4->get_csa().kq_mask &&
            inp_dsv4->get_lid().kq_mask &&
            inp_dsv4->get_lid().k_rot) {
        out = build_csa_lid_attention(model, inp_dsv4, inp_attn, q, kv, qr, cur, inp_pos, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il);
    } else if (ratio != 0 && tier.kq_mask && !(v41 && compress_off)) {
        // the compressed rows live on the source layer, which for V4 is this layer itself
        out = build_hca_attention(inp_dsv4, inp_attn, q, kv, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il, v41 && tier_idx);
    } else {
        out = build_raw_attention(inp_attn, q, kv, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il);
    }

    out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nt);
    out = ggml_rope_ext_back(ctx0, out, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    out = ggml_rope_set_offset(out, n_embd_head_nope);
    cb(out, "attn_derope", il);

    out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, nt);
    out = ggml_permute(ctx0, out, 0, 2, 1, 3);
    ggml_tensor * oa = ggml_mul_mat(ctx0, layer.wo_a, out);
    cb(oa, "attn_wo_a", il);
    oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, nt);

    out = build_lora_mm(layer.wo_b, oa);
    cb(out, "attn_out", il);

    return out;
}

llama_model_deepseek4::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();
    ggml_build_forward_expand(gf, inp_attn->self_kq_mask);

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    // V4.1 shifts the hyper-connection coefficients by one sublayer, so the mix a sublayer computes is carried to the next one.
    // The run starts from a one-hot mix on copy 0, and collapsing with a one-hot is just selecting that copy, so no constant tensor is needed.
    const bool hc_shift = model.hc_head_fn == nullptr;
    ggml_tensor * carry_pre = nullptr;

    // LLAMA_DSV41_NO_ENGRAM=1 skips the engram contribution, to tell a broken gate from a broken stack.
    // The flag is read here rather than inside the loop because an input that no consumer reads is still expanded, and an expanded input with no buffer aborts at load.
    static const bool engram_off = [] {
        const char * e = getenv("LLAMA_DSV41_NO_ENGRAM");
        const bool off = e != nullptr && atoi(e) != 0;
        if (off) {
            LLAMA_LOG_WARN("deepseek41: LLAMA_DSV41_NO_ENGRAM=1, engram disabled\n");
        }
        return off;
    }();

    // one row index per hash column per table, hashed host side from the token history
    ggml_tensor * inp_engram = nullptr;
    if (hparams.engram_n_layers > 0 && !engram_off) {
        inp_engram = build_inp_engram(model);
        ggml_build_forward_expand(gf, inp_engram);
    }

    for (int il = 0; il < n_layer; ++il) {
        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = dsv4_hc_mean(ctx0, inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            // expanded by set_outputs at the END of the build - expanding here walks
            // the dependency chain in a different DFS order than the natural build
            // and interleaves every device boundary, adding 2 sched splits per
            // boundary under -sm layer (measured 65 vs 51 splits)
        }

        // the engram writes into the residual before the block body, so the whole layer sees it.
        if (!engram_off && model.layers[il].engram_embed) {
            inpL = build_engram(model, inpL, inp_engram, il);
            cb(inpL, "engram_out", il);
        }

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        ggml_tensor * attn_pre = nullptr;
        cur = build_hc_pre(inpL,
                model.layers[il].hc_attn_fn,
                model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base,
                &post, &comb, il, &attn_pre, hc_shift ? carry_pre : nullptr,
                /*collapse =*/ !(hc_shift && carry_pre == nullptr));
        if (hc_shift && carry_pre == nullptr) {
            // the initial one-hot mix selects copy 0
            cur = ggml_cont_2d(ctx0, ggml_view_2d(ctx0, inpL, n_embd, inpL->ne[2], inpL->nb[2], 0),
                    n_embd, inpL->ne[2]);
        }
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention(model, inp_dsv4, cur, inp_pos, il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = build_hc_pre(inpL,
                model.layers[il].hc_ffn_fn,
                model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base,
                &post, &comb, il, &last_ffn_pre, hc_shift ? attn_pre : nullptr);
        if (hc_shift) {
            carry_pre = last_ffn_pre;
        }
        cb(cur, "hc_ffn_pre", il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        const auto & layer = model.layers[il];
        ggml_tensor * selected_experts = nullptr;
        ggml_tensor * exp_probs_b = layer.ffn_exp_probs_b;

        // may apply exp_probs_b_vl is input is from mtmd
        const bool is_media = ubatch.embd != nullptr;
        if (is_media) {
            if (layer.ffn_exp_probs_b_vl) {
                exp_probs_b = layer.ffn_exp_probs_b_vl;
            }
        } else if ((uint32_t) il < hparams.dsv4_hash_layer_count) {
            selected_experts = ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, res->t_inp_tokens);
            exp_probs_b = nullptr;
        }

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                exp_probs_b,
                n_expert, hparams.n_expert_used(),
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                selected_experts);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_last", il);
    }

    if ((size_t) n_layer < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[n_layer]) {
        res->t_layer_inp[n_layer] = dsv4_hc_mean(ctx0, inpL);
        cb(res->t_layer_inp[n_layer], "layer_inp", n_layer);
        // expanded by set_outputs, see the per-layer taps above
    }

    ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
    ggml_tensor * flat_out = inp_out_ids ? ggml_get_rows(ctx0, flat, inp_out_ids) : flat;

    if (cparams.embeddings_nextn) {
        ggml_tensor * h_nextn = cparams.embeddings_nextn_masked ? flat_out : inpL;
        cb(h_nextn, "h_nextn", -1);
        res->t_h_nextn = h_nextn;
    }

    if (inp_out_ids) {
        inpL = ggml_reshape_3d(ctx0, flat_out, n_embd, hc, n_outputs);

        // V4.1 folds with the mix the last layer computed, and that mix has one row per token. build_hc_pre() takes its row count from x, so once x is narrowed to the output rows an un-narrowed mix silently supplies row 0 - the first token's coefficients applied to the last token.
        // Narrow it the same way.
        if (last_ffn_pre) {
            last_ffn_pre = ggml_get_rows(ctx0, last_ffn_pre, inp_out_ids);
            cb(last_ffn_pre, "hc_ffn_pre_out", -1);
        }
    }

    cur = build_head_fold(model, inpL);
    cb(cur, "hc_head", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}


llama_model_deepseek4::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "DEEPSEEK4 MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "DEEPSEEK4 MTP currently only supports a single MTP block");
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
            cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
            "nextn_layer_offset out of range [0, n_layer_nextn)");
    GGML_ASSERT(ubatch.token && "DEEPSEEK4 MTP requires token input");

    const int64_t hc = hparams.dsv4_hc_mult;
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) (n_embd*hc) && "DEEPSEEK4 MTP hidden width mismatch");

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_out());

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_out(), n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_out(), n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
    ggml_tensor * tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h_state = ggml_reshape_3d(ctx0, inp->h, n_embd, hc, n_tokens);
    cb(h_state, "mtp_h_state", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    llm_graph_input_attn_k_iswa * inp_attn = build_attn_inp_k_iswa();

    ggml_tensor * h_norm = build_norm(h_state, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens);
    e_norm = ggml_repeat_4d(ctx0, e_norm, n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * inpL = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(inpL, "mtp_eh_proj", il);

    ggml_tensor * residual = inpL;
    ggml_tensor * post = nullptr;
    ggml_tensor * comb = nullptr;

    ggml_tensor * cur = build_hc_pre(inpL,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            &post, &comb, il);
    cb(cur, "mtp_hc_attn_pre", il);

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    cur = build_attention(model, inp_attn, cur, inp_pos, il);

    inpL = build_hc_post(cur, residual, post, comb, il);
    cb(inpL, "mtp_hc_attn_post", il);

    residual = inpL;
    cur = build_hc_pre(inpL,
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            &post, &comb, il);
    cb(cur, "mtp_hc_ffn_pre", il);

    cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    GGML_ASSERT((uint32_t) il >= hparams.dsv4_hash_layer_count && "DEEPSEEK4 MTP does not support hash-routed MTP blocks");
    ggml_tensor * moe_out = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, hparams.n_expert_used(),
            LLM_FFN_SILU, hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il);
    cb(moe_out, "mtp_ffn_moe_out", il);

    ggml_tensor * ffn_shexp = build_ffn(cur,
            layer.ffn_up_shexp, nullptr, nullptr,
            layer.ffn_gate_shexp, nullptr, nullptr,
            layer.ffn_down_shexp, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(ffn_shexp, "mtp_ffn_shexp", il);

    cur = ggml_add(ctx0, moe_out, ffn_shexp);
    cb(cur, "mtp_ffn_out", il);

    inpL = build_hc_post(cur, residual, post, comb, il);
    inpL = build_cvec(inpL, il);
    cb(inpL, "mtp_l_out", il);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
    ggml_tensor * h_nextn = ggml_get_rows(ctx0, flat, inp_out_ids);
    cb(h_nextn, "h_nextn", -1);
    res->t_h_nextn = h_nextn;

    inpL = ggml_reshape_3d(ctx0, h_nextn, n_embd, hc, n_outputs);

    cur = build_head_fold(model, inpL);
    cb(cur, "mtp_hc_head", -1);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm ? layer.nextn.shared_head_norm : model.output_norm;
    GGML_ASSERT(head_norm_w && "DEEPSEEK4 MTP missing shared head norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "mtp_shared_head_norm", -1);
    res->t_embd = cur;

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    GGML_ASSERT(head_w && "DEEPSEEK4 MTP missing LM head");
    cur = ggml_mul_mat(ctx0, head_w, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
