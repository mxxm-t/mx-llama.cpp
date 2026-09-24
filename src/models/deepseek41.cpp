#include "llama-hparams.h"
#include "llama-kv-cache-dsv4.h"
#include "models.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <vector>

// Copy a bounded table out of the model file, refusing anything that would not fit.
template <typename T, size_t N>
static void dsv41_copy(const std::vector<T> & src, std::array<T, N> & dst, const char * what) {
    if (src.size() > N) {
        throw std::runtime_error(std::string("deepseek41: ") + what + " is larger than the fixed bound");
    }
    std::copy(src.begin(), src.end(), dst.begin());
}

void llama_model_deepseek41::load_arch_hparams(llama_model_loader & ml) {
    llama_model_deepseek4::load_arch_hparams(ml);

    std::vector<uint32_t> kv_src, index_src, layer_ids;
    std::vector<uint64_t> num_embeddings, primes, offsets, multipliers;

    ml.get_arr(LLM_KV_ATTENTION_KV_SOURCE_LAYER_IDS,    kv_src,    false);
    ml.get_arr(LLM_KV_ATTENTION_INDEX_SOURCE_LAYER_IDS, index_src, false);
    ml.get_key(LLM_KV_ATTENTION_CANDIDATE_SOURCE_LAYER_ID, hparams.dsv41_candidate_src_layer, false);
    ml.get_key(LLM_KV_ATTENTION_CANDIDATE_BLOCK_SIZE,      hparams.dsv41_candidate_block,     false);
    ml.get_key(LLM_KV_ATTENTION_CANDIDATE_TOPK_BLOCKS,     hparams.dsv41_candidate_topk,      false);

    // V4 runs one tier at ratio 4 with an indexer and one at 128.
    // V4.1 runs 2 and 1, and declares them per layer, so take the pair from the array rather than assuming either.
    uint32_t lo = 0, hi = 0;
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        const uint32_t r = hparams.dsv4_compress_ratios[il];
        if (r == 0 || r == lo || r == hi) {
            continue;
        }
        if (hi == 0 || r > hi) {
            lo = hi;
            hi = r;
        } else if (lo == 0 || r > lo) {
            lo = r;
        } else {
            throw std::runtime_error("deepseek41: more than two compression ratios");
        }
    }
    if (hi == 0) {
        throw std::runtime_error("deepseek41: no compressed layers in compress_ratios");
    }
    hparams.dsv4_ratio_idx   = hi;
    hparams.dsv4_ratio_plain = lo ? lo : hi;

    hparams.dsv41_n_kv_source    = (uint32_t) kv_src.size();
    hparams.dsv41_n_index_source = (uint32_t) index_src.size();
    dsv41_copy(kv_src,    hparams.dsv41_kv_source_layers,    "kv_source_layer_ids");
    dsv41_copy(index_src, hparams.dsv41_index_source_layers, "index_source_layer_ids");

    ml.get_arr(LLM_KV_ENGRAM_LAYER_IDS,      layer_ids,        false);
    ml.get_arr(LLM_KV_ENGRAM_NUM_EMBEDDINGS, num_embeddings,   false);
    ml.get_arr(LLM_KV_ENGRAM_PRIMES,         primes,           false);
    ml.get_arr(LLM_KV_ENGRAM_OFFSETS,        offsets,          false);
    ml.get_arr(LLM_KV_ENGRAM_MULTIPLIERS,    multipliers,      false);
    ml.get_arr(LLM_KV_ENGRAM_TOKEN_MAP,      engram_token_map, false);

    ml.get_key(LLM_KV_ENGRAM_HEAD_DIM,              hparams.engram_head_dim,         false);
    ml.get_key(LLM_KV_ENGRAM_N_HEADS,               hparams.engram_n_heads,          false);
    ml.get_key(LLM_KV_ENGRAM_MAX_NGRAM_SIZE,        hparams.engram_max_ngram_size,   false);
    ml.get_key(LLM_KV_ENGRAM_PAD_TOKEN_ID,          hparams.engram_pad_token_id,     false);
    ml.get_key(LLM_KV_ENGRAM_COMPRESSED_VOCAB_SIZE, hparams.engram_compressed_vocab, false);

    if (layer_ids.empty()) {
        return;
    }

    hparams.engram_n_layers = (uint32_t) layer_ids.size();
    dsv41_copy(layer_ids,      hparams.engram_layer_ids,      "engram layer_ids");
    dsv41_copy(num_embeddings, hparams.engram_num_embeddings, "engram num_embeddings");
    dsv41_copy(primes,         hparams.engram_primes,         "engram primes");
    dsv41_copy(offsets,        hparams.engram_offsets,        "engram offsets");
    dsv41_copy(multipliers,    hparams.engram_multipliers,    "engram multipliers");

    // Every lookup is indexed by a hash built from these tables, so a partial set would read the wrong rows rather than fail.
    const size_t n_eng  = hparams.engram_n_layers;
    const size_t n_cols = hparams.engram_n_hash_cols();

    if (num_embeddings.size() != n_eng ||
        multipliers.size()    != n_eng*hparams.engram_max_ngram_size ||
        primes.size()         != n_eng*n_cols ||
        offsets.size()        != n_eng*n_cols ||
        engram_token_map.empty()) {
        throw std::runtime_error("deepseek41: incomplete engram hash tables in the model file");
    }
}

void llama_model_deepseek41::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const int64_t n_embd_head  = hparams.n_embd_head_k();
    const int64_t o_groups     = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank  = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult      = hparams.dsv4_hc_mult;
    const int64_t hc_dim       = hc_mult * n_embd;
    const int64_t hc_mix_dim   = (2 + hc_mult) * hc_mult;
    const int64_t n_embd_index = hparams.indexer_head_size;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    // V4.1 folds the hyper-connection copies with the mix of the last layer, so it carries no output_hc_* head tensors of its own.

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, 0);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, 0);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, 0);
        layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, 0);
        layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, 0);
        layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank, o_groups}, TENSOR_ALLOW_RESHAPE);
        layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, 0);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, 0);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);

        // V4 keys the compressor off a compression ratio of 4.
        // V4.1 instead keeps one copy on each source layer and lets the layers between them reuse it.
        if (hparams.dsv41_is_kv_source(i)) {
            layer.attn_comp_wkv   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV,   "weight", i), {n_embd, n_embd_head}, 0);
            layer.attn_comp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM,  "weight", i), {n_embd_head}, 0);
            // ratio 1 pools nothing, so the source layer that runs at ratio 1 carries no gate
            layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", i), {n_embd, n_embd_head}, TENSOR_NOT_REQUIRED);

            // this pair reads the compressed KV rather than the residual, so it is {n_embd_head, n_embd_index} and not the {n_embd, 2*n_embd_index} that V4 uses
            layer.indexer_comp_wkv  = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_WKV,  "weight", i), {n_embd_head, n_embd_index}, 0);
            layer.indexer_comp_norm = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_NORM, "weight", i), {n_embd_index}, 0);
        }

        if (hparams.dsv41_is_index_source(i)) {
            layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, 0);
            layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * n_embd_index}, 0);
        }

        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);
        layer.ffn_norm        = create_tensor(tn(LLM_TENSOR_FFN_NORM,        "weight", i), {n_embd}, 0);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    }, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
    }

    if (hparams.engram_n_layers == 0) {
        return;
    }

    const int64_t head_dim = hparams.engram_head_dim;
    const int64_t n_cols   = hparams.engram_n_hash_cols();

    for (uint32_t e = 0; e < hparams.engram_n_layers; ++e) {
        const int il = (int) hparams.engram_layer_ids[e];
        auto & layer = layers[il];

        const int64_t rows = (int64_t) hparams.engram_num_embeddings[e];

        // 48.6 GiB per table and only n_hash_cols rows are read per token, so it is a gather target rather than a weight: read rows on demand, as the PLE table of qwen4exp does.
        layer.engram_embed = create_tensor(tn(LLM_TENSOR_ENGRAM_EMBED, "weight", il), {head_dim, rows}, TENSOR_READ_LAZY);
        layer.engram_wkv   = create_tensor(tn(LLM_TENSOR_ENGRAM_WKV,   "weight", il), {n_cols*head_dim, n_embd*(hc_mult + 1)}, 0);
        layer.engram_q     = create_tensor(tn(LLM_TENSOR_ENGRAM_Q,     "weight", il), {n_embd, hc_mult}, 0);
        layer.engram_k     = create_tensor(tn(LLM_TENSOR_ENGRAM_K,     "weight", il), {n_embd, hc_mult}, 0);
    }
}

// Engram: an n-gram hash memory written into the hyper-connection residual at a few layers.
// Each position hashes the 2-, 3- and 4-gram ending on it, once per head, giving n_hash_cols row indices into that layer table.
// The rows become one key per hc copy plus a shared value, and the value is added in through a gate that measures how well the key matches the stream.
//
// The hash itself runs host side, as the PLE of qwen4exp does: it is int64 multiply, xor and modulo over the token history, none of which ggml has, and the tables it needs (primes from a sympy search, multipliers from a numpy PCG64 stream, a normalizer-derived token map) are baked into the model file because they cannot be recomputed here.
class llm_graph_input_engram : public llm_graph_input_i {
public:
    llm_graph_input_engram(const llama_model_deepseek4 & pmodel,
                           const llama_kv_cache_dsv4_context * mctx) : pmodel(pmodel), mctx(mctx) {}
    virtual ~llm_graph_input_engram() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx);
        return rows->ne[0] == (int64_t) pmodel.hparams.engram_n_hash_cols() * params.ubatch.n_tokens;
    }

    ggml_tensor * rows = nullptr;   // I32 [n_hash_cols*n_tokens, n_engram_layers]
    int n_threads = 0;
    std::vector<ggml_tensor *> emb;
    std::vector<float> gathered;

    const llama_model_deepseek4 & pmodel;

    // the preceding tokens live in the sliding-window KV cells
    const llama_kv_cache_dsv4_context * mctx;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
};

void llm_graph_input_engram::set_input(const llama_ubatch * ubatch) {
    const auto & hp = pmodel.hparams;

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hp.engram_max_ngram_size;
    const int64_t n_heads  = hp.engram_n_heads;
    const int64_t n_cols   = hp.engram_n_hash_cols();
    const int64_t n_eng    = hp.engram_n_layers;
    const int64_t n_prev   = n_gram - 1;

    GGML_ASSERT(mctx != nullptr);
    GGML_ASSERT(!pmodel.engram_token_map.empty());

    // the reference hashes compressed ids, so the padding it substitutes is compressed too
    const int64_t n_vocab_map = (int64_t) pmodel.engram_token_map.size();
    auto compress = [&](llama_token t) -> int64_t {
        return t >= 0 && t < n_vocab_map ? (int64_t) pmodel.engram_token_map[t] : 0;
    };
    const int64_t pad = compress((llama_token) hp.engram_pad_token_id);

    for (int64_t i = 0; i < n_tokens; ++i) {
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "engram hashing does not support tokens shared by multiple sequences");
    }

    // apply_ubatch() has already stored this ubatch, so a token predecessor inside it counts too
    mctx->get_raw()->get_prev_tokens(*ubatch, (uint32_t) n_prev, prev);

    std::vector<int32_t> idx(n_cols*n_tokens*n_eng);
    std::vector<int64_t> tok(n_gram);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // an embd ubatch carries no ids; the reference gives image spans no engram contribution at all, and until that mask exists here, hashing them as padding is the closest stand-in
        tok[0] = ubatch->token ? compress(ubatch->token[i]) : pad;

        // look-back stops at the start of the sequence, and everything at or before the cut pads
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            const llama_token t = cut ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            cut = cut || t < 0;
            tok[s] = cut ? pad : compress(t);
        }

        for (int64_t e = 0; e < n_eng; ++e) {
            const uint64_t * mult   = &hp.engram_multipliers[e*hp.engram_max_ngram_size];
            const uint64_t * primes = &hp.engram_primes[e*n_cols];
            const uint64_t * offs   = &hp.engram_offsets[e*n_cols];

            // xor one look-back in at a time, so the running value after step s is the hash of the (s+1)-gram; each lands in its own prime-sized bucket range of the table
            uint64_t rolling = (uint64_t) tok[0] * mult[0];
            for (int64_t s = 1; s < n_gram; ++s) {
                rolling ^= (uint64_t) tok[s] * mult[s];
                for (int64_t h = 0; h < n_heads; ++h) {
                    const int64_t c = (s - 1)*n_heads + h;
                    idx[e*n_cols*n_tokens + i*n_cols + c] = (int32_t) (rolling % primes[c] + offs[c]);
                }
            }
        }
    }

    for (size_t e = 0; e < emb.size(); ++e) {
        const ggml_tensor * table = pmodel.layers[hp.engram_layer_ids[e]].engram_embed;
        const int64_t per_table = n_cols*n_tokens;
        gathered.resize(static_cast<size_t>(per_table*table->ne[0]));
        const auto to_float = ggml_get_type_traits(table->type)->to_float;
        GGML_ASSERT(to_float && ggml_nelements(emb[e]) == static_cast<int64_t>(gathered.size()));
        const auto * base = static_cast<const uint8_t *>(table->data);
        const int nt = std::max(1, std::min(n_threads, static_cast<int>((per_table + 63)/64)));
        auto work = [&](int t) {
            for (int64_t k = per_table*t/nt; k < per_table*(t + 1)/nt; ++k) {
                const int32_t row = idx[e*per_table + k];
                GGML_ASSERT(row >= 0 && row < table->ne[1]);
                to_float(base + row*table->nb[1], gathered.data() + k*table->ne[0], table->ne[0]);
            }
        };
        std::vector<std::thread> workers;
        try {
            for (int t = 1; t < nt; ++t) { workers.emplace_back(work, t); }
            work(0);
        } catch (...) {
            for (auto & worker : workers) { worker.join(); }
            throw;
        }
        for (auto & worker : workers) { worker.join(); }
        ggml_backend_tensor_set(emb[e], gathered.data(), 0, gathered.size()*sizeof(float));
    }
    ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
}

ggml_tensor * llama_model_deepseek4::graph::build_inp_engram(const llama_model & model) const {
    const auto & pmodel = static_cast<const llama_model_deepseek4 &>(model);

    const int64_t n_cols = hparams.engram_n_hash_cols();
    const int64_t n_eng  = hparams.engram_n_layers;

    auto inp = std::make_unique<llm_graph_input_engram>(
            pmodel, static_cast<const llama_kv_cache_dsv4_context *>(mctx));

    // one contiguous run of row indices per table, so each table gather is a plain 1d view
    inp->rows = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_cols*n_tokens, n_eng);
    ggml_set_input(inp->rows);

    static const int n_threads = [] {
        const char * env = getenv("LLAMA_DSV41_HOST_INPUT_THREADS");
        if (!env) { return 0; }
        char * end = nullptr;
        const long count = std::strtol(env, &end, 10);
        if (end == env || *end || count < 0 || count > 64) {
            throw std::runtime_error("LLAMA_DSV41_HOST_INPUT_THREADS requires 0..64");
        }
        if (count) {
            LLAMA_LOG_WARN("deepseek41: experimental host Engram inputs, threads=%ld, workers joined before upload\n", count);
        }
        return static_cast<int>(count);
    }();
    engram_host_emb.clear();
    if (n_threads > 0) {
        if (model.split_mode() != LLAMA_SPLIT_MODE_LAYER && model.split_mode() != LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("Experimental host Engram inputs require layer or tensor split");
        }
        inp->n_threads = n_threads;
        for (int64_t e = 0; e < n_eng; ++e) {
            const ggml_tensor * table = model.layers[hparams.engram_layer_ids[e]].engram_embed;
            if (!table || (!hparams.no_alloc && !table->data) || !table->buffer || !ggml_backend_buffer_is_host(table->buffer) ||
                    table->type != GGML_TYPE_MXFP4 || table->ne[0] != hparams.engram_head_dim || !ggml_is_contiguous(table)) {
                throw std::runtime_error("Experimental host Engram inputs require canonical host MXFP4 tables");
            }
            auto * tensor = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.engram_head_dim*n_cols, n_tokens);
            ggml_set_input(tensor);
            engram_host_emb.push_back(tensor);
            inp->emb.push_back(tensor);
        }
    }

    ggml_tensor * rows = inp->rows;
    res->add_input(std::move(inp));
    cb(rows, "engram_rows", -1);

    return rows;
}

ggml_tensor * llama_model_deepseek4::graph::build_engram(
        const llama_model & model,
        ggml_tensor * x,
        ggml_tensor * idx,
        int il) const {
    const auto & layer = model.layers[il];

    const int e = hparams.engram_layer_index(il);
    GGML_ASSERT(e >= 0 && idx != nullptr);

    const int64_t n_cols   = hparams.engram_n_hash_cols();
    const int64_t head_dim = hparams.engram_head_dim;
    const int64_t hc       = hparams.dsv4_hc_mult;
    const int64_t nt       = x->ne[2];

    ggml_tensor * emb = nullptr;
    if (static_cast<size_t>(e) < engram_host_emb.size()) {
        emb = engram_host_emb[e];
    } else {
        ggml_tensor * rows = ggml_view_1d(ctx0, idx, n_cols*nt, e*idx->nb[1]);
        emb = ggml_get_rows(ctx0, layer.engram_embed, rows);
        emb = ggml_reshape_2d(ctx0, emb, head_dim*n_cols, nt);
    }
    cb(emb, "engram_embd", il);

    // one key per hc copy, then a single value shared by all of them
    ggml_tensor * kv = build_lora_mm(layer.engram_wkv, emb);
    cb(kv, "engram_kv", il);

    const size_t esz = ggml_element_size(kv);
    ggml_tensor * key = ggml_cont(ctx0,
            ggml_view_3d(ctx0, kv, n_embd, hc, nt, n_embd*esz, kv->nb[1], 0));
    ggml_tensor * value = ggml_cont(ctx0,
            ggml_view_3d(ctx0, kv, n_embd, 1, nt, n_embd*esz, kv->nb[1], hc*n_embd*esz));

    // q and k are only ever used as a product, and the file stores them in bf16
    ggml_tensor * weight = ggml_mul(ctx0,
            ggml_cast(ctx0, layer.engram_q, GGML_TYPE_F32),
            ggml_cast(ctx0, layer.engram_k, GGML_TYPE_F32));

    // the gate is a dot product of stream against key, each normalized per (token, hc copy) over n_embd - which is what rms_norm already computes, so the two rstd factors of the reference come out of the normalization instead of a separate scale
    ggml_tensor * dot = ggml_mul(ctx0,
            ggml_rms_norm(ctx0, x,   hparams.f_norm_rms_eps),
            ggml_rms_norm(ctx0, key, hparams.f_norm_rms_eps));
    dot = ggml_mul(ctx0, dot, weight);
    dot = ggml_sum_rows(ctx0, dot);
    dot = ggml_scale(ctx0, dot, 1.0f/sqrtf((float) n_embd));
    cb(dot, "engram_dot", il);

    // signed sqrt before the sigmoid, matching the training kernel. ggml_sgn is 0 at exactly zero where copysign would give +1, which moves that one gate by 0.00025 and nothing else.
    ggml_tensor * mag = ggml_sqrt(ctx0,
            ggml_clamp(ctx0, ggml_abs(ctx0, dot), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, mag, ggml_sgn(ctx0, dot)));
    cb(gate, "engram_gate", il);

    ggml_tensor * out = ggml_repeat_4d(ctx0, value, n_embd, hc, nt, 1);
    out = ggml_mul(ctx0, out, gate);

    return ggml_add(ctx0, x, out);
}

llama_model_deepseek41::graph::graph(const llama_model & model, const llm_graph_params & params)
    : llama_model_deepseek4::graph(model, params) {
}

std::unique_ptr<llm_graph_context> llama_model_deepseek41::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<llama_model_deepseek41::graph>(*this, params);
}
