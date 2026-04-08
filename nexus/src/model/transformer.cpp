/// NEXUS Transformer — Streaming layer-by-layer execution.
///
/// Core innovation: weights are streamed from SSD via mmap, computed, and evicted.
/// Only 2-3 layers are resident in memory at any time, enabling models far larger
/// than available RAM.

#include "model/transformer.h"
#include "memory/memory_manager.h"
#include "compute/accelerate/gemm.h"
#include "compute/cpu/dequant_neon.h"
#include "core/scheduler.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>
#include <numeric>

namespace nexus::model {

struct Transformer::Impl {
    format::ModelManifest manifest;
    format::NXFReader* reader = nullptr;
    MemoryManager* memory = nullptr;
    std::unique_ptr<Scheduler> scheduler;

    // Per-layer weights (loaded on demand, evicted after use)
    std::vector<LayerWeights> layer_weights;

    // KV cache (persistent across tokens)
    std::vector<LayerKVCache> kv_cache;

    // Embedding table
    float* token_embeddings = nullptr;   // [vocab_size, hidden_dim]
    float* output_norm = nullptr;        // [hidden_dim]
    float* output_weight = nullptr;      // [vocab_size, hidden_dim]

    // Scratch buffers for intermediate activations
    float* hidden_state = nullptr;       // [max_batch, hidden_dim]
    float* residual = nullptr;           // [max_batch, hidden_dim]
    float* attn_output = nullptr;        // [max_batch, hidden_dim]
    float* ffn_output = nullptr;         // [max_batch, hidden_dim]
    float* logits = nullptr;             // [vocab_size]

    // Sequence state
    int current_seq_len = 0;
    int max_seq_len = 0;

    // ─── Layer execution ──────────────────────────────────────────────────
    void execute_layer(uint32_t layer_idx, const float* input, float* output, int seq_pos, int num_tokens);
    void attention(uint32_t layer_idx, const float* x, float* out, int seq_pos, int num_tokens);
    void ffn(uint32_t layer_idx, const float* x, float* out, int num_tokens);

    // ─── Weight loading ───────────────────────────────────────────────────
    bool load_layer_weights(uint32_t layer_idx);
    void evict_layer_weights(uint32_t layer_idx);
    bool load_embedding_weights();

    // ─── Sampling ─────────────────────────────────────────────────────────
    int32_t sample_token(const float* logits, const SamplingParams& params);
};

Transformer::~Transformer() = default;

std::unique_ptr<Transformer> Transformer::create(
    const format::ModelManifest& manifest,
    format::NXFReader& reader,
    MemoryManager& memory) {

    auto tf = std::unique_ptr<Transformer>(new Transformer());
    tf->impl_ = std::make_unique<Impl>();
    tf->impl_->manifest = manifest;
    tf->impl_->reader = &reader;
    tf->impl_->memory = &memory;

    // Initialize scheduler for layer streaming
    SchedulerConfig sched_config;
    sched_config.prefetch_window = 2;
    tf->impl_->scheduler = std::make_unique<Scheduler>(memory, manifest.num_layers, sched_config);

    // Allocate per-layer structures
    tf->impl_->layer_weights.resize(manifest.num_layers);
    tf->impl_->kv_cache.resize(manifest.num_layers);

    // Allocate scratch buffers
    size_t hidden_bytes = manifest.hidden_dim * sizeof(float);
    tf->impl_->hidden_state = static_cast<float*>(memory.alloc_pages(hidden_bytes));
    tf->impl_->residual = static_cast<float*>(memory.alloc_pages(hidden_bytes));
    tf->impl_->attn_output = static_cast<float*>(memory.alloc_pages(hidden_bytes));
    tf->impl_->ffn_output = static_cast<float*>(memory.alloc_pages(hidden_bytes));
    tf->impl_->logits = static_cast<float*>(memory.alloc_pages(manifest.vocab_size * sizeof(float)));

    tf->impl_->max_seq_len = manifest.max_seq_len;

    // Allocate KV cache for each layer
    size_t kv_dim = manifest.num_kv_heads * manifest.head_dim;
    for (uint32_t i = 0; i < manifest.num_layers; i++) {
        size_t kv_bytes = manifest.max_seq_len * kv_dim * sizeof(float);
        tf->impl_->kv_cache[i].keys = static_cast<float*>(memory.alloc_pages(kv_bytes));
        tf->impl_->kv_cache[i].values = static_cast<float*>(memory.alloc_pages(kv_bytes));
        tf->impl_->kv_cache[i].seq_len = 0;
    }

    // Load embedding weights (these stay resident)
    if (!tf->impl_->load_embedding_weights()) {
        fprintf(stderr, "[nexus] WARNING: Could not load embeddings, using zero initialization\n");
    }

    fprintf(stderr, "[nexus] Transformer initialized: %u layers, streaming mode\n", manifest.num_layers);
    return tf;
}

void Transformer::prefill(const std::vector<int32_t>& tokens) {
    auto& m = *impl_;

    // Process all prompt tokens through the model
    int num_tokens = static_cast<int>(tokens.size());

    // Lookup embeddings
    for (int t = 0; t < num_tokens; t++) {
        if (m.token_embeddings && tokens[t] >= 0 && tokens[t] < static_cast<int32_t>(m.manifest.vocab_size)) {
            // Copy embedding for this token into hidden_state
            // For prefill we process one token at a time through all layers
            // (Full batch prefill will come in Phase 2 with Metal)
            memcpy(m.hidden_state,
                   m.token_embeddings + tokens[t] * m.manifest.hidden_dim,
                   m.manifest.hidden_dim * sizeof(float));
        } else {
            memset(m.hidden_state, 0, m.manifest.hidden_dim * sizeof(float));
        }

        // Stream through all layers
        m.scheduler->execute_layers([&](uint32_t layer_idx) {
            m.load_layer_weights(layer_idx);
            m.execute_layer(layer_idx, m.hidden_state, m.hidden_state,
                            m.current_seq_len, 1);
        });

        m.current_seq_len++;
    }

    fprintf(stderr, "[nexus] Prefill complete: %d tokens, seq_len=%d\n",
            num_tokens, m.current_seq_len);
}

int32_t Transformer::decode_step(const SamplingParams& params) {
    auto& m = *impl_;

    if (m.current_seq_len >= m.max_seq_len) {
        fprintf(stderr, "[nexus] Max sequence length reached (%d)\n", m.max_seq_len);
        return -1;
    }

    // Stream through all layers for single token decode
    m.scheduler->execute_layers([&](uint32_t layer_idx) {
        m.load_layer_weights(layer_idx);
        m.execute_layer(layer_idx, m.hidden_state, m.hidden_state,
                        m.current_seq_len, 1);
    });

    // Apply output norm
    if (m.output_norm) {
        compute::rms_norm(m.hidden_state, m.hidden_state, m.output_norm,
                         m.manifest.hidden_dim, m.manifest.rms_norm_eps);
    }

    // Compute logits: hidden_state @ output_weight^T -> logits
    if (m.output_weight) {
        compute::gemm_f32(m.hidden_state, m.output_weight, m.logits,
                         1, m.manifest.vocab_size, m.manifest.hidden_dim,
                         false, true);
    }

    // Sample next token
    int32_t next_token = m.sample_token(m.logits, params);
    m.current_seq_len++;

    return next_token;
}

void Transformer::reset_kv_cache() {
    for (auto& kv : impl_->kv_cache) {
        kv.seq_len = 0;
    }
    impl_->current_seq_len = 0;
}

int Transformer::seq_len() const {
    return impl_->current_seq_len;
}

// ─── Layer execution ────────────────────────────────────────────────────────

void Transformer::Impl::execute_layer(uint32_t layer_idx, const float* input,
                                       float* output, int seq_pos, int num_tokens) {
    auto& lw = layer_weights[layer_idx];
    int dim = manifest.hidden_dim;

    // Save residual
    memcpy(residual, input, dim * sizeof(float));

    // Pre-attention RMSNorm
    if (lw.attn_norm) {
        compute::rms_norm(hidden_state, input, lw.attn_norm, dim, manifest.rms_norm_eps);
    } else {
        memcpy(hidden_state, input, dim * sizeof(float));
    }

    // Self-attention
    attention(layer_idx, hidden_state, attn_output, seq_pos, num_tokens);

    // Residual connection
    for (int i = 0; i < dim; i++) {
        hidden_state[i] = residual[i] + attn_output[i];
    }

    // Save residual
    memcpy(residual, hidden_state, dim * sizeof(float));

    // Pre-FFN RMSNorm
    if (lw.ffn_norm) {
        compute::rms_norm(hidden_state, hidden_state, lw.ffn_norm, dim, manifest.rms_norm_eps);
    }

    // FFN (SwiGLU)
    ffn(layer_idx, hidden_state, ffn_output, num_tokens);

    // Residual connection -> output
    for (int i = 0; i < dim; i++) {
        output[i] = residual[i] + ffn_output[i];
    }
}

void Transformer::Impl::attention(uint32_t layer_idx, const float* x, float* out,
                                   int seq_pos, int num_tokens) {
    auto& lw = layer_weights[layer_idx];
    auto& kv = kv_cache[layer_idx];
    int dim = manifest.hidden_dim;
    int head_dim = manifest.head_dim;
    int n_heads = manifest.num_heads;
    int n_kv_heads = manifest.num_kv_heads;
    int kv_dim = n_kv_heads * head_dim;

    // Allocate temporary Q, K, V
    // (In production, these come from scratch buffer pool)
    std::vector<float> q(dim), k(kv_dim), v(kv_dim);

    // Q = x @ Wq
    if (lw.wq) compute::gemm_f32(x, lw.wq, q.data(), 1, dim, dim, false, true);
    // K = x @ Wk
    if (lw.wk) compute::gemm_f32(x, lw.wk, k.data(), 1, kv_dim, dim, false, true);
    // V = x @ Wv
    if (lw.wv) compute::gemm_f32(x, lw.wv, v.data(), 1, kv_dim, dim, false, true);

    // TODO: Apply RoPE to Q and K

    // Store K, V in cache
    if (kv.keys && kv.values && seq_pos < static_cast<int>(manifest.max_seq_len)) {
        memcpy(kv.keys + seq_pos * kv_dim, k.data(), kv_dim * sizeof(float));
        memcpy(kv.values + seq_pos * kv_dim, v.data(), kv_dim * sizeof(float));
        kv.seq_len = seq_pos + 1;
    }

    // Grouped-Query Attention (GQA)
    int heads_per_group = n_heads / n_kv_heads;
    std::vector<float> attn_out(dim, 0.0f);

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_group;
        float* q_head = q.data() + h * head_dim;

        // Compute attention scores: Q @ K^T / sqrt(d)
        float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
        float max_score = -1e30f;
        std::vector<float> scores(kv.seq_len);

        for (int t = 0; t < kv.seq_len; t++) {
            float* k_t = kv.keys + t * kv_dim + kv_h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_head[d] * k_t[d];
            }
            scores[t] = dot * scale;
            if (scores[t] > max_score) max_score = scores[t];
        }

        // Softmax
        float sum_exp = 0.0f;
        for (int t = 0; t < kv.seq_len; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }
        for (int t = 0; t < kv.seq_len; t++) {
            scores[t] /= sum_exp;
        }

        // Weighted sum of values
        for (int t = 0; t < kv.seq_len; t++) {
            float* v_t = kv.values + t * kv_dim + kv_h * head_dim;
            for (int d = 0; d < head_dim; d++) {
                attn_out[h * head_dim + d] += scores[t] * v_t[d];
            }
        }
    }

    // Output projection: attn_out @ Wo
    if (lw.wo) {
        compute::gemm_f32(attn_out.data(), lw.wo, out, 1, dim, dim, false, true);
    } else {
        memcpy(out, attn_out.data(), dim * sizeof(float));
    }
}

void Transformer::Impl::ffn(uint32_t layer_idx, const float* x, float* out, int num_tokens) {
    auto& lw = layer_weights[layer_idx];
    int dim = manifest.hidden_dim;
    // FFN intermediate dim is typically 8/3 * hidden_dim rounded up
    int ffn_dim = (dim * 8 / 3 + 127) & ~127;  // Rounded to 128

    std::vector<float> gate(ffn_dim), up(ffn_dim);

    // SwiGLU: out = (silu(x @ W1) * (x @ W3)) @ W2
    if (lw.w1) compute::gemm_f32(x, lw.w1, gate.data(), 1, ffn_dim, dim, false, true);
    if (lw.w3) compute::gemm_f32(x, lw.w3, up.data(), 1, ffn_dim, dim, false, true);

    // SiLU on gate, then elementwise multiply with up
    for (int i = 0; i < ffn_dim; i++) {
        float s = gate[i] / (1.0f + expf(-gate[i]));  // SiLU
        gate[i] = s * up[i];
    }

    // Down projection
    if (lw.w2) {
        compute::gemm_f32(gate.data(), lw.w2, out, 1, dim, ffn_dim, false, true);
    }
}

// ─── Weight loading ─────────────────────────────────────────────────────────

bool Transformer::Impl::load_layer_weights(uint32_t layer_idx) {
    auto& lw = layer_weights[layer_idx];
    if (lw.loaded) return true;

    char name_buf[256];
    auto load_tensor = [&](const char* suffix, float*& ptr) -> bool {
        snprintf(name_buf, sizeof(name_buf), "layers.%u.%s", layer_idx, suffix);
        const auto* info = reader->get_tensor(name_buf);
        if (!info || info->chunks.empty()) return false;

        // Map the first chunk (simplified: assumes single chunk per tensor)
        const void* mapped = reader->map_chunk(info->chunks[0]);
        if (!mapped) return false;

        // For FP16/INT4 codecs, we'd dequantize here.
        // For now, assume FP32 or treat as FP32 pointer.
        ptr = const_cast<float*>(static_cast<const float*>(mapped));
        return true;
    };

    load_tensor("attention.wq.weight", lw.wq);
    load_tensor("attention.wk.weight", lw.wk);
    load_tensor("attention.wv.weight", lw.wv);
    load_tensor("attention.wo.weight", lw.wo);
    load_tensor("feed_forward.w1.weight", lw.w1);
    load_tensor("feed_forward.w2.weight", lw.w2);
    load_tensor("feed_forward.w3.weight", lw.w3);
    load_tensor("attention_norm.weight", lw.attn_norm);
    load_tensor("ffn_norm.weight", lw.ffn_norm);

    lw.loaded = true;
    return true;
}

void Transformer::Impl::evict_layer_weights(uint32_t layer_idx) {
    auto& lw = layer_weights[layer_idx];
    // With mmap, eviction = madvise(MADV_DONTNEED)
    // The OS will reclaim the pages and re-fault from SSD if needed.
    // Explicit unmap will come in Phase 3 with double-buffering.
    lw.loaded = false;
    lw.wq = lw.wk = lw.wv = lw.wo = nullptr;
    lw.w1 = lw.w2 = lw.w3 = nullptr;
    lw.attn_norm = lw.ffn_norm = nullptr;
}

bool Transformer::Impl::load_embedding_weights() {
    const auto* embed = reader->get_tensor("tok_embeddings.weight");
    if (embed && !embed->chunks.empty()) {
        const void* mapped = reader->map_chunk(embed->chunks[0]);
        token_embeddings = const_cast<float*>(static_cast<const float*>(mapped));
    }

    const auto* norm = reader->get_tensor("norm.weight");
    if (norm && !norm->chunks.empty()) {
        const void* mapped = reader->map_chunk(norm->chunks[0]);
        output_norm = const_cast<float*>(static_cast<const float*>(mapped));
    }

    const auto* out = reader->get_tensor("output.weight");
    if (out && !out->chunks.empty()) {
        const void* mapped = reader->map_chunk(out->chunks[0]);
        output_weight = const_cast<float*>(static_cast<const float*>(mapped));
    }

    return token_embeddings != nullptr;
}

// ─── Sampling ───────────────────────────────────────────────────────────────

int32_t Transformer::Impl::sample_token(const float* logits_ptr,
                                          const SamplingParams& params) {
    int vocab = manifest.vocab_size;

    if (params.temperature <= 0.0f) {
        // Greedy: argmax
        int best = 0;
        float best_val = logits_ptr[0];
        for (int i = 1; i < vocab; i++) {
            if (logits_ptr[i] > best_val) {
                best_val = logits_ptr[i];
                best = i;
            }
        }
        return best;
    }

    // Temperature scaling
    std::vector<float> probs(vocab);
    float max_logit = *std::max_element(logits_ptr, logits_ptr + vocab);
    float sum = 0.0f;
    for (int i = 0; i < vocab; i++) {
        probs[i] = expf((logits_ptr[i] - max_logit) / params.temperature);
        sum += probs[i];
    }
    for (int i = 0; i < vocab; i++) probs[i] /= sum;

    // Top-k filtering
    if (params.top_k > 0 && params.top_k < vocab) {
        std::vector<int> indices(vocab);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + params.top_k,
                          indices.end(),
                          [&](int a, int b) { return probs[a] > probs[b]; });

        std::vector<float> filtered(vocab, 0.0f);
        for (int i = 0; i < params.top_k; i++) {
            filtered[indices[i]] = probs[indices[i]];
        }
        probs = std::move(filtered);
    }

    // Top-p (nucleus) filtering
    if (params.top_p < 1.0f) {
        std::vector<int> indices(vocab);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&](int a, int b) { return probs[a] > probs[b]; });

        float cumsum = 0.0f;
        for (int i = 0; i < vocab; i++) {
            cumsum += probs[indices[i]];
            if (cumsum > params.top_p) {
                for (int j = i + 1; j < vocab; j++) {
                    probs[indices[j]] = 0.0f;
                }
                break;
            }
        }
    }

    // Renormalize and sample
    sum = 0.0f;
    for (int i = 0; i < vocab; i++) sum += probs[i];
    if (sum <= 0.0f) return 0;

    std::mt19937 rng(params.seed ? params.seed : std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, sum);
    float r = dist(rng);

    float cumsum = 0.0f;
    for (int i = 0; i < vocab; i++) {
        cumsum += probs[i];
        if (cumsum >= r) return i;
    }
    return vocab - 1;
}

}  // namespace nexus::model
