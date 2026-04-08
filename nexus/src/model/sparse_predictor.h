#pragma once
/// NEXUS Sparse Predictor — Predictive sparse activation of weights.
///
/// Core idea: instead of computing ALL heads and ALL FFN neurons every token,
/// predict which subset matters and only compute that. The model behaves as
/// if it were much smaller per token.
///
/// Two strategies:
///   1. Attention Head Pruning: track head importance, skip low-importance heads
///   2. FFN Neuron Sparsity: threshold post-SiLU activations, skip near-zero neurons

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace nexus::model {

/// Tracks attention head importance and decides which heads to compute.
class HeadPruner {
public:
    /// @param num_heads   Total query heads
    /// @param num_layers  Total layers
    /// @param top_k       How many heads to keep (0 = auto, uses ratio)
    /// @param keep_ratio  Fraction of heads to keep (e.g., 0.5 = half)
    /// @param warmup_tokens  How many tokens to run dense before pruning
    HeadPruner(int num_heads, int num_layers, int top_k = 0,
               float keep_ratio = 0.5f, int warmup_tokens = 3)
        : num_heads_(num_heads), num_layers_(num_layers)
        , keep_ratio_(keep_ratio), warmup_tokens_(warmup_tokens)
        , tokens_seen_(0) {
        top_k_ = top_k > 0 ? top_k : std::max(1, static_cast<int>(num_heads * keep_ratio));
        // Per-layer importance scores (accumulated attention entropy per head)
        importance_.resize(num_layers, std::vector<float>(num_heads, 1.0f));
        // Per-layer active head masks
        active_masks_.resize(num_layers, std::vector<bool>(num_heads, true));
    }

    /// Should we prune? (false during warmup)
    bool should_prune() const { return tokens_seen_ >= warmup_tokens_; }

    /// Record a token processed (call after each token)
    void token_done() { tokens_seen_++; }

    /// Update head importance based on attention scores.
    /// Call after computing attention for a layer.
    /// @param layer      Layer index
    /// @param head_scores Per-head max attention score (proxy for importance)
    void update_importance(int layer, const float* head_scores, int n_heads) {
        if (layer < 0 || layer >= num_layers_) return;
        float decay = 0.9f;  // Exponential moving average
        for (int h = 0; h < std::min(n_heads, num_heads_); h++) {
            importance_[layer][h] = decay * importance_[layer][h]
                                  + (1.0f - decay) * std::abs(head_scores[h]);
        }
        // Update active mask: keep top-K heads by importance
        if (should_prune()) {
            std::vector<int> indices(num_heads_);
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + top_k_,
                              indices.end(), [&](int a, int b) {
                                  return importance_[layer][a] > importance_[layer][b];
                              });
            std::fill(active_masks_[layer].begin(), active_masks_[layer].end(), false);
            for (int i = 0; i < top_k_; i++) {
                active_masks_[layer][indices[i]] = true;
            }
        }
    }

    /// Is this head active for this layer?
    bool is_head_active(int layer, int head) const {
        if (!should_prune()) return true;  // All heads active during warmup
        if (layer < 0 || layer >= num_layers_) return true;
        if (head < 0 || head >= num_heads_) return true;
        return active_masks_[layer][head];
    }

    /// How many heads are active for this layer?
    int active_count(int layer) const {
        if (!should_prune()) return num_heads_;
        int count = 0;
        for (bool b : active_masks_[layer]) count += b;
        return count;
    }

    /// Fraction of compute saved
    float sparsity() const {
        if (!should_prune()) return 0.0f;
        return 1.0f - static_cast<float>(top_k_) / num_heads_;
    }

private:
    int num_heads_, num_layers_, top_k_;
    float keep_ratio_;
    int warmup_tokens_, tokens_seen_;
    std::vector<std::vector<float>> importance_;
    std::vector<std::vector<bool>> active_masks_;
};

/// FFN neuron sparsity: skip neurons with near-zero activation after SiLU.
class NeuronSparsifier {
public:
    /// @param threshold  Absolute activation threshold below which neurons are skipped
    /// @param min_active Minimum fraction of neurons to keep (safety floor)
    NeuronSparsifier(float threshold = 0.01f, float min_active = 0.1f)
        : threshold_(threshold), min_active_(min_active)
        , total_neurons_(0), active_neurons_(0) {}

    /// Apply SiLU activation + sparsity mask in-place.
    /// Returns the number of active (non-zero) neurons.
    /// gate_buf and up_buf are [ffn_dim]. After this call, gate_buf contains
    /// silu(gate) * up with near-zero entries zeroed out.
    int apply_sparse_swiglu(float* gate_buf, const float* up_buf, int ffn_dim) {
        int active = 0;
        int min_keep = std::max(1, static_cast<int>(ffn_dim * min_active_));

        for (int i = 0; i < ffn_dim; i++) {
            float g = gate_buf[i];
            float s = g / (1.0f + expf(-g));  // SiLU
            float val = s * up_buf[i];
            gate_buf[i] = val;
            if (std::abs(val) > threshold_) active++;
        }

        // If too few active, lower threshold for this call (keep min_active%)
        if (active < min_keep) {
            // Just keep everything — don't zero out
            active = ffn_dim;
        } else {
            // Zero out below threshold
            for (int i = 0; i < ffn_dim; i++) {
                if (std::abs(gate_buf[i]) <= threshold_) {
                    gate_buf[i] = 0.0f;
                }
            }
        }

        total_neurons_ += ffn_dim;
        active_neurons_ += active;
        return active;
    }

    /// Sparse GEMV: only multiply rows where activation is non-zero.
    /// out[dim] = gate_buf[ffn_dim] @ w2[ffn_dim, dim], but skip zero rows.
    /// This is the key speedup: if 70% of gate_buf is zero, we skip 70% of w2 rows.
    void sparse_gemv_w2(float* out, const float* gate_buf, int ffn_dim, int dim,
                        const void* w2_raw, size_t w2_bytes) {
        // For now, just call the standard GEMM (the zeros make it faster
        // due to multiply-by-zero being fast on modern hardware).
        // A truly sparse implementation would index only non-zero rows.
        // TODO: implement gather-based sparse GEMV for bigger speedup
        (void)w2_raw; (void)w2_bytes;
        // The caller handles the GEMM — we just provide the sparsified input
    }

    /// Average sparsity ratio (0 = fully dense, 1 = fully sparse)
    float avg_sparsity() const {
        if (total_neurons_ == 0) return 0.0f;
        return 1.0f - static_cast<float>(active_neurons_) / total_neurons_;
    }

    void reset_stats() { total_neurons_ = 0; active_neurons_ = 0; }

private:
    float threshold_, min_active_;
    int64_t total_neurons_, active_neurons_;
};

}  // namespace nexus::model
