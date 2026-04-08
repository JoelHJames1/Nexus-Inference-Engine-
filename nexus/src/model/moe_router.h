#pragma once
/// NEXUS MoE Router — Auxiliary-loss-free gating with predictive prefetch.
///
/// Implements DeepSeek-V3 style load balancing: instead of adding an auxiliary
/// loss to the training objective, we maintain per-expert bias terms that are
/// adjusted at runtime to balance token-to-expert assignments.  This avoids
/// the representation-collapse problem of traditional MoE auxiliary losses.
///
/// The router also predicts which experts the NEXT token will likely need,
/// enabling the streaming engine to start prefetching expert weights from SSD
/// before they're needed.

#include "core/config.h"
#include <cstdint>
#include <mutex>
#include <vector>
#include <utility>

namespace nexus::model {

/// Result of top-k expert routing for a single token.
struct TopKResult {
    /// (expert_id, normalized_weight) pairs, sorted by weight descending.
    std::vector<std::pair<int, float>> expert_weights;
};

/// Mixture-of-Experts gating router with auxiliary-loss-free load balancing.
class MoERouter {
public:
    /// \param num_experts     Total number of experts (e.g., 256 for DeepSeek-V3).
    /// \param num_active      Number of experts activated per token (e.g., 8).
    /// \param hidden_dim      Dimension of token hidden states.
    MoERouter(int num_experts, int num_active, int hidden_dim);
    ~MoERouter();

    MoERouter(const MoERouter&) = delete;
    MoERouter& operator=(const MoERouter&) = delete;

    /// Route a token's hidden state to the top-k experts.
    ///
    /// 1. Compute gate logits: hidden_state @ gate_weights^T  [num_experts]
    /// 2. Add per-expert bias for load balancing
    /// 3. Apply softmax
    /// 4. Select top-k experts by gated score
    /// 5. Renormalize selected weights to sum to 1
    /// 6. Update per-expert token counts and adjust biases
    TopKResult route(const float* hidden_state);

    /// Predict which experts the next token will likely need, based on
    /// the current token's gate logits.  Returns expert IDs sorted by
    /// predicted likelihood (up to `count` experts).
    ///
    /// Strategy: take the top experts from a smoothed running average of
    /// recent gate logit distributions.  This works because consecutive
    /// tokens in natural language tend to activate similar experts.
    std::vector<int> predict_next_experts(const float* current_gate_logits, int count) const;

    /// Load gate weights from an external buffer (e.g., from NXF tensor).
    /// Expects row-major float[hidden_dim x num_experts].
    void load_gate_weights(const float* weights);

    /// Reset load-balancing statistics (e.g., at start of new sequence).
    void reset_balancing();

    // ── Accessors ──────────────────────────────────────────────────────────
    int num_experts()  const { return num_experts_; }
    int num_active()   const { return num_active_; }
    int hidden_dim()   const { return hidden_dim_; }

    /// Per-expert token counts since last reset (for diagnostics).
    const std::vector<int64_t>& expert_counts() const { return expert_counts_; }

    /// Current bias terms (for diagnostics).
    const std::vector<float>& expert_biases() const { return expert_biases_; }

private:
    int num_experts_;
    int num_active_;
    int hidden_dim_;

    /// Gate projection weights: [hidden_dim x num_experts], row-major.
    /// gate_logit[e] = dot(hidden_state, gate_weights_[e * hidden_dim ...])
    std::vector<float> gate_weights_;

    /// Per-expert additive bias for auxiliary-loss-free load balancing.
    /// Adjusted after each routing decision to steer tokens toward
    /// underutilized experts.
    std::vector<float> expert_biases_;

    /// Running count of tokens assigned to each expert.
    std::vector<int64_t> expert_counts_;

    /// Total tokens routed since last reset.
    int64_t total_tokens_routed_ = 0;

    /// Exponential moving average of gate logits for next-expert prediction.
    std::vector<float> ema_gate_logits_;
    static constexpr float kEmaAlpha = 0.3f;

    /// Bias update rate: how aggressively to rebalance.
    static constexpr float kBiasUpdateRate = 0.001f;

    /// Mutex for thread-safe bias/count updates.
    mutable std::mutex mu_;

    /// Compute raw gate logits (before bias and softmax).
    void compute_gate_logits(const float* hidden_state, float* logits) const;

    /// Apply softmax in-place.
    static void softmax(float* logits, int n);

    /// Update load-balancing biases after a routing decision.
    void update_biases(const TopKResult& result);
};

}  // namespace nexus::model
