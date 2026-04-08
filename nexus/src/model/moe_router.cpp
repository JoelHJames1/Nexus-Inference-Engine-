/// NEXUS MoE Router — Auxiliary-loss-free gating with predictive prefetch.
///
/// Load balancing strategy (DeepSeek-V3 style):
///   - Each expert has a bias term added to its gate logit before softmax.
///   - After routing, we compare each expert's actual usage fraction to
///     the ideal uniform fraction (1/num_experts).
///   - Overloaded experts get their bias decreased; underloaded experts
///     get their bias increased.
///   - The bias update rate is small enough that routing remains stable
///     but large enough to prevent expert collapse over a sequence.
///
/// Next-expert prediction:
///   - We maintain an EMA of gate logit distributions across tokens.
///   - Consecutive tokens in natural language tend to activate similar
///     experts (semantic locality), so the EMA is a reasonable predictor.
///   - The streaming engine uses these predictions to prefetch expert
///     weights from SSD before they're needed.

#include "model/moe_router.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace nexus::model {

MoERouter::MoERouter(int num_experts, int num_active, int hidden_dim)
    : num_experts_(num_experts)
    , num_active_(num_active)
    , hidden_dim_(hidden_dim)
    , gate_weights_(static_cast<size_t>(hidden_dim) * num_experts, 0.0f)
    , expert_biases_(num_experts, 0.0f)
    , expert_counts_(num_experts, 0)
    , ema_gate_logits_(num_experts, 0.0f)
{
}

MoERouter::~MoERouter() = default;

void MoERouter::load_gate_weights(const float* weights) {
    std::memcpy(gate_weights_.data(), weights,
                static_cast<size_t>(hidden_dim_) * num_experts_ * sizeof(float));
}

void MoERouter::reset_balancing() {
    std::lock_guard<std::mutex> lock(mu_);
    std::fill(expert_biases_.begin(), expert_biases_.end(), 0.0f);
    std::fill(expert_counts_.begin(), expert_counts_.end(), 0);
    std::fill(ema_gate_logits_.begin(), ema_gate_logits_.end(), 0.0f);
    total_tokens_routed_ = 0;
}

// ─── Core routing ──────────────────────────────────────────────────────────────

TopKResult MoERouter::route(const float* hidden_state) {
    // 1. Compute raw gate logits: hidden_state @ gate_weights^T
    std::vector<float> logits(num_experts_);
    compute_gate_logits(hidden_state, logits.data());

    // 2. Add per-expert bias for load balancing (read under lock)
    {
        std::lock_guard<std::mutex> lock(mu_);
        for (int e = 0; e < num_experts_; e++) {
            logits[e] += expert_biases_[e];
        }
    }

    // 3. Apply softmax to get gating probabilities
    softmax(logits.data(), num_experts_);

    // 4. Select top-k experts by gated score
    //    We use partial_sort via an index array for efficiency.
    std::vector<int> indices(num_experts_);
    std::iota(indices.begin(), indices.end(), 0);
    int k = std::min(num_active_, num_experts_);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });

    // 5. Build result with renormalized weights
    TopKResult result;
    result.expert_weights.reserve(k);

    float sum_weights = 0.0f;
    for (int i = 0; i < k; i++) {
        sum_weights += logits[indices[i]];
    }

    // Guard against degenerate case where all logits are ~0
    if (sum_weights <= 1e-12f) {
        sum_weights = 1.0f;
    }

    for (int i = 0; i < k; i++) {
        int eid = indices[i];
        float w = logits[eid] / sum_weights;
        result.expert_weights.emplace_back(eid, w);
    }

    // 6. Update EMA of gate logits (for next-expert prediction)
    //    and load-balancing statistics
    {
        std::lock_guard<std::mutex> lock(mu_);
        for (int e = 0; e < num_experts_; e++) {
            ema_gate_logits_[e] = kEmaAlpha * logits[e]
                                + (1.0f - kEmaAlpha) * ema_gate_logits_[e];
        }
    }

    // 7. Update biases for load balancing
    update_biases(result);

    return result;
}

// ─── Next-expert prediction ────────────────────────────────────────────────────

std::vector<int> MoERouter::predict_next_experts(const float* /*current_gate_logits*/,
                                                  int count) const {
    // Use the EMA of gate logits as a predictor of upcoming expert usage.
    // This exploits semantic locality: consecutive tokens tend to activate
    // similar experts because they share semantic context.

    std::vector<float> ema_copy;
    {
        std::lock_guard<std::mutex> lock(mu_);
        ema_copy = ema_gate_logits_;
    }

    int n = std::min(count, num_experts_);
    std::vector<int> indices(num_experts_);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + n, indices.end(),
                      [&](int a, int b) { return ema_copy[a] > ema_copy[b]; });

    indices.resize(n);
    return indices;
}

// ─── Gate logit computation ────────────────────────────────────────────────────

void MoERouter::compute_gate_logits(const float* hidden_state, float* logits) const {
    // gate_weights_ layout: [num_experts x hidden_dim], row-major.
    // logits[e] = dot(hidden_state, gate_weights_[e * hidden_dim .. (e+1)*hidden_dim])
    //
    // This is effectively: logits = gate_weights_ @ hidden_state
    // i.e., a matrix-vector product where the matrix is [num_experts x hidden_dim].

    for (int e = 0; e < num_experts_; e++) {
        const float* row = gate_weights_.data() + static_cast<size_t>(e) * hidden_dim_;
        float dot = 0.0f;

        // Manual dot product.  For production, this should use vDSP_dotpr
        // or Accelerate cblas_sdot for NEON vectorization.
        int i = 0;

        // Process 4 elements at a time (compiler will auto-vectorize with -O2)
        for (; i + 3 < hidden_dim_; i += 4) {
            dot += row[i]     * hidden_state[i]
                 + row[i + 1] * hidden_state[i + 1]
                 + row[i + 2] * hidden_state[i + 2]
                 + row[i + 3] * hidden_state[i + 3];
        }
        // Remainder
        for (; i < hidden_dim_; i++) {
            dot += row[i] * hidden_state[i];
        }

        logits[e] = dot;
    }
}

// ─── Softmax ───────────────────────────────────────────────────────────────────

void MoERouter::softmax(float* logits, int n) {
    // Numerically stable softmax: subtract max first.
    float max_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        logits[i] = std::exp(logits[i] - max_val);
        sum += logits[i];
    }

    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < n; i++) {
            logits[i] *= inv_sum;
        }
    }
}

// ─── Load-balancing bias update ────────────────────────────────────────────────

void MoERouter::update_biases(const TopKResult& result) {
    std::lock_guard<std::mutex> lock(mu_);

    // Increment token counts for selected experts
    for (const auto& [eid, _weight] : result.expert_weights) {
        expert_counts_[eid]++;
    }
    total_tokens_routed_++;

    // Don't update biases until we have enough statistics
    if (total_tokens_routed_ < num_experts_) return;

    // Target: each expert should receive (num_active / num_experts) fraction
    // of total tokens.  If an expert is over-utilized, decrease its bias;
    // if under-utilized, increase its bias.
    float target_fraction = static_cast<float>(num_active_) / static_cast<float>(num_experts_);
    float inv_total = 1.0f / static_cast<float>(total_tokens_routed_);

    for (int e = 0; e < num_experts_; e++) {
        float actual_fraction = static_cast<float>(expert_counts_[e]) * inv_total;
        float error = target_fraction - actual_fraction;

        // Proportional update: push bias toward balance
        expert_biases_[e] += kBiasUpdateRate * error;

        // Clamp biases to prevent runaway drift
        constexpr float kMaxBias = 1.0f;
        if (expert_biases_[e] > kMaxBias)  expert_biases_[e] = kMaxBias;
        if (expert_biases_[e] < -kMaxBias) expert_biases_[e] = -kMaxBias;
    }
}

}  // namespace nexus::model
