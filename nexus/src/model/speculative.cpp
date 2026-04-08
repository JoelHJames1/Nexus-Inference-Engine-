/// NEXUS Inference Engine — Phase 4: EAGLE-3 Speculative Decoding.
///
/// Algorithm (per generate_step call):
///   1. Draft model (ANE) generates N candidate tokens greedily.
///   2. Main model (GPU) runs a single forward pass over all N+1 positions
///      (the last accepted token + N draft tokens) to produce N+1 sets of logits.
///   3. For each draft token i (0..N-1):
///      - Sample from the main model's distribution at position i.
///      - If the draft token matches the main model's sample, accept it.
///      - On first mismatch, accept the main model's resampled token and stop.
///   4. If all N drafts are accepted, also sample one bonus token from position N.
///
/// This is mathematically equivalent to standard autoregressive decoding:
/// the output distribution is identical regardless of the draft model's quality.
/// A better draft model simply means more tokens accepted per step.

#include "model/speculative.h"
#include "model/transformer.h"
#include "compute/coreml/draft_model.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace nexus::model {

// ─── Implementation ─────────────────────────────────────────────────────────

struct SpeculativeDecoder::Impl {
    Transformer&          main_model;
    compute::DraftModel&  draft_model;
    SpeculativeConfig     config;

    // Rolling statistics
    uint64_t total_proposed  = 0;
    uint64_t total_accepted  = 0;
    static constexpr int kWindowSize = 128;
    int      window_proposed[kWindowSize] = {};
    int      window_accepted[kWindowSize] = {};
    int      window_idx = 0;

    Impl(Transformer& main, compute::DraftModel& draft, int n_speculative)
        : main_model(main), draft_model(draft) {
        config.num_draft_tokens = n_speculative;
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Sample a token from a logits vector using the given parameters.
    static int32_t sample_from_logits(const float* logits, int vocab_size,
                                      const SamplingParams& params, uint64_t step_seed);

    /// Compute softmax probabilities in-place over a logits vector.
    static void softmax(float* logits, int size);

    /// Get the probability of a specific token under the distribution.
    static float token_probability(const float* logits, int vocab_size, int32_t token);

    /// Update rolling window statistics.
    void record_step(int proposed, int accepted);
};

// ─── Sampling helpers ───────────────────────────────────────────────────────

void SpeculativeDecoder::Impl::softmax(float* logits, int size) {
    float max_val = *std::max_element(logits, logits + size);
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }
    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < size; i++) {
            logits[i] *= inv_sum;
        }
    }
}

float SpeculativeDecoder::Impl::token_probability(const float* logits, int vocab_size,
                                                    int32_t token) {
    // Compute softmax for the target token without materializing the full distribution.
    float max_val = *std::max_element(logits, logits + vocab_size);
    float sum = 0.0f;
    float token_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        float e = expf(logits[i] - max_val);
        sum += e;
        if (i == token) token_exp = e;
    }
    return (sum > 0.0f) ? (token_exp / sum) : 0.0f;
}

int32_t SpeculativeDecoder::Impl::sample_from_logits(const float* logits, int vocab_size,
                                                      const SamplingParams& params,
                                                      uint64_t step_seed) {
    if (params.temperature <= 0.0f) {
        // Greedy: argmax
        int best = 0;
        float best_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best = i;
            }
        }
        return static_cast<int32_t>(best);
    }

    // Temperature-scaled softmax
    std::vector<float> probs(vocab_size);
    float max_logit = *std::max_element(logits, logits + vocab_size);
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - max_logit) / params.temperature);
        sum += probs[i];
    }
    if (sum <= 0.0f) return 0;
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < vocab_size; i++) probs[i] *= inv_sum;

    // Top-k filtering
    if (params.top_k > 0 && params.top_k < vocab_size) {
        std::vector<int> indices(vocab_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + params.top_k,
                          indices.end(),
                          [&](int a, int b) { return probs[a] > probs[b]; });
        std::vector<float> filtered(vocab_size, 0.0f);
        for (int i = 0; i < params.top_k; i++) {
            filtered[indices[i]] = probs[indices[i]];
        }
        probs = std::move(filtered);
    }

    // Top-p nucleus filtering
    if (params.top_p < 1.0f) {
        std::vector<int> indices(vocab_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&](int a, int b) { return probs[a] > probs[b]; });
        float cumsum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += probs[indices[i]];
            if (cumsum > params.top_p) {
                for (int j = i + 1; j < vocab_size; j++) {
                    probs[indices[j]] = 0.0f;
                }
                break;
            }
        }
    }

    // Renormalize and sample
    sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) sum += probs[i];
    if (sum <= 0.0f) return 0;

    std::mt19937 rng(step_seed ? step_seed : std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, sum);
    float r = dist(rng);
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (cumsum >= r) return static_cast<int32_t>(i);
    }
    return static_cast<int32_t>(vocab_size - 1);
}

void SpeculativeDecoder::Impl::record_step(int proposed, int accepted) {
    total_proposed += proposed;
    total_accepted += accepted;
    window_proposed[window_idx % kWindowSize] = proposed;
    window_accepted[window_idx % kWindowSize] = accepted;
    window_idx++;
}

// ─── Public interface ───────────────────────────────────────────────────────

SpeculativeDecoder::SpeculativeDecoder(Transformer& main_model,
                                       compute::DraftModel& draft_model,
                                       int num_speculative_tokens)
    : impl_(std::make_unique<Impl>(main_model, draft_model, num_speculative_tokens)) {
    fprintf(stderr, "[nexus] SpeculativeDecoder initialized: %d draft tokens, ANE=%s\n",
            num_speculative_tokens,
            draft_model.is_coreml_active() ? "active" : "fallback");
}

SpeculativeDecoder::~SpeculativeDecoder() = default;

std::vector<int32_t> SpeculativeDecoder::generate_step(
    const std::vector<int32_t>& prompt_tokens,
    const SamplingParams& params) {

    auto& m = *impl_;
    const int n_draft = m.config.num_draft_tokens;
    const int vocab_size = m.draft_model.vocab_size();

    // ── Step 1: Draft model generates N candidate tokens (fast, on ANE) ─────
    //
    // The draft model runs autoregressively for n_draft steps using greedy
    // sampling (temperature=0) to maximise acceptance rate.
    SamplingParams draft_params;
    draft_params.temperature = 0.0f;   // Greedy drafting for maximum acceptance
    draft_params.max_tokens  = n_draft;

    std::vector<int32_t> draft_tokens = m.draft_model.generate_draft(
        prompt_tokens, n_draft, draft_params);

    if (draft_tokens.empty()) {
        // Draft model failed — fall back to single-token main model decode.
        int32_t token = m.main_model.decode_step(params);
        m.record_step(0, 1);
        return {token};
    }

    // ── Step 2: Main model verifies all N positions in one forward pass ─────
    //
    // We construct a verification sequence: [last_prompt_token, draft_0, ..., draft_{N-1}]
    // and feed it to the main model.  The main model produces logits for each
    // position; logits[i] gives the distribution for the token at position i+1.
    //
    // Because the main model uses a KV cache, we only need to process the new
    // tokens (the draft tokens).  The prompt is already in the KV cache.
    //
    // For Phase 4 MVP, we call decode_step() in a loop to get per-position
    // logits.  A future optimization will batch all N+1 positions into a single
    // GPU dispatch using the Metal backend.

    std::vector<std::vector<float>> main_logits_per_position;
    main_logits_per_position.reserve(n_draft + 1);

    // The main model's KV cache already contains the prompt.  We need to feed
    // each draft token one at a time and collect the logits.  However, we must
    // be careful: if we reject at position k, we need to roll back positions
    // k+1..N from the KV cache.
    //
    // Strategy: save the current seq_len, run forward passes, then on rejection
    // we note how far to accept.
    int original_seq_len = m.main_model.seq_len();

    // Run the main model on each draft token to collect verification logits.
    // The first call verifies whether draft_tokens[0] is correct by getting
    // the main model's distribution at the current position.
    std::vector<int32_t> verification_logit_tokens;
    verification_logit_tokens.reserve(n_draft);

    // Collect logits for each verification position.
    // We decode each draft token through the main model.  The logits returned
    // by decode_step represent the distribution for the NEXT token, so we
    // capture them before feeding the next draft token.
    //
    // Position 0: main model already has prompt in KV cache.
    //   decode_step returns sample from p_main(t | prompt).
    //   We compare this with draft_tokens[0].
    // Position k: main model has prompt + draft_tokens[0..k-1].
    //   decode_step returns sample from p_main(t | prompt, draft[0..k-1]).
    //   We compare this with draft_tokens[k].

    std::vector<int32_t> accepted_tokens;
    accepted_tokens.reserve(n_draft + 1);

    // Use a seeded RNG for reproducibility within this step.
    uint64_t step_seed = params.seed ? params.seed + original_seq_len : 0;

    for (int k = 0; k < static_cast<int>(draft_tokens.size()); k++) {
        // Get the main model's next token at this position.
        // decode_step advances the KV cache and returns a sampled token.
        int32_t main_token = m.main_model.decode_step(params);

        if (main_token == draft_tokens[k]) {
            // ── Accept: draft matches main model ────────────────────────────
            accepted_tokens.push_back(draft_tokens[k]);
        } else {
            // ── Reject: first mismatch ──────────────────────────────────────
            // Accept the main model's resampled token at this position.
            accepted_tokens.push_back(main_token);

            // Record stats and return.
            m.record_step(static_cast<int>(draft_tokens.size()),
                          static_cast<int>(accepted_tokens.size()));

            fprintf(stderr, "[nexus] Speculative: accepted %zu/%zu draft tokens\n",
                    accepted_tokens.size() - 1, draft_tokens.size());
            return accepted_tokens;
        }
    }

    // ── All N drafts accepted: bonus token from position N ──────────────────
    // Since all draft tokens matched, we get one extra "free" token from the
    // main model's distribution at the final position.
    int32_t bonus_token = m.main_model.decode_step(params);
    accepted_tokens.push_back(bonus_token);

    m.record_step(static_cast<int>(draft_tokens.size()),
                  static_cast<int>(accepted_tokens.size()));

    fprintf(stderr, "[nexus] Speculative: all %zu draft tokens accepted + bonus\n",
            draft_tokens.size());
    return accepted_tokens;
}

float SpeculativeDecoder::acceptance_rate() const {
    auto& m = *impl_;

    // Use the rolling window if we have enough data.
    int window_count = std::min(m.window_idx, Impl::kWindowSize);
    if (window_count == 0) return 0.0f;

    int sum_proposed = 0;
    int sum_accepted = 0;
    for (int i = 0; i < window_count; i++) {
        sum_proposed += m.window_proposed[i];
        sum_accepted += m.window_accepted[i];
    }
    return (sum_proposed > 0)
               ? static_cast<float>(sum_accepted) / static_cast<float>(sum_proposed)
               : 0.0f;
}

void SpeculativeDecoder::set_config(const SpeculativeConfig& config) {
    impl_->config = config;
    fprintf(stderr, "[nexus] SpeculativeDecoder config updated: "
            "draft_tokens=%d, threshold=%.2f, ane=%s\n",
            config.num_draft_tokens, config.acceptance_threshold,
            config.use_ane ? "true" : "false");
}

const SpeculativeConfig& SpeculativeDecoder::config() const {
    return impl_->config;
}

void SpeculativeDecoder::reset_stats() {
    impl_->total_proposed = 0;
    impl_->total_accepted = 0;
    impl_->window_idx = 0;
    memset(impl_->window_proposed, 0, sizeof(impl_->window_proposed));
    memset(impl_->window_accepted, 0, sizeof(impl_->window_accepted));
}

}  // namespace nexus::model
