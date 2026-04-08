#pragma once
/// NEXUS Inference Engine — Phase 4: EAGLE-3 Speculative Decoding.
///
/// Speculative decoding uses a small "draft" model (running on the Apple Neural
/// Engine via CoreML) to propose candidate tokens, which the full model then
/// verifies in a single batched forward pass on the GPU.  Matching tokens are
/// accepted for free; the first mismatch is resampled from the main model.
///
/// Net effect: 2-3x tokens/sec improvement with ZERO quality loss, because the
/// output distribution is mathematically identical to standard autoregressive
/// decoding.

#include "core/config.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace nexus {

class MemoryManager;

namespace model {

class Transformer;

}  // namespace model

namespace compute {

class DraftModel;

}  // namespace compute

namespace model {

// ─── Configuration ──────────────────────────────────────────────────────────

struct SpeculativeConfig {
    int   num_draft_tokens     = 5;     // Number of tokens the draft model proposes per step
    float acceptance_threshold = 0.0f;  // Minimum probability ratio for acceptance (0 = standard)
    bool  use_ane              = true;  // Route draft model to Apple Neural Engine via CoreML
};

// ─── Speculative Decoder ────────────────────────────────────────────────────

class SpeculativeDecoder {
public:
    /// Construct with a main (verifier) model and a draft model.
    /// @param main_model  Full-size transformer for verification (runs on GPU).
    /// @param draft_model Lightweight model for candidate generation (runs on ANE).
    /// @param num_speculative_tokens  How many draft tokens to propose per step.
    SpeculativeDecoder(Transformer& main_model,
                       compute::DraftModel& draft_model,
                       int num_speculative_tokens = 5);
    ~SpeculativeDecoder();

    // Non-copyable.
    SpeculativeDecoder(const SpeculativeDecoder&) = delete;
    SpeculativeDecoder& operator=(const SpeculativeDecoder&) = delete;

    /// Run one speculative decoding step:
    ///   1. Draft model generates N candidate tokens (fast, on ANE).
    ///   2. Main model verifies all N+1 positions in one forward pass (GPU).
    ///   3. Accept matching tokens; reject and resample on first mismatch.
    ///
    /// Returns 1..N+1 accepted tokens.  The caller should append all of them
    /// to the sequence.
    std::vector<int32_t> generate_step(const std::vector<int32_t>& prompt_tokens,
                                       const SamplingParams& params);

    /// Rolling average acceptance rate (accepted / proposed).
    /// Useful for monitoring: a healthy draft model should achieve 0.6-0.8.
    float acceptance_rate() const;

    /// Update the speculative config at runtime (e.g. tune num_draft_tokens).
    void set_config(const SpeculativeConfig& config);
    const SpeculativeConfig& config() const;

    /// Reset internal statistics counters.
    void reset_stats();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace model
}  // namespace nexus
