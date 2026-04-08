#pragma once
/// NEXUS Inference Engine — Phase 4: CoreML Draft Model for ANE.
///
/// Wraps a compiled CoreML model (.mlmodelc) to run lightweight draft-token
/// generation on Apple's Neural Engine, freeing the GPU for the main model.
///
/// This is a pure C++ header.  The Objective-C++ CoreML interop lives in the
/// corresponding .mm translation unit behind an opaque pointer.

#include "core/config.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nexus {

class MemoryManager;

namespace model {
class Transformer;
}

namespace compute {

/// Configuration for the draft model backend.
struct DraftModelConfig {
    std::string coreml_model_path;  // Path to compiled .mlmodelc bundle
    int         max_seq_len = 2048;
    int         vocab_size  = 128256;   // LLaMA-3 default
    int         hidden_dim  = 1024;     // Draft model hidden dimension
    int         num_layers  = 6;        // Draft model depth
    bool        use_ane     = true;     // Route to Neural Engine (else CPU+GPU)
};

/// Lightweight draft model for speculative decoding candidate generation.
///
/// Two backends:
///   1. CoreML (preferred) — runs on the Apple Neural Engine via MLModel API.
///   2. Fallback — uses a small NEXUS Transformer on CPU when no .mlmodelc
///      is available, allowing the system to work without a converted model.
class DraftModel {
public:
    /// Load a CoreML draft model from a compiled .mlmodelc path.
    /// Falls back to a small CPU transformer if the path is empty or invalid.
    explicit DraftModel(const DraftModelConfig& config);
    ~DraftModel();

    // Non-copyable, movable.
    DraftModel(const DraftModel&) = delete;
    DraftModel& operator=(const DraftModel&) = delete;
    DraftModel(DraftModel&&) noexcept;
    DraftModel& operator=(DraftModel&&) noexcept;

    /// Returns true if the CoreML model loaded successfully (ANE path).
    bool is_coreml_active() const;

    /// Returns true if any backend (CoreML or fallback) is ready.
    bool is_ready() const;

    /// Run a single forward pass through the draft model.
    /// @param input_tokens  Token IDs to process.
    /// @param seq_len       Current sequence length (for KV cache positioning).
    /// @return Logits vector of size vocab_size for the last position.
    std::vector<float> predict(const std::vector<int32_t>& input_tokens, int seq_len);

    /// Autoregressively generate n_tokens draft candidates.
    /// @param prompt       Context tokens (already processed by the main model).
    /// @param n_tokens     Number of draft tokens to generate.
    /// @param params       Sampling parameters (typically greedy for drafts).
    /// @return Vector of n_tokens draft token IDs.
    std::vector<int32_t> generate_draft(const std::vector<int32_t>& prompt,
                                        int n_tokens,
                                        const SamplingParams& params);

    /// Reset any internal state (KV cache) for a new sequence.
    void reset();

    /// Vocab size of the draft model.
    int vocab_size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace compute
}  // namespace nexus
