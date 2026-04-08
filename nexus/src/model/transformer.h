#pragma once
/// NEXUS Transformer — Layer-by-layer streaming transformer execution.
///
/// Unlike llama.cpp which loads all weights into memory, NEXUS streams
/// weights layer-by-layer from the NXF file, computing and evicting
/// as it goes. Only 2-3 layers are resident at any time.

#include "core/config.h"
#include "format/nxf.h"
#include <memory>
#include <vector>

namespace nexus {

class MemoryManager;

namespace model {

/// Represents the weights for a single transformer layer.
struct LayerWeights {
    // Attention
    float* wq = nullptr;       // Query projection  [hidden_dim, hidden_dim]
    float* wk = nullptr;       // Key projection    [hidden_dim, kv_dim]
    float* wv = nullptr;       // Value projection   [hidden_dim, kv_dim]
    float* wo = nullptr;       // Output projection  [hidden_dim, hidden_dim]

    // FFN (SwiGLU)
    float* w1 = nullptr;       // Gate projection    [hidden_dim, ffn_dim]
    float* w2 = nullptr;       // Down projection    [ffn_dim, hidden_dim]
    float* w3 = nullptr;       // Up projection      [hidden_dim, ffn_dim]

    // Norms
    float* attn_norm = nullptr; // RMSNorm weight [hidden_dim]
    float* ffn_norm = nullptr;  // RMSNorm weight [hidden_dim]

    // Quantization metadata (if quantized)
    float* scales = nullptr;
    float* zeros = nullptr;
    Codec  codec = Codec::FP16;

    bool loaded = false;
};

/// KV cache for a single layer.
struct LayerKVCache {
    float* keys = nullptr;     // [max_seq, num_kv_heads, head_dim]
    float* values = nullptr;   // [max_seq, num_kv_heads, head_dim]
    int    seq_len = 0;        // Current sequence length
};

/// Streaming transformer model.
class Transformer {
public:
    ~Transformer();

    /// Create from NXF model file.
    static std::unique_ptr<Transformer> create(
        const format::ModelManifest& manifest,
        format::NXFReader& reader,
        MemoryManager& memory
    );

    /// Prefill: process all prompt tokens at once.
    void prefill(const std::vector<int32_t>& tokens);

    /// Decode one step: compute next token logits and sample.
    int32_t decode_step(const SamplingParams& params);

    /// Reset KV cache for new conversation.
    void reset_kv_cache();

    /// Current sequence length.
    int seq_len() const;

private:
    Transformer() = default;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace model
}  // namespace nexus
