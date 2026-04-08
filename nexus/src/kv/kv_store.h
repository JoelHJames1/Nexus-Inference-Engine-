#pragma once
/// NEXUS KV Store — Paged, compressed KV cache with prefix indexing.
///
/// Phase 1: Basic flat KV cache (this file)
/// Phase 3: Full paged attention + TurboQuant compression + radix prefix tree

#include "core/config.h"
#include <cstdint>
#include <cstddef>
#include <vector>

namespace nexus::kv {

/// Basic KV store interface (Phase 1: flat buffer).
/// Phase 3 will replace this with paged TurboQuant-compressed storage.
class KVStore {
public:
    KVStore(uint32_t num_layers, uint32_t num_kv_heads, uint32_t head_dim, uint32_t max_seq_len);
    ~KVStore();

    /// Get key buffer for a layer at a sequence position.
    float* key_at(uint32_t layer, uint32_t seq_pos);

    /// Get value buffer for a layer at a sequence position.
    float* value_at(uint32_t layer, uint32_t seq_pos);

    /// Current sequence length for a layer.
    uint32_t seq_len(uint32_t layer) const;

    /// Advance sequence length for a layer.
    void advance(uint32_t layer);

    /// Reset all caches.
    void reset();

    /// Total memory used by KV cache in bytes.
    size_t memory_bytes() const;

private:
    uint32_t num_layers_;
    uint32_t num_kv_heads_;
    uint32_t head_dim_;
    uint32_t max_seq_len_;
    uint32_t kv_dim_;  // num_kv_heads * head_dim

    struct LayerCache {
        float* keys = nullptr;
        float* values = nullptr;
        uint32_t seq_len = 0;
    };
    std::vector<LayerCache> layers_;
};

}  // namespace nexus::kv
