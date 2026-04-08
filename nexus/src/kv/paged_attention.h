#pragma once
/// NEXUS Paged KV Attention — Tiered compression with TurboQuant.
///
/// Phase 3: Paged KV cache with automatic compression tiering.
/// Pages transition: Hot (FP16) -> Warm (3.5-bit) -> Cool (2.5-bit) -> Cold (evicted).

#include "core/config.h"
#include "kv/turbo_quant.h"
#include <cstdint>
#include <cstddef>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>

namespace nexus::kv {

/// Compression tiers for KV cache pages.
enum class CompressionTier : uint8_t {
    Hot  = 0,   // FP16 — active attention window
    Warm = 1,   // TurboQuant ~3.5-bit (4-bit codes)
    Cool = 2,   // TurboQuant ~2.5-bit (3-bit codes)
    Cold = 3,   // Evicted / spilled to SSD
};

/// A single page of KV cache data.
struct PageEntry {
    void* data = nullptr;           // Raw data pointer (FP16 or compressed)
    size_t data_size = 0;           // Size of allocated data in bytes
    CompressionTier tier = CompressionTier::Hot;
    uint32_t ref_count = 0;         // Number of active references
    uint64_t timestamp = 0;         // Last access timestamp (monotonic)
    uint32_t n_tokens = 0;          // Number of valid tokens in this page
    bool is_key = true;             // true = key page, false = value page

    /// Compressed data (populated when tier != Hot).
    CompressedVectors compressed;
};

/// Key for the page table: (layer, head_group, page_index).
struct PageKey {
    uint32_t layer;
    uint32_t head_group;
    uint32_t page_index;
    bool is_key;  // true = key page, false = value page

    bool operator==(const PageKey& o) const {
        return layer == o.layer && head_group == o.head_group &&
               page_index == o.page_index && is_key == o.is_key;
    }
};

/// Hash for PageKey.
struct PageKeyHash {
    size_t operator()(const PageKey& k) const {
        // Combine fields into a single hash.
        size_t h = std::hash<uint32_t>{}(k.layer);
        h ^= std::hash<uint32_t>{}(k.head_group) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>{}(k.page_index) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>{}(k.is_key) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

/// Per-tier statistics.
struct TierStats {
    uint32_t page_count = 0;
    size_t memory_bytes = 0;
};

/// Paged KV cache with automatic compression tiering.
///
/// KV vectors are stored in fixed-size pages. As tokens age out of the
/// active attention window, their pages are progressively compressed:
///   Hot (FP16) -> Warm (TurboQuant 4-bit) -> Cool (TurboQuant 3-bit) -> Cold (evicted).
class PagedKVCache {
public:
    /// @param num_layers    Number of transformer layers
    /// @param num_kv_heads  Number of KV attention heads
    /// @param head_dim      Dimension per head
    /// @param max_seq_len   Maximum sequence length
    /// @param page_size     Tokens per page (default 256)
    PagedKVCache(uint32_t num_layers, uint32_t num_kv_heads, uint32_t head_dim,
                 uint32_t max_seq_len, uint32_t page_size = 256);
    ~PagedKVCache();

    // Non-copyable.
    PagedKVCache(const PagedKVCache&) = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;

    /// Insert one token's KV pair into the cache.
    ///
    /// @param layer     Transformer layer index
    /// @param seq_pos   Sequence position
    /// @param key_data  Key vector [head_dim * num_kv_heads] (FP32)
    /// @param value_data Value vector [head_dim * num_kv_heads] (FP32)
    void insert_kv(uint32_t layer, uint32_t seq_pos,
                   const float* key_data, const float* value_data);

    /// Get contiguous key vectors for a range.
    /// Dequantizes compressed pages on the fly.
    ///
    /// @param layer      Transformer layer
    /// @param seq_start  Starting sequence position
    /// @param seq_len    Number of tokens
    /// @return Pointer to FP32 key data [seq_len * head_dim * num_kv_heads].
    ///         Caller does NOT own this pointer; it is valid until next insert/compress.
    const float* get_keys(uint32_t layer, uint32_t seq_start, uint32_t seq_len);

    /// Get contiguous value vectors for a range (same semantics as get_keys).
    const float* get_values(uint32_t layer, uint32_t seq_start, uint32_t seq_len);

    /// Compress all pages in one tier to the next tier.
    ///
    /// @param from_tier  Source tier
    /// @param to_tier    Destination tier (must be > from_tier)
    void compress_tier(CompressionTier from_tier, CompressionTier to_tier);

    /// Evict all cold-tier pages (free memory).
    void evict_cold_pages();

    /// Total memory used by all pages.
    size_t memory_bytes() const;

    /// Per-tier statistics.
    struct Stats {
        TierStats hot;
        TierStats warm;
        TierStats cool;
        TierStats cold;
    };
    Stats stats() const;

    uint32_t num_layers() const { return num_layers_; }
    uint32_t num_kv_heads() const { return num_kv_heads_; }
    uint32_t head_dim() const { return head_dim_; }
    uint32_t page_size() const { return page_size_; }

private:
    uint32_t num_layers_;
    uint32_t num_kv_heads_;
    uint32_t head_dim_;
    uint32_t max_seq_len_;
    uint32_t page_size_;
    uint32_t max_pages_per_head_;  // ceil(max_seq_len / page_size)

    /// TurboQuant codecs for Warm (4-bit) and Cool (3-bit) tiers.
    std::unique_ptr<TurboQuantKV> quant_warm_;   // 4-bit
    std::unique_ptr<TurboQuantKV> quant_cool_;   // 3-bit

    /// Page table: maps PageKey -> PageEntry.
    std::unordered_map<PageKey, PageEntry, PageKeyHash> page_table_;

    /// Monotonic timestamp counter.
    uint64_t timestamp_counter_ = 0;

    /// Scratch buffer for decompression.
    mutable std::vector<float> scratch_;

    /// Mutex for thread safety.
    mutable std::mutex mutex_;

    /// Get or create a page for a given key.
    PageEntry& get_or_create_page(const PageKey& key);

    /// Decompress a page into the scratch buffer, returning a pointer.
    float* decompress_page(const PageEntry& page, int n_tokens) const;

    /// Get contiguous vectors (shared logic for keys/values).
    const float* get_vectors(uint32_t layer, uint32_t seq_start,
                             uint32_t seq_len, bool is_key);

    /// Allocate a Hot (FP16-stored-as-FP32) page buffer.
    static void* alloc_hot_page(uint32_t page_size, uint32_t head_dim);
};

}  // namespace nexus::kv
