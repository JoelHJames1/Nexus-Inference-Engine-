/// NEXUS Paged KV Attention — Implementation.
///
/// Paged KV cache with tiered TurboQuant compression.

#include "kv/paged_attention.h"
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <mutex>

namespace nexus::kv {

// ─── Construction / Destruction ─────────────────────────────────────────────

PagedKVCache::PagedKVCache(uint32_t num_layers, uint32_t num_kv_heads,
                           uint32_t head_dim, uint32_t max_seq_len,
                           uint32_t page_size)
    : num_layers_(num_layers)
    , num_kv_heads_(num_kv_heads)
    , head_dim_(head_dim)
    , max_seq_len_(max_seq_len)
    , page_size_(page_size) {

    max_pages_per_head_ = (max_seq_len + page_size - 1) / page_size;

    // Warm tier: 4-bit TurboQuant.
    quant_warm_ = std::make_unique<TurboQuantKV>(head_dim, 4, /*seed=*/42);

    // Cool tier: 3-bit TurboQuant.
    quant_cool_ = std::make_unique<TurboQuantKV>(head_dim, 3, /*seed=*/42);

    // Reserve scratch buffer for largest possible decompression.
    scratch_.resize(page_size * head_dim * num_kv_heads);
}

PagedKVCache::~PagedKVCache() {
    for (auto& [key, entry] : page_table_) {
        if (entry.data) {
            std::free(entry.data);
            entry.data = nullptr;
        }
    }
}

// ─── Page allocation ────────────────────────────────────────────────────────

void* PagedKVCache::alloc_hot_page(uint32_t page_size, uint32_t head_dim) {
    size_t buf_size = static_cast<size_t>(page_size) * head_dim * sizeof(float);
    // Align to system page size for efficient memory access.
    size_t aligned_size = (buf_size + nexus::kPageSize - 1) & ~(nexus::kPageSize - 1);
    void* ptr = aligned_alloc(nexus::kPageSize, aligned_size);
    if (ptr) std::memset(ptr, 0, aligned_size);
    return ptr;
}

PageEntry& PagedKVCache::get_or_create_page(const PageKey& key) {
    auto it = page_table_.find(key);
    if (it != page_table_.end()) {
        it->second.timestamp = ++timestamp_counter_;
        return it->second;
    }

    // Create new Hot page.
    PageEntry entry;
    entry.data = alloc_hot_page(page_size_, head_dim_);
    entry.data_size = static_cast<size_t>(page_size_) * head_dim_ * sizeof(float);
    entry.tier = CompressionTier::Hot;
    entry.ref_count = 0;
    entry.timestamp = ++timestamp_counter_;
    entry.n_tokens = 0;
    entry.is_key = key.is_key;

    auto [inserted_it, _] = page_table_.emplace(key, std::move(entry));
    return inserted_it->second;
}

// ─── Insert ─────────────────────────────────────────────────────────────────

void PagedKVCache::insert_kv(uint32_t layer, uint32_t seq_pos,
                              const float* key_data, const float* value_data) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (layer >= num_layers_ || seq_pos >= max_seq_len_) return;

    uint32_t page_idx = seq_pos / page_size_;
    uint32_t offset_in_page = seq_pos % page_size_;

    // Insert for each KV head.
    for (uint32_t h = 0; h < num_kv_heads_; ++h) {
        // Key page.
        {
            PageKey pk{layer, h, page_idx, /*is_key=*/true};
            PageEntry& page = get_or_create_page(pk);

            if (page.tier != CompressionTier::Hot) {
                // Cannot insert into compressed page — would need to decompress first.
                // For simplicity, skip (in production, decompress + recompress).
                continue;
            }

            float* dst = static_cast<float*>(page.data) + offset_in_page * head_dim_;
            std::memcpy(dst, key_data + h * head_dim_, head_dim_ * sizeof(float));

            if (offset_in_page + 1 > page.n_tokens) {
                page.n_tokens = offset_in_page + 1;
            }
        }

        // Value page.
        {
            PageKey pk{layer, h, page_idx, /*is_key=*/false};
            PageEntry& page = get_or_create_page(pk);

            if (page.tier != CompressionTier::Hot) {
                continue;
            }

            float* dst = static_cast<float*>(page.data) + offset_in_page * head_dim_;
            std::memcpy(dst, value_data + h * head_dim_, head_dim_ * sizeof(float));

            if (offset_in_page + 1 > page.n_tokens) {
                page.n_tokens = offset_in_page + 1;
            }
        }
    }
}

// ─── Decompression helpers ──────────────────────────────────────────────────

float* PagedKVCache::decompress_page(const PageEntry& page, int n_tokens) const {
    // scratch_ is pre-allocated large enough.
    float* out = scratch_.data();

    switch (page.tier) {
        case CompressionTier::Hot:
            // Already FP32 — just copy.
            std::memcpy(out, page.data, n_tokens * head_dim_ * sizeof(float));
            break;

        case CompressionTier::Warm:
            quant_warm_->dequantize_kv_page(out, page.compressed, n_tokens);
            break;

        case CompressionTier::Cool:
            quant_cool_->dequantize_kv_page(out, page.compressed, n_tokens);
            break;

        case CompressionTier::Cold:
            // Evicted — zero fill.
            std::memset(out, 0, n_tokens * head_dim_ * sizeof(float));
            break;
    }

    return out;
}

// ─── Get keys / values ──────────────────────────────────────────────────────

const float* PagedKVCache::get_vectors(uint32_t layer, uint32_t seq_start,
                                        uint32_t seq_len, bool is_key) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Ensure scratch buffer is large enough.
    size_t needed = static_cast<size_t>(seq_len) * head_dim_ * num_kv_heads_;
    if (scratch_.size() < needed) {
        scratch_.resize(needed);
    }

    float* output = scratch_.data();
    std::memset(output, 0, needed * sizeof(float));

    // Iterate over the requested range, gathering from pages.
    for (uint32_t h = 0; h < num_kv_heads_; ++h) {
        uint32_t pos = seq_start;
        uint32_t remaining = seq_len;

        while (remaining > 0) {
            uint32_t page_idx = pos / page_size_;
            uint32_t offset_in_page = pos % page_size_;
            uint32_t count = std::min(remaining, page_size_ - offset_in_page);

            PageKey pk{layer, h, page_idx, is_key};
            auto it = page_table_.find(pk);

            if (it != page_table_.end()) {
                PageEntry& page = it->second;
                page.timestamp = ++timestamp_counter_;
                page.ref_count++;

                if (page.tier == CompressionTier::Hot) {
                    // Direct copy from hot page.
                    const float* src = static_cast<const float*>(page.data)
                                       + offset_in_page * head_dim_;
                    for (uint32_t t = 0; t < count; ++t) {
                        float* dst = output + (pos - seq_start + t) * (head_dim_ * num_kv_heads_)
                                     + h * head_dim_;
                        std::memcpy(dst, src + t * head_dim_, head_dim_ * sizeof(float));
                    }
                } else if (page.tier != CompressionTier::Cold) {
                    // Decompress the entire page to a temporary buffer.
                    std::vector<float> temp(page.n_tokens * head_dim_);
                    if (page.tier == CompressionTier::Warm) {
                        quant_warm_->dequantize_kv_page(temp.data(), page.compressed,
                                                         page.n_tokens);
                    } else {
                        quant_cool_->dequantize_kv_page(temp.data(), page.compressed,
                                                         page.n_tokens);
                    }

                    for (uint32_t t = 0; t < count; ++t) {
                        float* dst = output + (pos - seq_start + t) * (head_dim_ * num_kv_heads_)
                                     + h * head_dim_;
                        std::memcpy(dst, temp.data() + (offset_in_page + t) * head_dim_,
                                    head_dim_ * sizeof(float));
                    }
                }
                // Cold tier: already zero-filled.

                page.ref_count--;
            }

            pos += count;
            remaining -= count;
        }
    }

    return scratch_.data();
}

const float* PagedKVCache::get_keys(uint32_t layer, uint32_t seq_start,
                                     uint32_t seq_len) {
    return get_vectors(layer, seq_start, seq_len, /*is_key=*/true);
}

const float* PagedKVCache::get_values(uint32_t layer, uint32_t seq_start,
                                       uint32_t seq_len) {
    return get_vectors(layer, seq_start, seq_len, /*is_key=*/false);
}

// ─── Tier compression ───────────────────────────────────────────────────────

void PagedKVCache::compress_tier(CompressionTier from_tier, CompressionTier to_tier) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (to_tier <= from_tier) return;
    if (to_tier == CompressionTier::Cold) {
        // Use evict_cold_pages() for Cold transition.
        return;
    }

    TurboQuantKV* quantizer = nullptr;
    if (to_tier == CompressionTier::Warm) {
        quantizer = quant_warm_.get();
    } else if (to_tier == CompressionTier::Cool) {
        quantizer = quant_cool_.get();
    }
    if (!quantizer) return;

    for (auto& [key, page] : page_table_) {
        if (page.tier != from_tier) continue;
        if (page.ref_count > 0) continue;  // Skip pages with active references.
        if (page.n_tokens == 0) continue;

        // Get FP32 data — either directly from Hot or by decompressing Warm.
        std::vector<float> fp32_data(page.n_tokens * head_dim_);

        if (page.tier == CompressionTier::Hot) {
            std::memcpy(fp32_data.data(), page.data,
                        page.n_tokens * head_dim_ * sizeof(float));
        } else if (page.tier == CompressionTier::Warm) {
            quant_warm_->dequantize_kv_page(fp32_data.data(), page.compressed,
                                             page.n_tokens);
        }

        // Quantize to target tier.
        CompressedVectors compressed;
        quantizer->quantize_kv_page(compressed, fp32_data.data(), page.n_tokens);

        // Free old hot data.
        if (page.data) {
            std::free(page.data);
            page.data = nullptr;
        }

        page.compressed = std::move(compressed);
        page.data_size = page.compressed.byte_size();
        page.tier = to_tier;
    }
}

// ─── Eviction ───────────────────────────────────────────────────────────────

void PagedKVCache::evict_cold_pages() {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = page_table_.begin();
    while (it != page_table_.end()) {
        if (it->second.tier == CompressionTier::Cold) {
            if (it->second.data) {
                std::free(it->second.data);
            }
            it = page_table_.erase(it);
        } else {
            ++it;
        }
    }
}

// ─── Statistics ─────────────────────────────────────────────────────────────

size_t PagedKVCache::memory_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t total = 0;
    for (const auto& [key, page] : page_table_) {
        if (page.tier == CompressionTier::Hot) {
            total += page.data_size;
        } else {
            total += page.compressed.byte_size();
        }
    }
    return total;
}

PagedKVCache::Stats PagedKVCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    Stats s{};
    for (const auto& [key, page] : page_table_) {
        size_t page_mem = 0;
        if (page.tier == CompressionTier::Hot) {
            page_mem = page.data_size;
        } else if (page.tier != CompressionTier::Cold) {
            page_mem = page.compressed.byte_size();
        }

        switch (page.tier) {
            case CompressionTier::Hot:
                s.hot.page_count++;
                s.hot.memory_bytes += page_mem;
                break;
            case CompressionTier::Warm:
                s.warm.page_count++;
                s.warm.memory_bytes += page_mem;
                break;
            case CompressionTier::Cool:
                s.cool.page_count++;
                s.cool.memory_bytes += page_mem;
                break;
            case CompressionTier::Cold:
                s.cold.page_count++;
                break;
        }
    }
    return s;
}

}  // namespace nexus::kv
