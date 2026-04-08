/// NEXUS MoE Layer — Sparse expert execution with SSD-streaming weight loading.
///
/// This is where the NEXUS architecture really shines for MoE models:
///
///   Traditional approach (llama.cpp): load all 256 experts into RAM.
///     For DeepSeek-V3 at FP16, that's ~400 GB of expert weights alone.
///
///   NEXUS approach: only 8 experts are active per token.  We cache
///     recently-used experts in RAM (typically 16-32 experts, ~25-50 GB)
///     and stream cache misses from SSD via mmap.  With Apple Silicon's
///     ~7.5 GB/s SSD throughput, loading one expert (~1.6 GB at FP16)
///     takes ~210ms — but cache hits are instant, and predictive prefetch
///     hides most of the latency for cache misses.

#include "model/moe_layer.h"
#include "memory/memory_manager.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

namespace nexus::model {

// ═══════════════════════════════════════════════════════════════════════════════
// ExpertCache
// ═══════════════════════════════════════════════════════════════════════════════

ExpertCache::ExpertCache(int max_experts, int hidden_dim, int ffn_dim)
    : max_experts_(max_experts)
    , hidden_dim_(hidden_dim)
    , ffn_dim_(ffn_dim)
{
}

ExpertCache::~ExpertCache() = default;

ExpertWeights* ExpertCache::get(int expert_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = cache_.find(expert_id);
    if (it == cache_.end()) return nullptr;

    // Promote to most-recently-used (move to back of LRU list)
    lru_list_.erase(it->second.lru_it);
    lru_list_.push_back(expert_id);
    it->second.lru_it = std::prev(lru_list_.end());

    return &it->second.weights;
}

ExpertWeights* ExpertCache::insert(int expert_id) {
    std::lock_guard<std::mutex> lock(mu_);

    // If already present, just promote and return
    auto existing = cache_.find(expert_id);
    if (existing != cache_.end()) {
        lru_list_.erase(existing->second.lru_it);
        lru_list_.push_back(expert_id);
        existing->second.lru_it = std::prev(lru_list_.end());
        return &existing->second.weights;
    }

    // Evict if at capacity
    while (static_cast<int>(cache_.size()) >= max_experts_ && !lru_list_.empty()) {
        int victim = lru_list_.front();
        lru_list_.pop_front();
        auto vit = cache_.find(victim);
        if (vit != cache_.end()) {
            // The caller is responsible for unmapping/freeing the weight
            // pointers if they were allocated.  With mmap-based loading,
            // the OS reclaims pages via MADV_DONTNEED or munmap.
            cache_.erase(vit);
        }
    }

    // Insert new entry at back of LRU (most recently used)
    lru_list_.push_back(expert_id);
    CacheEntry entry;
    entry.lru_it = std::prev(lru_list_.end());
    auto [it, _] = cache_.emplace(expert_id, std::move(entry));
    return &it->second.weights;
}

bool ExpertCache::contains(int expert_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    return cache_.count(expert_id) > 0;
}

void ExpertCache::evict(int expert_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = cache_.find(expert_id);
    if (it != cache_.end()) {
        lru_list_.erase(it->second.lru_it);
        cache_.erase(it);
    }
}

int ExpertCache::evict_lru() {
    std::lock_guard<std::mutex> lock(mu_);
    if (lru_list_.empty()) return -1;

    int victim = lru_list_.front();
    lru_list_.pop_front();
    cache_.erase(victim);
    return victim;
}

int ExpertCache::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(cache_.size());
}

size_t ExpertCache::expert_weight_bytes() const {
    // Three matrices: w1, w2, w3 each connecting hidden_dim <-> ffn_dim
    return static_cast<size_t>(hidden_dim_) * ffn_dim_ * sizeof(float) * 3;
}

// ═══════════════════════════════════════════════════════════════════════════════
// MoELayer
// ═══════════════════════════════════════════════════════════════════════════════

MoELayer::MoELayer(int num_experts, int num_active, int hidden_dim, int ffn_dim,
                   MemoryManager& memory, format::NXFReader& reader,
                   uint32_t layer_idx, int cache_capacity)
    : num_experts_(num_experts)
    , num_active_(num_active)
    , hidden_dim_(hidden_dim)
    , ffn_dim_(ffn_dim)
    , layer_idx_(layer_idx)
    , memory_(&memory)
    , reader_(&reader)
    , router_(num_experts, num_active, hidden_dim)
    , cache_(cache_capacity > 0 ? cache_capacity : num_active * 4,
             hidden_dim, ffn_dim)
    , gate_buf_(ffn_dim, 0.0f)
    , up_buf_(ffn_dim, 0.0f)
    , expert_out_(hidden_dim, 0.0f)
{
    fprintf(stderr, "[nexus] MoE layer %u: %d experts, %d active, cache=%d\n",
            layer_idx_, num_experts_, num_active_, cache_.capacity());
}

MoELayer::~MoELayer() = default;

// ─── Forward pass ──────────────────────────────────────────────────────────────

void MoELayer::forward(const float* hidden_state, float* output,
                       const TopKResult& router_result) {
    // Zero the output accumulator
    std::memset(output, 0, static_cast<size_t>(hidden_dim_) * sizeof(float));

    // Execute each active expert and accumulate weighted outputs
    for (const auto& [expert_id, weight] : router_result.expert_weights) {
        // 1. Ensure expert weights are loaded (cache hit or SSD load)
        ExpertWeights* ew = cache_.get(expert_id);
        if (!ew || !ew->loaded) {
            ew = load_expert(expert_id);
            if (!ew || !ew->loaded) {
                fprintf(stderr, "[nexus] WARNING: Failed to load expert %d for layer %u\n",
                        expert_id, layer_idx_);
                continue;
            }
        }

        // 2. Execute this expert's SwiGLU FFN
        execute_expert_ffn(hidden_state, expert_out_.data(), *ew);

        // 3. Accumulate: output += weight * expert_output
        for (int i = 0; i < hidden_dim_; i++) {
            output[i] += weight * expert_out_[i];
        }
    }

    // 4. Trigger predictive prefetch for the next token.
    //    We use the router's EMA-based prediction (the current gate logits
    //    have already been folded into the EMA during route()).
    std::vector<int> predicted = router_.predict_next_experts(nullptr, num_active_ * 2);
    prefetch_experts(predicted);
}

// ─── Predictive prefetch ───────────────────────────────────────────────────────

void MoELayer::prefetch_experts(const std::vector<int>& expert_ids) {
    for (int eid : expert_ids) {
        // Skip experts that are already cached
        if (cache_.contains(eid)) continue;

        // Issue an advisory read-ahead for this expert's weight tensors.
        // This tells the OS to start paging in the mmap'd regions from SSD,
        // so they'll be in the unified memory page cache when we need them.
        const char* suffixes[] = {
            "feed_forward.w1.weight",
            "feed_forward.w2.weight",
            "feed_forward.w3.weight",
        };

        for (const char* suffix : suffixes) {
            std::string tname = expert_tensor_name(layer_idx_, eid, suffix);
            const auto* info = reader_->get_tensor(tname);
            if (info && !info->chunks.empty()) {
                // Use F_RDADVISE to hint the OS about upcoming reads.
                // This is non-blocking — the OS pages data in asynchronously.
                for (const auto& chunk : info->chunks) {
                    memory_->prefetch(reader_->fd(), chunk.file_offset,
                                     chunk.compressed_size);
                }
            }
        }
    }
}

// ─── Expert weight loading ─────────────────────────────────────────────────────

ExpertWeights* MoELayer::load_expert(int expert_id) {
    // Allocate a cache slot (may evict LRU expert)
    ExpertWeights* ew = cache_.insert(expert_id);
    if (!ew) return nullptr;

    // Load each weight matrix from NXF
    auto load_tensor = [&](const char* suffix, float*& ptr) -> bool {
        std::string tname = expert_tensor_name(layer_idx_, expert_id, suffix);
        const auto* info = reader_->get_tensor(tname);
        if (!info || info->chunks.empty()) return false;

        // Memory-map the chunk from the NXF file.
        // With unified memory on Apple Silicon, this gives us a pointer into
        // the page cache — data is faulted in from SSD on first access.
        const void* mapped = reader_->map_chunk(info->chunks[0]);
        if (!mapped) return false;

        // For FP16/quantized codecs, we'd dequantize into a scratch buffer here.
        // For now, treat as FP32 pointer (same pattern as transformer.cpp).
        ptr = const_cast<float*>(static_cast<const float*>(mapped));
        return true;
    };

    bool ok = true;
    ok &= load_tensor("feed_forward.w1.weight", ew->w1);
    ok &= load_tensor("feed_forward.w2.weight", ew->w2);
    ok &= load_tensor("feed_forward.w3.weight", ew->w3);
    ew->loaded = ok;

    if (ok) {
        fprintf(stderr, "[nexus] Loaded expert %d weights for layer %u\n",
                expert_id, layer_idx_);
    }

    return ew;
}

// ─── Expert FFN execution ──────────────────────────────────────────────────────

void MoELayer::execute_expert_ffn(const float* input, float* output,
                                   const ExpertWeights& weights) {
    // SwiGLU: output = (silu(input @ W1) * (input @ W3)) @ W2
    //
    // W1 (gate):  [hidden_dim, ffn_dim]  — gate projection
    // W3 (up):    [hidden_dim, ffn_dim]  — up projection
    // W2 (down):  [ffn_dim, hidden_dim]  — down projection

    float* gate = gate_buf_.data();
    float* up   = up_buf_.data();

    // Gate projection: gate = input @ W1^T
    // Manual GEMV — in production, use Accelerate cblas_sgemv or Metal.
    if (weights.w1) {
        for (int j = 0; j < ffn_dim_; j++) {
            float dot = 0.0f;
            const float* w1_row = weights.w1 + static_cast<size_t>(j) * hidden_dim_;
            int i = 0;
            for (; i + 3 < hidden_dim_; i += 4) {
                dot += w1_row[i]     * input[i]
                     + w1_row[i + 1] * input[i + 1]
                     + w1_row[i + 2] * input[i + 2]
                     + w1_row[i + 3] * input[i + 3];
            }
            for (; i < hidden_dim_; i++) {
                dot += w1_row[i] * input[i];
            }
            gate[j] = dot;
        }
    } else {
        std::memset(gate, 0, static_cast<size_t>(ffn_dim_) * sizeof(float));
    }

    // Up projection: up = input @ W3^T
    if (weights.w3) {
        for (int j = 0; j < ffn_dim_; j++) {
            float dot = 0.0f;
            const float* w3_row = weights.w3 + static_cast<size_t>(j) * hidden_dim_;
            int i = 0;
            for (; i + 3 < hidden_dim_; i += 4) {
                dot += w3_row[i]     * input[i]
                     + w3_row[i + 1] * input[i + 1]
                     + w3_row[i + 2] * input[i + 2]
                     + w3_row[i + 3] * input[i + 3];
            }
            for (; i < hidden_dim_; i++) {
                dot += w3_row[i] * input[i];
            }
            up[j] = dot;
        }
    } else {
        std::memset(up, 0, static_cast<size_t>(ffn_dim_) * sizeof(float));
    }

    // SiLU activation on gate, then elementwise multiply with up
    for (int i = 0; i < ffn_dim_; i++) {
        float silu = gate[i] / (1.0f + std::exp(-gate[i]));
        gate[i] = silu * up[i];
    }

    // Down projection: output = gate @ W2^T
    // W2 is [ffn_dim, hidden_dim], so output[i] = dot(gate, W2[:, i])
    // Equivalently with row-major W2: output[i] = sum_j gate[j] * W2[j * hidden_dim + i]
    if (weights.w2) {
        for (int i = 0; i < hidden_dim_; i++) {
            float dot = 0.0f;
            int j = 0;
            for (; j + 3 < ffn_dim_; j += 4) {
                dot += gate[j]     * weights.w2[static_cast<size_t>(j)     * hidden_dim_ + i]
                     + gate[j + 1] * weights.w2[static_cast<size_t>(j + 1) * hidden_dim_ + i]
                     + gate[j + 2] * weights.w2[static_cast<size_t>(j + 2) * hidden_dim_ + i]
                     + gate[j + 3] * weights.w2[static_cast<size_t>(j + 3) * hidden_dim_ + i];
            }
            for (; j < ffn_dim_; j++) {
                dot += gate[j] * weights.w2[static_cast<size_t>(j) * hidden_dim_ + i];
            }
            output[i] = dot;
        }
    } else {
        std::memset(output, 0, static_cast<size_t>(hidden_dim_) * sizeof(float));
    }
}

// ─── Tensor name construction ──────────────────────────────────────────────────

std::string MoELayer::expert_tensor_name(uint32_t layer, int expert_id,
                                          const char* suffix) {
    // Naming convention matching common MoE model formats:
    // "layers.{layer}.experts.{expert_id}.{suffix}"
    char buf[256];
    std::snprintf(buf, sizeof(buf), "layers.%u.experts.%d.%s",
                  layer, expert_id, suffix);
    return std::string(buf);
}

// ─── Accessors ─────────────────────────────────────────────────────────────────

int MoELayer::cached_experts() const {
    return cache_.size();
}

int MoELayer::cache_capacity() const {
    return cache_.capacity();
}

}  // namespace nexus::model
