#pragma once
/// NEXUS MoE Layer — Sparse expert execution with SSD-streaming weight loading.
///
/// Key insight for NEXUS on Apple Silicon:
///   A 671B MoE model (DeepSeek-V3) has 256 experts but activates only 8 per
///   token.  That means only ~37B of expert weights are needed per forward pass.
///   With our streaming architecture, expert weights live on SSD and are loaded
///   into RAM on demand via mmap.  An LRU cache keeps recently-used experts
///   resident, and the MoERouter's predict_next_experts() drives prefetching
///   so that the next token's experts are already loading while the current
///   token computes.
///
/// Memory lifecycle of expert weights:
///   1. Router selects top-k experts for current token
///   2. ExpertCache checks if each expert's weights are resident
///   3. Cache misses trigger mmap load from NXF (SSD -> unified memory)
///   4. Expert FFN executes on cached weights
///   5. LRU eviction frees least-recently-used experts when cache is full
///   6. Predicted next-experts are prefetched in background

#include "model/moe_router.h"
#include "format/nxf.h"
#include "core/config.h"
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace nexus {
class MemoryManager;
}

namespace nexus::model {

// ─── Expert weights ────────────────────────────────────────────────────────────

/// Weights for a single expert's FFN (SwiGLU architecture).
struct ExpertWeights {
    float* w1 = nullptr;   // Gate projection   [hidden_dim, ffn_dim]
    float* w2 = nullptr;   // Down projection   [ffn_dim, hidden_dim]
    float* w3 = nullptr;   // Up projection     [hidden_dim, ffn_dim]
    bool   loaded = false;

    /// Total bytes for this expert's weights (at FP32).
    size_t weight_bytes(int hidden_dim, int ffn_dim) const {
        // w1: hidden_dim * ffn_dim, w2: ffn_dim * hidden_dim, w3: hidden_dim * ffn_dim
        return static_cast<size_t>(hidden_dim) * ffn_dim * sizeof(float) * 3;
    }
};

// ─── LRU Expert Cache ──────────────────────────────────────────────────────────

/// LRU cache for expert weight matrices.
///
/// Tracks which experts are currently resident in memory and evicts the
/// least-recently-used expert when the cache reaches capacity.  This is
/// critical for MoE models where only a fraction of experts are active
/// per token but the full expert set far exceeds available RAM.
class ExpertCache {
public:
    /// \param max_experts  Maximum number of experts to keep resident.
    /// \param hidden_dim   Hidden dimension (for weight size calculation).
    /// \param ffn_dim      FFN intermediate dimension.
    ExpertCache(int max_experts, int hidden_dim, int ffn_dim);
    ~ExpertCache();

    ExpertCache(const ExpertCache&) = delete;
    ExpertCache& operator=(const ExpertCache&) = delete;

    /// Get expert weights if cached.  Returns nullptr if not resident.
    /// Promotes the expert to most-recently-used on hit.
    ExpertWeights* get(int expert_id);

    /// Insert expert weights into cache.  May evict LRU expert if full.
    /// Returns pointer to the cache entry (caller populates the weight pointers).
    ExpertWeights* insert(int expert_id);

    /// Check if an expert is currently cached (without promoting it).
    bool contains(int expert_id) const;

    /// Evict a specific expert from cache.  No-op if not present.
    void evict(int expert_id);

    /// Evict the least-recently-used expert.  Returns the evicted expert_id,
    /// or -1 if cache is empty.
    int evict_lru();

    /// Number of experts currently cached.
    int size() const;

    /// Maximum capacity.
    int capacity() const { return max_experts_; }

    /// Per-expert weight size in bytes (for memory accounting).
    size_t expert_weight_bytes() const;

private:
    int max_experts_;
    int hidden_dim_;
    int ffn_dim_;

    /// LRU list: front = least recently used, back = most recently used.
    using LRUList = std::list<int>;  // expert_id
    LRUList lru_list_;

    /// Map from expert_id to (weights, LRU iterator).
    struct CacheEntry {
        ExpertWeights weights;
        LRUList::iterator lru_it;
    };
    std::unordered_map<int, CacheEntry> cache_;

    mutable std::mutex mu_;
};

// ─── MoE Layer ─────────────────────────────────────────────────────────────────

/// Mixture-of-Experts layer that replaces the standard FFN in MoE models.
///
/// For each token:
///   1. The MoERouter determines which experts to activate
///   2. This layer loads the active experts' weights (from cache or SSD)
///   3. Each active expert runs its SwiGLU FFN on the token
///   4. Expert outputs are combined via the router's gating weights
///   5. Unused experts may be evicted; predicted next-experts are prefetched
class MoELayer {
public:
    /// \param num_experts     Total experts in this layer (e.g., 256).
    /// \param num_active      Experts activated per token (e.g., 8).
    /// \param hidden_dim      Token hidden state dimension.
    /// \param ffn_dim         FFN intermediate dimension per expert.
    /// \param memory          Memory manager for allocation and prefetch.
    /// \param reader          NXF reader for loading expert weights from SSD.
    /// \param layer_idx       This layer's index (for weight tensor names).
    /// \param cache_capacity  Max experts to keep resident (0 = 2x num_active).
    MoELayer(int num_experts, int num_active, int hidden_dim, int ffn_dim,
             MemoryManager& memory, format::NXFReader& reader,
             uint32_t layer_idx, int cache_capacity = 0);
    ~MoELayer();

    MoELayer(const MoELayer&) = delete;
    MoELayer& operator=(const MoELayer&) = delete;

    /// Execute the MoE layer on a single token's hidden state.
    ///
    /// \param hidden_state   Input: token hidden state [hidden_dim].
    /// \param output         Output: MoE result [hidden_dim].
    /// \param router_result  Top-k routing decision from MoERouter.
    void forward(const float* hidden_state, float* output,
                 const TopKResult& router_result);

    /// Trigger predictive prefetch of expert weights.
    /// Should be called after routing but potentially before forward().
    void prefetch_experts(const std::vector<int>& expert_ids);

    /// Get the router (owned by this layer for convenience).
    MoERouter& router() { return router_; }
    const MoERouter& router() const { return router_; }

    /// Get cache occupancy statistics.
    int cached_experts() const;
    int cache_capacity() const;

private:
    int num_experts_;
    int num_active_;
    int hidden_dim_;
    int ffn_dim_;
    uint32_t layer_idx_;

    MemoryManager* memory_;
    format::NXFReader* reader_;

    MoERouter router_;
    ExpertCache cache_;

    /// Scratch buffers for expert computation (reused across experts).
    std::vector<float> gate_buf_;   // [ffn_dim]
    std::vector<float> up_buf_;     // [ffn_dim]
    std::vector<float> expert_out_; // [hidden_dim]

    /// Load a single expert's weights from NXF into cache.
    /// Returns pointer to the cached ExpertWeights, or nullptr on failure.
    ExpertWeights* load_expert(int expert_id);

    /// Execute one expert's SwiGLU FFN: out = (silu(x @ W1) * (x @ W3)) @ W2
    void execute_expert_ffn(const float* input, float* output,
                            const ExpertWeights& weights);

    /// Build the NXF tensor name for an expert weight matrix.
    /// e.g., "layers.12.experts.42.feed_forward.w1.weight"
    static std::string expert_tensor_name(uint32_t layer, int expert_id,
                                          const char* suffix);
};

}  // namespace nexus::model
