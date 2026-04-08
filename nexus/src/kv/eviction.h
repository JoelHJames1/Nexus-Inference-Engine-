#pragma once
/// NEXUS KV Eviction Strategies — H2O, SnapKV, and combined eviction manager.
///
/// H2O  (Heavy-Hitter Oracle): keeps initial tokens, recent window, and the
///       tokens with the highest cumulative attention scores.
///
/// SnapKV: uses an observation window of recent tokens to identify which KV
///         positions each head consistently attends to, then keeps those plus
///         a recent window.
///
/// EvictionManager: facade that selects / combines the above strategies and
///                  can run eviction asynchronously on a background thread.
///
/// Phase 3 component.

#include "core/config.h"
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace nexus::kv {

// ─── H2O Eviction ──────────────────────────────────────────────────────────────

/// Heavy-Hitter Oracle eviction.
/// Accumulates attention scores across layers and keeps:
///   - the first `initial_tokens` positions  (typically system-prompt tokens)
///   - the last `recent_window` positions     (sliding window)
///   - the top heavy hitters by cumulative score
class H2OEviction {
public:
    /// @param max_seq_len       maximum sequence length the model supports
    /// @param initial_tokens    number of initial (system-prompt) positions to always keep
    /// @param recent_window     number of most-recent positions to always keep
    H2OEviction(uint32_t max_seq_len,
                uint32_t initial_tokens = 4,
                uint32_t recent_window  = 256);
    ~H2OEviction() = default;

    /// Accumulate attention weights for one layer.
    /// `attention_weights` has `seq_len` elements representing the (averaged-
    /// over-heads) attention each new token pays to every prior position.
    void update_scores(uint32_t layer,
                       const float* attention_weights,
                       uint32_t seq_len);

    /// Return the token positions to *keep*, up to `budget` positions.
    /// Positions are sorted ascending.
    std::vector<int> select_keep(uint32_t budget) const;

    /// Return the token positions to *evict* so that at most `budget` remain.
    std::vector<int> select_evict(uint32_t budget) const;

    /// Reset all accumulated scores.
    void reset();

    /// Current sequence length being tracked.
    uint32_t current_seq_len() const { return current_seq_len_; }

private:
    uint32_t max_seq_len_;
    uint32_t initial_tokens_;
    uint32_t recent_window_;
    uint32_t current_seq_len_ = 0;

    /// Cumulative attention score per token position (summed across layers).
    std::vector<double> cumulative_scores_;
};

// ─── SnapKV Eviction ───────────────────────────────────────────────────────────

/// SnapKV eviction.
/// Uses an observation window of the most-recent `obs_window` tokens to measure
/// which prior KV positions each attention head consistently focuses on.
class SnapKVEviction {
public:
    /// @param max_seq_len   maximum sequence length
    /// @param num_layers    number of transformer layers
    /// @param num_heads     number of attention heads per layer
    /// @param obs_window    observation window size (default 64)
    /// @param recent_window recent positions always kept
    SnapKVEviction(uint32_t max_seq_len,
                   uint32_t num_layers,
                   uint32_t num_heads,
                   uint32_t obs_window    = 64,
                   uint32_t recent_window = 256);
    ~SnapKVEviction() = default;

    /// Observe attention weights for a single (layer, head).
    /// `attention_weights` has `seq_len` elements — the attention distribution
    /// over all prior positions from one query in the observation window.
    void observe(uint32_t layer,
                 uint32_t head,
                 const float* attention_weights,
                 uint32_t seq_len);

    /// Return positions to keep within `budget`, sorted ascending.
    std::vector<int> select_keep(uint32_t budget) const;

    /// Reset all observations.
    void reset();

    uint32_t current_seq_len() const { return current_seq_len_; }

private:
    uint32_t max_seq_len_;
    uint32_t num_layers_;
    uint32_t num_heads_;
    uint32_t obs_window_;
    uint32_t recent_window_;
    uint32_t current_seq_len_ = 0;

    /// Per-position importance vote count accumulated across all (layer, head)
    /// observations.  Higher value = more heads consistently attend to it.
    std::vector<double> importance_votes_;

    /// Number of observe() calls so far (for averaging).
    uint64_t observe_count_ = 0;
};

// ─── Eviction strategy enum ────────────────────────────────────────────────────

enum class EvictionStrategy : uint8_t {
    H2O      = 0,
    SnapKV   = 1,
    Combined = 2,   ///< Union of H2O and SnapKV keep-sets.
    LRU      = 3,   ///< Simple least-recently-used (timestamp-based).
};

// ─── Eviction manager ──────────────────────────────────────────────────────────

/// Facade that owns the concrete eviction strategies and can run eviction on a
/// background thread.
class EvictionManager {
public:
    /// @param strategy        which eviction strategy to use
    /// @param max_seq_len     maximum sequence length
    /// @param num_layers      transformer layer count
    /// @param num_heads       attention head count
    /// @param initial_tokens  H2O: positions always kept at the start
    /// @param recent_window   positions always kept at the end
    EvictionManager(EvictionStrategy strategy,
                    uint32_t max_seq_len,
                    uint32_t num_layers,
                    uint32_t num_heads,
                    uint32_t initial_tokens = 4,
                    uint32_t recent_window  = 256);
    ~EvictionManager();

    // ── Observation / scoring ───────────────────────────────────────────────

    /// Feed H2O-style per-layer attention weights.
    void update_h2o_scores(uint32_t layer,
                           const float* attention_weights,
                           uint32_t seq_len);

    /// Feed SnapKV-style per-head attention weights.
    void observe_snapkv(uint32_t layer,
                        uint32_t head,
                        const float* attention_weights,
                        uint32_t seq_len);

    // ── Eviction decisions ──────────────────────────────────────────────────

    /// Returns true if current memory exceeds the budget and eviction should run.
    bool should_evict(size_t current_memory, size_t memory_budget) const;

    /// Compute eviction candidates: positions to *remove* so that at most
    /// `budget` positions remain.
    std::vector<int> get_eviction_candidates(uint32_t budget);

    // ── Async eviction ──────────────────────────────────────────────────────

    /// Callback invoked when the background thread finishes computing
    /// eviction candidates.  Receives the list of positions to evict.
    using EvictionCallback = std::function<void(std::vector<int>)>;

    /// Request eviction on the background thread.  `callback` is called from
    /// the background thread when the result is ready.
    void request_async_eviction(uint32_t budget, EvictionCallback callback);

    /// Block until any pending async eviction completes.
    void wait_for_pending();

    /// Reset all internal state.
    void reset();

    /// Active strategy.
    EvictionStrategy strategy() const { return strategy_; }

private:
    void background_loop();

    EvictionStrategy strategy_;
    std::unique_ptr<H2OEviction>    h2o_;
    std::unique_ptr<SnapKVEviction> snapkv_;

    // LRU state (simple per-position timestamp).
    std::vector<uint64_t> lru_timestamps_;
    uint64_t              lru_clock_ = 0;
    uint32_t              recent_window_;
    uint32_t              max_seq_len_;

    // ── Background thread machinery ─────────────────────────────────────────
    std::thread              bg_thread_;
    std::mutex               bg_mu_;
    std::condition_variable  bg_cv_;
    std::atomic<bool>        bg_stop_{false};

    struct AsyncRequest {
        uint32_t         budget;
        EvictionCallback callback;
    };
    std::vector<AsyncRequest> pending_requests_;
};

}  // namespace nexus::kv
