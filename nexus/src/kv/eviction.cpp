/// NEXUS KV Eviction Strategies — H2O, SnapKV, and EvictionManager.

#include "kv/eviction.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

namespace nexus::kv {

// ═══════════════════════════════════════════════════════════════════════════════
//  H2OEviction
// ═══════════════════════════════════════════════════════════════════════════════

H2OEviction::H2OEviction(uint32_t max_seq_len,
                         uint32_t initial_tokens,
                         uint32_t recent_window)
    : max_seq_len_(max_seq_len)
    , initial_tokens_(initial_tokens)
    , recent_window_(recent_window)
    , cumulative_scores_(max_seq_len, 0.0) {}

void H2OEviction::update_scores(uint32_t /*layer*/,
                                const float* attention_weights,
                                uint32_t seq_len) {
    if (!attention_weights || seq_len == 0) return;
    current_seq_len_ = std::max(current_seq_len_, seq_len);

    uint32_t n = std::min(seq_len, max_seq_len_);
    for (uint32_t i = 0; i < n; ++i) {
        cumulative_scores_[i] += static_cast<double>(attention_weights[i]);
    }
}

std::vector<int> H2OEviction::select_keep(uint32_t budget) const {
    if (current_seq_len_ == 0) return {};

    uint32_t seq = std::min(current_seq_len_, max_seq_len_);

    // If budget >= seq we keep everything.
    if (budget >= seq) {
        std::vector<int> all(seq);
        std::iota(all.begin(), all.end(), 0);
        return all;
    }

    // Partition the budget: initial_tokens + recent_window + heavy hitters.
    uint32_t n_initial = std::min(initial_tokens_, seq);
    uint32_t n_recent  = std::min(recent_window_, seq);

    // Clamp so that initial + recent does not exceed budget.
    if (n_initial + n_recent > budget) {
        // Prioritize initial tokens, then recent.
        n_recent = (budget > n_initial) ? budget - n_initial : 0;
        if (n_initial > budget) n_initial = budget;
    }

    uint32_t n_heavy = (budget > n_initial + n_recent)
                           ? budget - n_initial - n_recent
                           : 0;

    // Mark positions that are already covered by initial / recent.
    std::vector<bool> covered(seq, false);
    for (uint32_t i = 0; i < n_initial && i < seq; ++i)
        covered[i] = true;
    for (uint32_t i = (seq > n_recent ? seq - n_recent : 0); i < seq; ++i)
        covered[i] = true;

    // Find heavy hitters among uncovered middle positions.
    std::vector<std::pair<double, int>> candidates;
    for (uint32_t i = 0; i < seq; ++i) {
        if (!covered[i]) {
            candidates.emplace_back(cumulative_scores_[i],
                                    static_cast<int>(i));
        }
    }

    // Partial sort to get the top n_heavy.
    if (n_heavy > 0 && !candidates.empty()) {
        uint32_t k = std::min(n_heavy, static_cast<uint32_t>(candidates.size()));
        std::partial_sort(candidates.begin(),
                          candidates.begin() + k,
                          candidates.end(),
                          [](const auto& a, const auto& b) {
                              return a.first > b.first;  // descending
                          });
        for (uint32_t i = 0; i < k; ++i) {
            covered[candidates[i].second] = true;
        }
    }

    // Collect all kept positions.
    std::vector<int> keep;
    keep.reserve(budget);
    for (uint32_t i = 0; i < seq; ++i) {
        if (covered[i]) keep.push_back(static_cast<int>(i));
    }

    return keep;
}

std::vector<int> H2OEviction::select_evict(uint32_t budget) const {
    auto keep_set = select_keep(budget);
    uint32_t seq = std::min(current_seq_len_, max_seq_len_);

    std::vector<bool> keep_mask(seq, false);
    for (int pos : keep_set) {
        if (pos >= 0 && static_cast<uint32_t>(pos) < seq)
            keep_mask[pos] = true;
    }

    std::vector<int> evict;
    evict.reserve(seq - keep_set.size());
    for (uint32_t i = 0; i < seq; ++i) {
        if (!keep_mask[i]) evict.push_back(static_cast<int>(i));
    }
    return evict;
}

void H2OEviction::reset() {
    std::fill(cumulative_scores_.begin(), cumulative_scores_.end(), 0.0);
    current_seq_len_ = 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SnapKVEviction
// ═══════════════════════════════════════════════════════════════════════════════

SnapKVEviction::SnapKVEviction(uint32_t max_seq_len,
                               uint32_t num_layers,
                               uint32_t num_heads,
                               uint32_t obs_window,
                               uint32_t recent_window)
    : max_seq_len_(max_seq_len)
    , num_layers_(num_layers)
    , num_heads_(num_heads)
    , obs_window_(obs_window)
    , recent_window_(recent_window)
    , importance_votes_(max_seq_len, 0.0) {}

void SnapKVEviction::observe(uint32_t /*layer*/,
                             uint32_t /*head*/,
                             const float* attention_weights,
                             uint32_t seq_len) {
    if (!attention_weights || seq_len == 0) return;
    current_seq_len_ = std::max(current_seq_len_, seq_len);
    ++observe_count_;

    uint32_t n = std::min(seq_len, max_seq_len_);

    // The observation window: we look at the attention the last `obs_window_`
    // query positions pay to all prior keys.  Here we receive a single
    // aggregated attention vector (summed/averaged over the observation window
    // queries).  We treat high attention weight as a "vote" for that position.

    // Find a threshold: positions that receive above-average attention get a
    // vote.  This loosely mimics the SnapKV "pooling + top-k per head"
    // strategy.
    double sum = 0.0;
    for (uint32_t i = 0; i < n; ++i)
        sum += static_cast<double>(attention_weights[i]);

    double mean = (n > 0) ? sum / n : 0.0;

    for (uint32_t i = 0; i < n; ++i) {
        if (static_cast<double>(attention_weights[i]) > mean) {
            importance_votes_[i] += 1.0;
        }
    }
}

std::vector<int> SnapKVEviction::select_keep(uint32_t budget) const {
    if (current_seq_len_ == 0) return {};

    uint32_t seq = std::min(current_seq_len_, max_seq_len_);
    if (budget >= seq) {
        std::vector<int> all(seq);
        std::iota(all.begin(), all.end(), 0);
        return all;
    }

    uint32_t n_recent = std::min(recent_window_, seq);
    uint32_t n_important = (budget > n_recent) ? budget - n_recent : 0;

    // Mark recent positions.
    std::vector<bool> covered(seq, false);
    for (uint32_t i = (seq > n_recent ? seq - n_recent : 0); i < seq; ++i)
        covered[i] = true;

    // Rank non-recent positions by importance votes.
    if (n_important > 0) {
        std::vector<std::pair<double, int>> candidates;
        uint32_t recent_start = (seq > n_recent) ? seq - n_recent : 0;
        for (uint32_t i = 0; i < recent_start; ++i) {
            candidates.emplace_back(importance_votes_[i], static_cast<int>(i));
        }

        uint32_t k = std::min(n_important,
                              static_cast<uint32_t>(candidates.size()));
        if (k > 0) {
            std::partial_sort(candidates.begin(),
                              candidates.begin() + k,
                              candidates.end(),
                              [](const auto& a, const auto& b) {
                                  return a.first > b.first;
                              });
            for (uint32_t i = 0; i < k; ++i)
                covered[candidates[i].second] = true;
        }
    }

    std::vector<int> keep;
    keep.reserve(budget);
    for (uint32_t i = 0; i < seq; ++i) {
        if (covered[i]) keep.push_back(static_cast<int>(i));
    }
    return keep;
}

void SnapKVEviction::reset() {
    std::fill(importance_votes_.begin(), importance_votes_.end(), 0.0);
    current_seq_len_ = 0;
    observe_count_ = 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  EvictionManager
// ═══════════════════════════════════════════════════════════════════════════════

EvictionManager::EvictionManager(EvictionStrategy strategy,
                                 uint32_t max_seq_len,
                                 uint32_t num_layers,
                                 uint32_t num_heads,
                                 uint32_t initial_tokens,
                                 uint32_t recent_window)
    : strategy_(strategy)
    , recent_window_(recent_window)
    , max_seq_len_(max_seq_len) {
    // Instantiate the strategies that are needed.
    if (strategy == EvictionStrategy::H2O ||
        strategy == EvictionStrategy::Combined) {
        h2o_ = std::make_unique<H2OEviction>(max_seq_len, initial_tokens,
                                             recent_window);
    }
    if (strategy == EvictionStrategy::SnapKV ||
        strategy == EvictionStrategy::Combined) {
        snapkv_ = std::make_unique<SnapKVEviction>(max_seq_len, num_layers,
                                                   num_heads, /*obs_window=*/64,
                                                   recent_window);
    }
    if (strategy == EvictionStrategy::LRU) {
        lru_timestamps_.resize(max_seq_len, 0);
    }

    // Start the background eviction thread.
    bg_thread_ = std::thread(&EvictionManager::background_loop, this);
}

EvictionManager::~EvictionManager() {
    {
        std::lock_guard<std::mutex> lock(bg_mu_);
        bg_stop_ = true;
    }
    bg_cv_.notify_all();
    if (bg_thread_.joinable()) bg_thread_.join();
}

// ── Observation forwarding ──────────────────────────────────────────────────

void EvictionManager::update_h2o_scores(uint32_t layer,
                                        const float* attention_weights,
                                        uint32_t seq_len) {
    if (h2o_) h2o_->update_scores(layer, attention_weights, seq_len);

    // Also update LRU timestamps: the most recent position touched is seq_len-1.
    if (strategy_ == EvictionStrategy::LRU && seq_len > 0) {
        uint32_t pos = seq_len - 1;
        if (pos < max_seq_len_) {
            lru_timestamps_[pos] = ++lru_clock_;
        }
    }
}

void EvictionManager::observe_snapkv(uint32_t layer, uint32_t head,
                                     const float* attention_weights,
                                     uint32_t seq_len) {
    if (snapkv_) snapkv_->observe(layer, head, attention_weights, seq_len);
}

// ── Eviction decisions ──────────────────────────────────────────────────────

bool EvictionManager::should_evict(size_t current_memory,
                                   size_t memory_budget) const {
    return current_memory > memory_budget;
}

std::vector<int> EvictionManager::get_eviction_candidates(uint32_t budget) {
    switch (strategy_) {
        case EvictionStrategy::H2O: {
            if (h2o_) return h2o_->select_evict(budget);
            return {};
        }
        case EvictionStrategy::SnapKV: {
            if (!snapkv_) return {};
            // SnapKV returns keep-set; invert to eviction set.
            auto keep = snapkv_->select_keep(budget);
            uint32_t seq = snapkv_->current_seq_len();
            std::vector<bool> keep_mask(seq, false);
            for (int p : keep) {
                if (p >= 0 && static_cast<uint32_t>(p) < seq) keep_mask[p] = true;
            }
            std::vector<int> evict;
            for (uint32_t i = 0; i < seq; ++i) {
                if (!keep_mask[i]) evict.push_back(static_cast<int>(i));
            }
            return evict;
        }
        case EvictionStrategy::Combined: {
            // Union of H2O and SnapKV keep-sets.
            uint32_t seq = 0;
            std::vector<bool> keep_mask;

            if (h2o_) {
                seq = std::max(seq, h2o_->current_seq_len());
            }
            if (snapkv_) {
                seq = std::max(seq, snapkv_->current_seq_len());
            }
            if (seq == 0) return {};

            keep_mask.assign(seq, false);

            if (h2o_) {
                for (int p : h2o_->select_keep(budget)) {
                    if (p >= 0 && static_cast<uint32_t>(p) < seq)
                        keep_mask[p] = true;
                }
            }
            if (snapkv_) {
                for (int p : snapkv_->select_keep(budget)) {
                    if (p >= 0 && static_cast<uint32_t>(p) < seq)
                        keep_mask[p] = true;
                }
            }

            // If the union exceeds budget, trim by removing the lowest-scored
            // positions from the combined set (using H2O scores as tie-breaker).
            uint32_t kept = 0;
            for (bool b : keep_mask) if (b) ++kept;

            if (kept > budget && h2o_) {
                // Build sorted list of kept positions by H2O score ascending.
                std::vector<std::pair<double, int>> scored;
                // We cannot directly access cumulative_scores_ since it is
                // private.  We use select_keep ordering as a proxy: positions
                // NOT in the H2O keep-set are the weakest.
                auto h2o_keep = h2o_->select_keep(budget);
                std::vector<bool> h2o_mask(seq, false);
                for (int p : h2o_keep) {
                    if (p >= 0 && static_cast<uint32_t>(p) < seq)
                        h2o_mask[p] = true;
                }
                // Remove positions that neither H2O nor SnapKV uniquely needs.
                // Strategy: drop SnapKV-only positions first.
                for (uint32_t i = 0; i < seq && kept > budget; ++i) {
                    if (keep_mask[i] && !h2o_mask[i]) {
                        keep_mask[i] = false;
                        --kept;
                    }
                }
                // If still over budget, drop from H2O set (lowest positions
                // that are not initial/recent).
                for (uint32_t i = 0; i < seq && kept > budget; ++i) {
                    if (keep_mask[i]) {
                        keep_mask[i] = false;
                        --kept;
                    }
                }
            }

            std::vector<int> evict;
            for (uint32_t i = 0; i < seq; ++i) {
                if (!keep_mask[i]) evict.push_back(static_cast<int>(i));
            }
            return evict;
        }
        case EvictionStrategy::LRU: {
            // Evict positions with the oldest timestamps, keeping `budget`.
            uint32_t seq = 0;
            for (uint32_t i = 0; i < max_seq_len_; ++i) {
                if (lru_timestamps_[i] > 0) seq = i + 1;
            }
            if (budget >= seq) return {};

            // Build (timestamp, position) pairs and keep the newest `budget`.
            std::vector<std::pair<uint64_t, int>> tps;
            tps.reserve(seq);
            for (uint32_t i = 0; i < seq; ++i) {
                tps.emplace_back(lru_timestamps_[i], static_cast<int>(i));
            }

            // Always keep recent window.
            uint32_t n_recent = std::min(recent_window_, seq);
            uint32_t recent_start = seq - n_recent;

            // Sort non-recent by timestamp descending, keep the top ones.
            std::vector<std::pair<uint64_t, int>> middle;
            for (auto& [ts, pos] : tps) {
                if (static_cast<uint32_t>(pos) >= recent_start) continue;
                middle.emplace_back(ts, pos);
            }

            uint32_t middle_budget =
                (budget > n_recent) ? budget - n_recent : 0;
            uint32_t k = std::min(middle_budget,
                                  static_cast<uint32_t>(middle.size()));

            std::vector<bool> keep_mask(seq, false);
            // Keep recent window.
            for (uint32_t i = recent_start; i < seq; ++i) keep_mask[i] = true;

            if (k > 0 && !middle.empty()) {
                std::partial_sort(middle.begin(), middle.begin() + k,
                                  middle.end(),
                                  [](const auto& a, const auto& b) {
                                      return a.first > b.first;  // newest first
                                  });
                for (uint32_t i = 0; i < k; ++i)
                    keep_mask[middle[i].second] = true;
            }

            std::vector<int> evict;
            for (uint32_t i = 0; i < seq; ++i) {
                if (!keep_mask[i]) evict.push_back(static_cast<int>(i));
            }
            return evict;
        }
    }
    return {};
}

// ── Async eviction ──────────────────────────────────────────────────────────

void EvictionManager::request_async_eviction(uint32_t budget,
                                             EvictionCallback callback) {
    {
        std::lock_guard<std::mutex> lock(bg_mu_);
        pending_requests_.push_back({budget, std::move(callback)});
    }
    bg_cv_.notify_one();
}

void EvictionManager::wait_for_pending() {
    // Spin until pending_requests_ is empty.  The background thread drains it.
    std::unique_lock<std::mutex> lock(bg_mu_);
    bg_cv_.wait(lock, [this] { return pending_requests_.empty() || bg_stop_; });
}

void EvictionManager::background_loop() {
    while (true) {
        std::vector<AsyncRequest> batch;
        {
            std::unique_lock<std::mutex> lock(bg_mu_);
            bg_cv_.wait(lock, [this] {
                return !pending_requests_.empty() || bg_stop_;
            });
            if (bg_stop_ && pending_requests_.empty()) return;
            batch.swap(pending_requests_);
        }
        // Notify waiters that the queue is drained (requests are being
        // processed).
        bg_cv_.notify_all();

        for (auto& req : batch) {
            auto candidates = get_eviction_candidates(req.budget);
            if (req.callback) {
                req.callback(std::move(candidates));
            }
        }
    }
}

void EvictionManager::reset() {
    if (h2o_) h2o_->reset();
    if (snapkv_) snapkv_->reset();
    if (!lru_timestamps_.empty()) {
        std::fill(lru_timestamps_.begin(), lru_timestamps_.end(), 0ULL);
        lru_clock_ = 0;
    }
}

}  // namespace nexus::kv
