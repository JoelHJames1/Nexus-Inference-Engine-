#pragma once
/// NEXUS Prefix Cache — Radix tree for KV page reuse across requests.
///
/// Inspired by SGLang's RadixAttention: a compressed radix tree (trie) indexed
/// by token sequences that maps prefixes to KV page lists.  When a new request
/// shares a prompt prefix with a previous one the engine can skip recomputation
/// and directly reuse the cached KV pages.
///
/// Phase 3 component.

#include "core/config.h"
#include <cstdint>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace nexus::kv {

// ─── Radix tree node ───────────────────────────────────────────────────────────

/// Each node stores a *segment* of token IDs (compact representation: edges
/// carry multiple tokens so internal chains without branching are collapsed).
struct PrefixNode {
    /// The token-ID segment stored on the edge leading TO this node.
    /// For the root node this is empty.
    std::vector<uint32_t> edge_tokens;

    /// KV page IDs that correspond to the tokens accumulated on the path from
    /// the root through this node.  Non-empty only at page boundaries (i.e.
    /// when the accumulated token count is a multiple of the page token width).
    std::vector<uint32_t> kv_page_ids;

    /// Children keyed by the *first* token of their edge segment.
    std::unordered_map<uint32_t, std::unique_ptr<PrefixNode>> children;

    /// How many active sequences currently reference this node's pages.
    uint32_t ref_count = 0;

    /// Monotonic timestamp of the last access (for LRU eviction).
    uint64_t last_access_ts = 0;

    /// True if this node has no children.
    bool is_leaf() const { return children.empty(); }
};

// ─── Cache statistics ──────────────────────────────────────────────────────────

struct PrefixCacheStats {
    uint64_t num_nodes       = 0;  ///< Total nodes in the tree (including root).
    uint64_t num_pages       = 0;  ///< Total KV pages referenced by the tree.
    size_t   memory_bytes    = 0;  ///< Estimated memory footprint of the tree.
    uint64_t lookup_count    = 0;  ///< Total number of lookups.
    uint64_t hit_count       = 0;  ///< Lookups that matched at least one token.
    double   hit_rate() const { return lookup_count ? static_cast<double>(hit_count) / lookup_count : 0.0; }
};

// ─── Lookup result ─────────────────────────────────────────────────────────────

struct PrefixMatch {
    uint32_t              match_length;  ///< Number of tokens matched.
    std::vector<uint32_t> kv_page_ids;   ///< KV page IDs covering the matched prefix.
};

// ─── Prefix cache (thread-safe) ────────────────────────────────────────────────

class PrefixCache {
public:
    /// Construct with a soft cap on the number of nodes before eviction kicks in.
    explicit PrefixCache(size_t max_entries = 65536);
    ~PrefixCache();

    // ── Mutators ────────────────────────────────────────────────────────────

    /// Insert a token sequence and its associated KV page list.
    /// Pages are associated with page-boundary nodes along the path.
    void insert(const std::vector<uint32_t>& token_ids,
                const std::vector<uint32_t>& kv_page_ids);

    /// Evict the least-recently-used prefix entries until the node count is
    /// at most `target` (defaults to max_entries_).  Returns the number of
    /// nodes removed.
    size_t evict_lru(size_t target = 0);

    // ── Queries ─────────────────────────────────────────────────────────────

    /// Find the longest prefix match for `token_ids`.
    PrefixMatch lookup(const std::vector<uint32_t>& token_ids);

    /// Snapshot of current statistics.
    PrefixCacheStats stats() const;

    // ── Persistence ─────────────────────────────────────────────────────────

    /// Serialize the entire prefix tree to a binary file.
    bool save_to_disk(const std::string& path) const;

    /// Load a previously saved prefix tree, replacing the current one.
    bool load_from_disk(const std::string& path);

private:
    // ── Helpers ─────────────────────────────────────────────────────────────

    /// Advance the global clock and return the new timestamp.
    uint64_t tick();

    /// Internal eviction (caller must hold mu_).
    size_t evict_lru_internal(size_t target);

    /// Count nodes in a subtree rooted at `node`.
    static size_t count_nodes(const PrefixNode* node);

    /// Estimate memory used by the subtree rooted at `node`.
    static size_t estimate_memory(const PrefixNode* node);

    /// Collect unreferenced leaf nodes with their timestamps for eviction.
    void collect_leaves_for_eviction(
        PrefixNode* node,
        std::vector<std::pair<uint64_t, PrefixNode*>>& out) const;

    /// Remove a specific leaf from the tree, merging edges if needed.
    bool remove_oldest_leaf(PrefixNode* parent, PrefixNode* target);

    /// Recursive serialization helpers.
    bool serialize_node(const PrefixNode* node, std::vector<uint8_t>& buf) const;
    bool deserialize_node(PrefixNode* node, const uint8_t*& ptr, const uint8_t* end);

    // ── Data ────────────────────────────────────────────────────────────────
    std::unique_ptr<PrefixNode> root_;
    size_t                      max_entries_;
    uint64_t                    clock_ = 0;

    // Stats accumulators (mutable so `lookup` can update hit counters).
    mutable uint64_t            lookup_count_ = 0;
    mutable uint64_t            hit_count_    = 0;

    mutable std::mutex          mu_;
};

}  // namespace nexus::kv
