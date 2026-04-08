/// NEXUS Prefix Cache — Radix tree implementation.

#include "kv/prefix_cache.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <numeric>
#include <queue>

namespace nexus::kv {

// ─── Construction / destruction ────────────────────────────────────────────────

PrefixCache::PrefixCache(size_t max_entries)
    : root_(std::make_unique<PrefixNode>())
    , max_entries_(max_entries)
    , clock_(0) {}

PrefixCache::~PrefixCache() = default;

// ─── Tick (monotonic clock) ────────────────────────────────────────────────────

uint64_t PrefixCache::tick() { return ++clock_; }

// ─── Insert ────────────────────────────────────────────────────────────────────

void PrefixCache::insert(const std::vector<uint32_t>& token_ids,
                         const std::vector<uint32_t>& kv_page_ids) {
    if (token_ids.empty()) return;

    std::lock_guard<std::mutex> lock(mu_);
    uint64_t ts = tick();

    // Walk / build the tree.
    PrefixNode* cur = root_.get();
    cur->last_access_ts = ts;

    size_t ti = 0;  // index into token_ids
    size_t pi = 0;  // index into kv_page_ids

    while (ti < token_ids.size()) {
        uint32_t first = token_ids[ti];
        auto it = cur->children.find(first);

        if (it == cur->children.end()) {
            // No child starts with this token — create a new leaf with the
            // remaining tokens.
            auto child = std::make_unique<PrefixNode>();
            child->edge_tokens.assign(token_ids.begin() + ti, token_ids.end());
            child->last_access_ts = ts;

            // Assign remaining page IDs to this node.
            if (pi < kv_page_ids.size()) {
                child->kv_page_ids.assign(kv_page_ids.begin() + pi,
                                          kv_page_ids.end());
            }
            cur->children[first] = std::move(child);
            return;  // done
        }

        PrefixNode* child = it->second.get();
        const auto& edge = child->edge_tokens;

        // Match as far as possible along the edge.
        size_t match = 0;
        while (match < edge.size() && ti + match < token_ids.size() &&
               edge[match] == token_ids[ti + match]) {
            ++match;
        }

        if (match < edge.size()) {
            // Partial match — need to split this edge.
            // Create a new intermediate node that contains the matched prefix.
            auto mid = std::make_unique<PrefixNode>();
            mid->edge_tokens.assign(edge.begin(), edge.begin() + match);
            mid->last_access_ts = ts;

            // The existing child keeps the suffix of its edge.
            child->edge_tokens.erase(child->edge_tokens.begin(),
                                     child->edge_tokens.begin() + match);

            // The new intermediate's child map: the old child, keyed by its
            // new first edge token.
            uint32_t old_first = child->edge_tokens[0];

            // Move pages: if the intermediate should own some pages we split.
            // For simplicity the intermediate inherits pages proportional to
            // the match length; the original child keeps the rest.
            // Pages correspond roughly to segments so we split by ratio.
            size_t total_pages = child->kv_page_ids.size();
            size_t mid_pages = 0;
            if (!child->kv_page_ids.empty() && edge.size() > 0) {
                mid_pages = (total_pages * match) / edge.size();
            }
            if (mid_pages > 0) {
                mid->kv_page_ids.assign(child->kv_page_ids.begin(),
                                        child->kv_page_ids.begin() + mid_pages);
                child->kv_page_ids.erase(child->kv_page_ids.begin(),
                                         child->kv_page_ids.begin() + mid_pages);
            }

            // Reparent: mid owns old child.
            mid->children[old_first] = std::move(it->second);
            // Replace the entry in cur->children.
            cur->children[first] = std::move(mid);

            PrefixNode* mid_ptr = cur->children[first].get();
            ti += match;
            pi += mid_pages;

            if (ti < token_ids.size()) {
                // Still have tokens to insert — create a new leaf off mid.
                auto leaf = std::make_unique<PrefixNode>();
                leaf->edge_tokens.assign(token_ids.begin() + ti,
                                         token_ids.end());
                leaf->last_access_ts = ts;
                if (pi < kv_page_ids.size()) {
                    leaf->kv_page_ids.assign(kv_page_ids.begin() + pi,
                                             kv_page_ids.end());
                }
                uint32_t leaf_first = leaf->edge_tokens[0];
                mid_ptr->children[leaf_first] = std::move(leaf);
            } else {
                // The insertion ends exactly at the split point.
                // Associate any remaining page IDs.
                if (pi < kv_page_ids.size()) {
                    mid_ptr->kv_page_ids.insert(
                        mid_ptr->kv_page_ids.end(),
                        kv_page_ids.begin() + pi,
                        kv_page_ids.end());
                }
            }
            return;
        }

        // Full edge matched — advance and continue.
        ti += match;
        // Advance page index proportionally.
        size_t edge_pages = child->kv_page_ids.size();
        pi += edge_pages;  // consume all pages at this edge

        child->last_access_ts = ts;
        child->ref_count++;
        cur = child;
    }

    // We exhausted all tokens while walking existing edges.
    // Ensure pages are set on the current node.
    if (pi < kv_page_ids.size() && cur->kv_page_ids.empty()) {
        cur->kv_page_ids.assign(kv_page_ids.begin() + pi, kv_page_ids.end());
    }

    // Auto-evict if we exceed the soft cap.
    size_t n = count_nodes(root_.get());
    if (n > max_entries_) {
        // Unlock not needed — evict_lru_internal does not re-lock.
        evict_lru_internal(max_entries_);
    }
}

// ─── Lookup ────────────────────────────────────────────────────────────────────

PrefixMatch PrefixCache::lookup(const std::vector<uint32_t>& token_ids) {
    std::lock_guard<std::mutex> lock(mu_);
    ++lookup_count_;

    PrefixMatch result{0, {}};
    if (token_ids.empty()) return result;

    uint64_t ts = tick();

    PrefixNode* cur = root_.get();
    cur->last_access_ts = ts;
    size_t ti = 0;

    while (ti < token_ids.size()) {
        uint32_t first = token_ids[ti];
        auto it = cur->children.find(first);
        if (it == cur->children.end()) break;  // no further match

        PrefixNode* child = it->second.get();
        const auto& edge = child->edge_tokens;

        // Match along the edge.
        size_t match = 0;
        while (match < edge.size() && ti + match < token_ids.size() &&
               edge[match] == token_ids[ti + match]) {
            ++match;
        }

        ti += match;
        child->last_access_ts = ts;
        child->ref_count++;

        // Collect page IDs from this node.
        result.kv_page_ids.insert(result.kv_page_ids.end(),
                                  child->kv_page_ids.begin(),
                                  child->kv_page_ids.end());

        if (match < edge.size()) {
            // Partial edge match — stop here.
            break;
        }

        cur = child;
    }

    result.match_length = static_cast<uint32_t>(ti);
    if (result.match_length > 0) ++hit_count_;
    return result;
}

// ─── LRU eviction ──────────────────────────────────────────────────────────────

size_t PrefixCache::evict_lru(size_t target) {
    std::lock_guard<std::mutex> lock(mu_);
    return evict_lru_internal(target == 0 ? max_entries_ : target);
}

size_t PrefixCache::evict_lru_internal(size_t target) {
    size_t removed = 0;
    size_t cur_count = count_nodes(root_.get());

    while (cur_count > target) {
        // Collect all unreferenced leaf nodes.
        std::vector<std::pair<uint64_t, PrefixNode*>> candidates;
        collect_leaves_for_eviction(root_.get(), candidates);

        if (candidates.empty()) break;  // everything is referenced

        // Sort by timestamp ascending (oldest first).
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });

        // Remove the oldest leaf.
        bool did_remove = remove_oldest_leaf(root_.get(), candidates[0].second);
        if (!did_remove) break;
        ++removed;
        cur_count = count_nodes(root_.get());
    }

    return removed;
}

// ─── Helpers — node counting / memory ──────────────────────────────────────────

size_t PrefixCache::count_nodes(const PrefixNode* node) {
    if (!node) return 0;
    size_t n = 1;
    for (const auto& [_, child] : node->children) {
        n += count_nodes(child.get());
    }
    return n;
}

size_t PrefixCache::estimate_memory(const PrefixNode* node) {
    if (!node) return 0;
    size_t mem = sizeof(PrefixNode);
    mem += node->edge_tokens.capacity() * sizeof(uint32_t);
    mem += node->kv_page_ids.capacity() * sizeof(uint32_t);
    // Rough estimate for the hash map overhead.
    mem += node->children.size() * (sizeof(uint32_t) + sizeof(std::unique_ptr<PrefixNode>) + 32);
    for (const auto& [_, child] : node->children) {
        mem += estimate_memory(child.get());
    }
    return mem;
}

// ─── Helpers — eviction candidates ─────────────────────────────────────────────

void PrefixCache::collect_leaves_for_eviction(
        PrefixNode* node,
        std::vector<std::pair<uint64_t, PrefixNode*>>& out) const {
    if (!node) return;
    if (node->is_leaf() && node != root_.get() && node->ref_count == 0) {
        out.emplace_back(node->last_access_ts, node);
        return;
    }
    for (auto& [_, child] : node->children) {
        collect_leaves_for_eviction(child.get(), out);
    }
}

bool PrefixCache::remove_oldest_leaf(PrefixNode* parent, PrefixNode* target) {
    for (auto it = parent->children.begin(); it != parent->children.end(); ++it) {
        if (it->second.get() == target) {
            parent->children.erase(it);
            // If parent now has exactly one child and is not root, merge edge
            // (path compression).  We skip this for the root.
            if (parent != root_.get() && parent->children.size() == 1) {
                auto& only = parent->children.begin()->second;
                // Merge: extend parent's edge_tokens with the child's.
                parent->edge_tokens.insert(parent->edge_tokens.end(),
                                           only->edge_tokens.begin(),
                                           only->edge_tokens.end());
                parent->kv_page_ids.insert(parent->kv_page_ids.end(),
                                           only->kv_page_ids.begin(),
                                           only->kv_page_ids.end());
                parent->ref_count = std::max(parent->ref_count, only->ref_count);
                parent->last_access_ts = std::max(parent->last_access_ts,
                                                  only->last_access_ts);
                auto grandchildren = std::move(only->children);
                parent->children = std::move(grandchildren);
            }
            return true;
        }
        if (remove_oldest_leaf(it->second.get(), target)) return true;
    }
    return false;
}

// ─── Statistics ────────────────────────────────────────────────────────────────

PrefixCacheStats PrefixCache::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    PrefixCacheStats s;
    s.num_nodes    = count_nodes(root_.get());
    s.memory_bytes = estimate_memory(root_.get());
    s.lookup_count = lookup_count_;
    s.hit_count    = hit_count_;

    // Count total pages referenced.
    std::function<void(const PrefixNode*)> count_pages =
        [&](const PrefixNode* n) {
            if (!n) return;
            s.num_pages += n->kv_page_ids.size();
            for (const auto& [_, c] : n->children) count_pages(c.get());
        };
    count_pages(root_.get());

    return s;
}

// ─── Persistence — binary serialization ────────────────────────────────────────
//
// Format (little-endian):
//   [magic 4B] [version 4B]
//   Then recursive node encoding:
//     [edge_len:uint32] [edge tokens: edge_len * uint32]
//     [page_count:uint32] [page ids: page_count * uint32]
//     [ref_count:uint32] [last_access_ts:uint64]
//     [child_count:uint32]
//     For each child: [first_token:uint32] [child node...]

static constexpr uint32_t kPrefixCacheMagic   = 0x50435458;  // "PCTX"
static constexpr uint32_t kPrefixCacheVersion  = 1;

namespace detail {

template <typename T>
static void push_val(std::vector<uint8_t>& buf, T v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    buf.insert(buf.end(), p, p + sizeof(T));
}

template <typename T>
static bool read_val(const uint8_t*& ptr, const uint8_t* end, T& v) {
    if (ptr + sizeof(T) > end) return false;
    std::memcpy(&v, ptr, sizeof(T));
    ptr += sizeof(T);
    return true;
}

}  // namespace detail

bool PrefixCache::serialize_node(const PrefixNode* node,
                                 std::vector<uint8_t>& buf) const {
    using namespace detail;
    if (!node) return false;

    push_val<uint32_t>(buf, static_cast<uint32_t>(node->edge_tokens.size()));
    for (auto t : node->edge_tokens) push_val<uint32_t>(buf, t);

    push_val<uint32_t>(buf, static_cast<uint32_t>(node->kv_page_ids.size()));
    for (auto p : node->kv_page_ids) push_val<uint32_t>(buf, p);

    push_val<uint32_t>(buf, node->ref_count);
    push_val<uint64_t>(buf, node->last_access_ts);

    push_val<uint32_t>(buf, static_cast<uint32_t>(node->children.size()));
    for (const auto& [first_tok, child] : node->children) {
        push_val<uint32_t>(buf, first_tok);
        if (!serialize_node(child.get(), buf)) return false;
    }
    return true;
}

bool PrefixCache::deserialize_node(PrefixNode* node,
                                   const uint8_t*& ptr,
                                   const uint8_t* end) {
    using namespace detail;
    if (!node) return false;

    uint32_t edge_len = 0;
    if (!read_val(ptr, end, edge_len)) return false;
    node->edge_tokens.resize(edge_len);
    for (uint32_t i = 0; i < edge_len; ++i) {
        if (!read_val(ptr, end, node->edge_tokens[i])) return false;
    }

    uint32_t page_count = 0;
    if (!read_val(ptr, end, page_count)) return false;
    node->kv_page_ids.resize(page_count);
    for (uint32_t i = 0; i < page_count; ++i) {
        if (!read_val(ptr, end, node->kv_page_ids[i])) return false;
    }

    if (!read_val(ptr, end, node->ref_count)) return false;
    if (!read_val(ptr, end, node->last_access_ts)) return false;

    uint32_t child_count = 0;
    if (!read_val(ptr, end, child_count)) return false;
    for (uint32_t i = 0; i < child_count; ++i) {
        uint32_t first_tok = 0;
        if (!read_val(ptr, end, first_tok)) return false;
        auto child = std::make_unique<PrefixNode>();
        if (!deserialize_node(child.get(), ptr, end)) return false;
        node->children[first_tok] = std::move(child);
    }
    return true;
}

bool PrefixCache::save_to_disk(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<uint8_t> buf;
    buf.reserve(1024 * 1024);

    detail::push_val<uint32_t>(buf, kPrefixCacheMagic);
    detail::push_val<uint32_t>(buf, kPrefixCacheVersion);

    if (!serialize_node(root_.get(), buf)) return false;

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    ofs.write(reinterpret_cast<const char*>(buf.data()),
              static_cast<std::streamsize>(buf.size()));
    return ofs.good();
}

bool PrefixCache::load_from_disk(const std::string& path) {
    std::lock_guard<std::mutex> lock(mu_);

    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) return false;
    auto size = ifs.tellg();
    if (size <= 0) return false;
    ifs.seekg(0);

    std::vector<uint8_t> buf(static_cast<size_t>(size));
    ifs.read(reinterpret_cast<char*>(buf.data()), size);
    if (!ifs.good()) return false;

    const uint8_t* ptr = buf.data();
    const uint8_t* end = ptr + buf.size();

    uint32_t magic = 0, version = 0;
    if (!detail::read_val(ptr, end, magic) || magic != kPrefixCacheMagic)
        return false;
    if (!detail::read_val(ptr, end, version) || version != kPrefixCacheVersion)
        return false;

    auto new_root = std::make_unique<PrefixNode>();
    if (!deserialize_node(new_root.get(), ptr, end)) return false;

    root_ = std::move(new_root);
    // Reset stats on load.
    lookup_count_ = 0;
    hit_count_ = 0;
    return true;
}

}  // namespace nexus::kv
