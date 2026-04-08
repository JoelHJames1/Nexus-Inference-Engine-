#pragma once
/// NEXUS Inference Engine — UMA-aware Memory Manager (Apple Silicon)
///
/// Provides page-aligned allocation, slab allocation for weight chunks and
/// KV pages, LRU eviction tracking, prefetch via F_RDADVISE, and async I/O
/// via GCD dispatch_io.  Thread-safe.

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "../core/config.h"

namespace nexus {

// ─── LRU region descriptor ─────────────────────────────────────────────────
struct Region {
    void*    ptr;
    size_t   size;
    using Clock = std::chrono::steady_clock;
    Clock::time_point last_access;
};

// ─── Slab pool for fixed-size allocations ──────────────────────────────────
class SlabPool {
public:
    /// \param slab_size  Size of each slab (e.g. 1 MB for weights, 64 KB for KV).
    /// \param max_slabs  Maximum number of slabs this pool may hold.
    explicit SlabPool(size_t slab_size, size_t max_slabs);
    ~SlabPool();

    SlabPool(const SlabPool&) = delete;
    SlabPool& operator=(const SlabPool&) = delete;

    /// Allocate one slab.  Returns nullptr if the pool is exhausted.
    void* allocate();

    /// Return a slab to the free list.
    void  deallocate(void* ptr);

    /// Number of slabs currently in use.
    size_t in_use() const;

    /// Total bytes committed by this pool.
    size_t committed_bytes() const;

private:
    size_t              slab_size_;
    size_t              max_slabs_;
    std::vector<void*>  free_list_;
    size_t              total_allocated_ = 0;
    mutable std::mutex  mu_;
};

// ─── Main memory manager ───────────────────────────────────────────────────
class MemoryManager {
public:
    /// Construct with an explicit memory config (ram_limit is the hard cap).
    explicit MemoryManager(const MemoryConfig& cfg = MemoryConfig{});
    ~MemoryManager();

    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;

    // ── Raw page allocation ────────────────────────────────────────────────
    /// Allocate anonymous private pages via mmap.  Returns nullptr on failure
    /// or if the allocation would exceed the RAM cap (after attempting eviction).
    void* alloc_pages(size_t bytes);

    /// Free pages previously returned by alloc_pages().
    void  free_pages(void* ptr, size_t bytes);

    // ── Slab allocators ────────────────────────────────────────────────────
    /// Allocate a 1 MB weight slab.
    void* alloc_weight_slab();
    void  free_weight_slab(void* ptr);

    /// Allocate a 64 KB KV page slab.
    void* alloc_kv_slab();
    void  free_kv_slab(void* ptr);

    // ── I/O helpers ────────────────────────────────────────────────────────
    /// Advisory read-ahead via fcntl(F_RDADVISE).
    /// Returns 0 on success, -1 on error (errno set).
    int prefetch(int fd, uint64_t offset, size_t length);

    /// Asynchronous read via GCD dispatch_io.
    /// Calls \p callback(int error) when complete (error == 0 on success).
    using IOCallback = void(*)(int error, void* ctx);
    void async_read(int fd, void* dst, uint64_t offset, size_t length,
                    IOCallback callback, void* ctx);

    // ── Diagnostics ────────────────────────────────────────────────────────
    /// Current RSS of this process (via mach_task_info).
    size_t current_rss() const;

    /// Configured RAM cap.
    size_t ram_limit() const { return ram_limit_; }

    /// Headroom = ram_limit - current_rss (clamped to 0).
    size_t headroom() const;

private:
    // Try to evict enough LRU regions to free at least \p needed bytes.
    // Returns true if enough was freed.
    bool evict_lru(size_t needed);

    // Register / unregister a region for LRU tracking.
    void track_region(void* ptr, size_t size);
    void untrack_region(void* ptr);

    size_t ram_limit_;

    // LRU list (front = oldest).
    using LRUList = std::list<Region>;
    LRUList                                     lru_list_;
    std::unordered_map<void*, LRUList::iterator> lru_map_;
    mutable std::mutex                           lru_mu_;

    // Slab pools.
    SlabPool weight_pool_;
    SlabPool kv_pool_;
};

}  // namespace nexus
