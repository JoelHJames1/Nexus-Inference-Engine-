/// NEXUS Inference Engine — UMA-aware Memory Manager (Apple Silicon)
///
/// Implementation: mmap-based page allocation, slab pools, LRU eviction,
/// fcntl(F_RDADVISE) prefetch, and GCD dispatch_io async reads.

#include "memory_manager.h"

#include <cerrno>
#include <cstring>
#include <algorithm>

// macOS / POSIX
#include <mach/mach.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <dispatch/dispatch.h>

namespace nexus {

// ═══════════════════════════════════════════════════════════════════════════
//  SlabPool
// ═══════════════════════════════════════════════════════════════════════════

SlabPool::SlabPool(size_t slab_size, size_t max_slabs)
    : slab_size_(slab_size), max_slabs_(max_slabs) {
    free_list_.reserve(max_slabs);
}

SlabPool::~SlabPool() {
    std::lock_guard<std::mutex> lock(mu_);
    for (void* p : free_list_) {
        ::munmap(p, slab_size_);
    }
    // NOTE: slabs that are still "in use" are the caller's responsibility.
}

void* SlabPool::allocate() {
    std::lock_guard<std::mutex> lock(mu_);

    // Reuse a previously freed slab if available.
    if (!free_list_.empty()) {
        void* p = free_list_.back();
        free_list_.pop_back();
        return p;
    }

    // Allocate a fresh slab if under the cap.
    if (total_allocated_ >= max_slabs_) {
        return nullptr;
    }

    void* p = ::mmap(nullptr, slab_size_,
                     PROT_READ | PROT_WRITE,
                     MAP_ANON | MAP_PRIVATE, -1, 0);
    if (p == MAP_FAILED) {
        return nullptr;
    }
    ++total_allocated_;
    return p;
}

void SlabPool::deallocate(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mu_);
    free_list_.push_back(ptr);
}

size_t SlabPool::in_use() const {
    std::lock_guard<std::mutex> lock(mu_);
    return total_allocated_ - free_list_.size();
}

size_t SlabPool::committed_bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return total_allocated_ * slab_size_;
}

// ═══════════════════════════════════════════════════════════════════════════
//  MemoryManager
// ═══════════════════════════════════════════════════════════════════════════

// Helpers: compute max slab counts from MemoryConfig.
static size_t weight_max_slabs(const MemoryConfig& cfg) {
    return (cfg.weight_buffer_mb * 1024ULL * 1024ULL) / kWeightSlabSize;
}

static size_t kv_max_slabs(const MemoryConfig& cfg) {
    // Sum of all KV tiers for total KV budget.
    size_t total_kv_mb = cfg.kv_hot_mb + cfg.kv_warm_mb + cfg.kv_cool_mb;
    return (total_kv_mb * 1024ULL * 1024ULL) / kKVPageSize;
}

MemoryManager::MemoryManager(const MemoryConfig& cfg)
    : ram_limit_(cfg.ram_limit),
      weight_pool_(kWeightSlabSize, weight_max_slabs(cfg)),
      kv_pool_(kKVPageSize, kv_max_slabs(cfg)) {}

MemoryManager::~MemoryManager() {
    // Unmap all tracked LRU regions.
    std::lock_guard<std::mutex> lock(lru_mu_);
    for (auto& region : lru_list_) {
        ::munmap(region.ptr, region.size);
    }
    lru_list_.clear();
    lru_map_.clear();
}

// ─── RSS query via Mach ────────────────────────────────────────────────────

size_t MemoryManager::current_rss() const {
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(),
                                MACH_TASK_BASIC_INFO,
                                reinterpret_cast<task_info_t>(&info),
                                &count);
    if (kr != KERN_SUCCESS) {
        return 0;
    }
    return static_cast<size_t>(info.resident_size);
}

size_t MemoryManager::headroom() const {
    size_t rss = current_rss();
    return (rss >= ram_limit_) ? 0 : (ram_limit_ - rss);
}

// ─── LRU tracking ──────────────────────────────────────────────────────────

void MemoryManager::track_region(void* ptr, size_t size) {
    // Caller must hold lru_mu_.
    Region r;
    r.ptr         = ptr;
    r.size        = size;
    r.last_access = Region::Clock::now();
    lru_list_.push_back(r);
    lru_map_[ptr] = std::prev(lru_list_.end());
}

void MemoryManager::untrack_region(void* ptr) {
    // Caller must hold lru_mu_.
    auto it = lru_map_.find(ptr);
    if (it != lru_map_.end()) {
        lru_list_.erase(it->second);
        lru_map_.erase(it);
    }
}

bool MemoryManager::evict_lru(size_t needed) {
    // Caller must hold lru_mu_.
    size_t freed = 0;
    while (freed < needed && !lru_list_.empty()) {
        Region& oldest = lru_list_.front();
        ::munmap(oldest.ptr, oldest.size);
        freed += oldest.size;
        lru_map_.erase(oldest.ptr);
        lru_list_.pop_front();
    }
    return freed >= needed;
}

// ─── Page allocation ───────────────────────────────────────────────────────

void* MemoryManager::alloc_pages(size_t bytes) {
    if (bytes == 0) return nullptr;

    // Round up to page boundary.
    bytes = (bytes + kPageSize - 1) & ~(kPageSize - 1);

    std::lock_guard<std::mutex> lock(lru_mu_);

    // Check headroom; attempt eviction if tight.
    if (current_rss() + bytes > ram_limit_) {
        size_t deficit = (current_rss() + bytes) - ram_limit_;
        if (!evict_lru(deficit)) {
            return nullptr;  // Cannot free enough memory.
        }
    }

    void* p = ::mmap(nullptr, bytes,
                     PROT_READ | PROT_WRITE,
                     MAP_ANON | MAP_PRIVATE, -1, 0);
    if (p == MAP_FAILED) {
        return nullptr;
    }

    track_region(p, bytes);
    return p;
}

void MemoryManager::free_pages(void* ptr, size_t bytes) {
    if (!ptr) return;
    bytes = (bytes + kPageSize - 1) & ~(kPageSize - 1);

    std::lock_guard<std::mutex> lock(lru_mu_);
    untrack_region(ptr);
    ::munmap(ptr, bytes);
}

// ─── Slab allocators ──────────────────────────────────────────────────────

void* MemoryManager::alloc_weight_slab() { return weight_pool_.allocate(); }
void  MemoryManager::free_weight_slab(void* ptr) { weight_pool_.deallocate(ptr); }

void* MemoryManager::alloc_kv_slab() { return kv_pool_.allocate(); }
void  MemoryManager::free_kv_slab(void* ptr) { kv_pool_.deallocate(ptr); }

// ─── Prefetch via fcntl(F_RDADVISE) ───────────────────────────────────────

int MemoryManager::prefetch(int fd, uint64_t offset, size_t length) {
    struct radvisory ra{};
    ra.ra_offset = static_cast<off_t>(offset);
    ra.ra_count  = static_cast<int>(length);
    return ::fcntl(fd, F_RDADVISE, &ra);
}

// ─── Async read via GCD dispatch_io ───────────────────────────────────────

namespace {
struct AsyncReadCtx {
    MemoryManager::IOCallback callback;
    void*   user_ctx;
    void*   dst;
    size_t  bytes_read;
};
}  // namespace

void MemoryManager::async_read(int fd, void* dst, uint64_t offset,
                               size_t length, IOCallback callback,
                               void* ctx) {
    dispatch_queue_t queue =
        dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);

    // Create a dispatch_io channel for the file descriptor.
    // We pass a cleanup handler that is a no-op (caller owns the fd).
    dispatch_io_t channel = dispatch_io_create(
        DISPATCH_IO_RANDOM, fd, queue,
        ^(int /*error*/) { /* fd lifetime managed by caller */ });

    if (!channel) {
        if (callback) callback(-1, ctx);
        return;
    }

    // Capture the callback context for the read handler.
    auto* arc = new AsyncReadCtx{callback, ctx, dst, 0};

    dispatch_io_read(channel,
                     static_cast<off_t>(offset),
                     length,
                     queue,
                     ^(bool done, dispatch_data_t data, int error) {
        if (data) {
            // Copy data into caller's buffer.
            dispatch_data_apply(data,
                ^bool(dispatch_data_t /*region*/, size_t /*region_off*/,
                      const void* buffer, size_t size) {
                    std::memcpy(static_cast<char*>(arc->dst) + arc->bytes_read,
                                buffer, size);
                    arc->bytes_read += size;
                    return true;  // continue iteration
                });
        }
        if (done) {
            if (arc->callback) {
                arc->callback(error, arc->user_ctx);
            }
            delete arc;
            dispatch_io_close(channel, 0);
            dispatch_release(channel);
        }
    });
}

}  // namespace nexus
