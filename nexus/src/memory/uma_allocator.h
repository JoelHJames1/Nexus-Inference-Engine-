#pragma once
/// NEXUS Inference Engine — UMA-specific Metal Buffer Allocator (stub)
///
/// Phase 1: thin wrapper over mmap for CPU+GPU shared memory.
/// Phase 2: will use MTLBuffer with storageModeShared for true UMA zero-copy.

#include <cstddef>
#include <mutex>
#include <unordered_map>

namespace nexus {

class UMAAllocator {
public:
    UMAAllocator();
    ~UMAAllocator();

    UMAAllocator(const UMAAllocator&) = delete;
    UMAAllocator& operator=(const UMAAllocator&) = delete;

    /// Allocate shared CPU/GPU memory.
    /// Phase 1: returns mmap'd anonymous pages (page-aligned).
    /// Phase 2: will return MTLBuffer.contents() backed by storageModeShared.
    void* allocate_shared(size_t bytes);

    /// Free memory previously returned by allocate_shared().
    void  free_shared(void* ptr, size_t bytes);

    /// Number of live allocations.
    size_t allocation_count() const;

    /// Total bytes currently allocated.
    size_t total_bytes() const;

private:
    mutable std::mutex mu_;
    std::unordered_map<void*, size_t> allocations_;  // ptr -> size
    size_t total_bytes_ = 0;
};

}  // namespace nexus
