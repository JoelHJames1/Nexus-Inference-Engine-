/// NEXUS Inference Engine — UMA-specific Metal Buffer Allocator (stub)
///
/// Phase 1 implementation: mmap-based shared memory.
/// Phase 2 will replace the mmap calls with MTLDevice newBufferWithLength:
///   options:MTLResourceStorageModeShared for true UMA zero-copy between
///   CPU and GPU on Apple Silicon.

#include "uma_allocator.h"

#include <sys/mman.h>

#include "../core/config.h"   // kPageSize

namespace nexus {

UMAAllocator::UMAAllocator() = default;

UMAAllocator::~UMAAllocator() {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& [ptr, size] : allocations_) {
        ::munmap(ptr, size);
    }
    allocations_.clear();
    total_bytes_ = 0;
}

void* UMAAllocator::allocate_shared(size_t bytes) {
    if (bytes == 0) return nullptr;

    // Round up to VM page boundary.
    bytes = (bytes + kPageSize - 1) & ~(kPageSize - 1);

    void* p = ::mmap(nullptr, bytes,
                     PROT_READ | PROT_WRITE,
                     MAP_ANON | MAP_PRIVATE, -1, 0);
    if (p == MAP_FAILED) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mu_);
    allocations_[p] = bytes;
    total_bytes_ += bytes;
    return p;
}

void UMAAllocator::free_shared(void* ptr, size_t bytes) {
    if (!ptr) return;

    bytes = (bytes + kPageSize - 1) & ~(kPageSize - 1);

    std::lock_guard<std::mutex> lock(mu_);
    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        total_bytes_ -= it->second;
        allocations_.erase(it);
    }
    ::munmap(ptr, bytes);
}

size_t UMAAllocator::allocation_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return allocations_.size();
}

size_t UMAAllocator::total_bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return total_bytes_;
}

}  // namespace nexus
