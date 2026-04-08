#pragma once
/// NEXUS Prefetcher — Async weight streaming pipeline.
///
/// Coordinates double-buffered weight loading from SSD:
///   Buffer A: GPU computing layer L
///   Buffer B: CPU loading + dequanting layer L+1 from SSD
///   Swap on layer completion.
///
/// Uses GCD dispatch queues for CPU↔GPU overlap.

#include "core/config.h"
#include <cstdint>
#include <functional>
#include <memory>

namespace nexus {

class MemoryManager;

/// Callback when a prefetch completes. Receives pointer to decoded weight data.
using PrefetchCallback = std::function<void(void* data, size_t size)>;

/// Double-buffered weight prefetcher for layer streaming.
class Prefetcher {
public:
    /// Create prefetcher with given buffer size (per buffer).
    /// Two buffers are allocated for double-buffering.
    Prefetcher(MemoryManager& memory, size_t buffer_size);
    ~Prefetcher();

    /// Start async prefetch of a chunk from file.
    /// @param fd          File descriptor to read from
    /// @param offset      Byte offset in file
    /// @param size        Bytes to read
    /// @param codec       Codec for decompression (done on CPU during prefetch)
    /// @param callback    Called on completion with decoded data pointer
    void prefetch(int fd, uint64_t offset, size_t size,
                  Codec codec, PrefetchCallback callback);

    /// Wait for all pending prefetches to complete.
    void drain();

    /// Swap buffers (call after GPU finishes with current buffer).
    void swap();

    /// Get pointer to current (GPU-side) buffer.
    void* current_buffer();

    /// Get pointer to next (prefetch-side) buffer.
    void* next_buffer();

    /// Current buffer index (0 or 1).
    int current_index() const { return current_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int current_ = 0;
};

}  // namespace nexus
