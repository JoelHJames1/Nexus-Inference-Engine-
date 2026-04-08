/// NEXUS Prefetcher — Double-buffered async weight streaming.
///
/// Key insight: on Apple Silicon with fast NVMe (5-7 GB/s) and UMA,
/// we can overlap SSD reads with GPU compute. While the GPU processes
/// layer L's weights from buffer A, the CPU reads and dequantizes
/// layer L+1's weights into buffer B via GCD dispatch_io.

#include "memory/prefetcher.h"
#include "memory/memory_manager.h"
#include "quant/gptq.h"
#include <dispatch/dispatch.h>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

namespace nexus {

struct Prefetcher::Impl {
    MemoryManager& memory;
    size_t buffer_size;

    // Double buffers
    void* buffers[2] = {nullptr, nullptr};

    // GCD queue for async I/O and decompression
    dispatch_queue_t io_queue = nullptr;
    dispatch_group_t group = nullptr;

    Impl(MemoryManager& mem, size_t buf_size) : memory(mem), buffer_size(buf_size) {}
};

Prefetcher::Prefetcher(MemoryManager& memory, size_t buffer_size)
    : impl_(std::make_unique<Impl>(memory, buffer_size)) {

    // Allocate double buffers (page-aligned for UMA)
    impl_->buffers[0] = memory.alloc_pages(buffer_size);
    impl_->buffers[1] = memory.alloc_pages(buffer_size);

    if (!impl_->buffers[0] || !impl_->buffers[1]) {
        fprintf(stderr, "[nexus] WARNING: Failed to allocate prefetch buffers (%.1f MB each)\n",
                buffer_size / (1024.0 * 1024.0));
    }

    // Create GCD concurrent queue for I/O operations
    impl_->io_queue = dispatch_queue_create("nexus.prefetcher.io",
                                             DISPATCH_QUEUE_CONCURRENT);
    impl_->group = dispatch_group_create();

    fprintf(stderr, "[nexus] Prefetcher: 2 × %.1f MB double buffers\n",
            buffer_size / (1024.0 * 1024.0));
}

Prefetcher::~Prefetcher() {
    drain();

    if (impl_->io_queue) {
        dispatch_release(impl_->io_queue);
    }
    if (impl_->group) {
        dispatch_release(impl_->group);
    }

    if (impl_->buffers[0]) {
        impl_->memory.free_pages(impl_->buffers[0], impl_->buffer_size);
    }
    if (impl_->buffers[1]) {
        impl_->memory.free_pages(impl_->buffers[1], impl_->buffer_size);
    }
}

void Prefetcher::prefetch(int fd, uint64_t offset, size_t size,
                           Codec codec, PrefetchCallback callback) {
    if (!impl_->buffers[0] || !impl_->buffers[1]) return;

    int next = 1 - current_;
    void* dst = impl_->buffers[next];

    if (size > impl_->buffer_size) {
        fprintf(stderr, "[nexus] WARNING: Prefetch size (%zu) exceeds buffer (%zu)\n",
                size, impl_->buffer_size);
        size = impl_->buffer_size;
    }

    // Advisory prefetch to warm the OS file cache
    impl_->memory.prefetch(fd, offset, size);

    // Async read + decompress on GCD queue
    dispatch_group_async(impl_->group, impl_->io_queue, ^{
        // Read from file
        ssize_t bytes_read = pread(fd, dst, size, static_cast<off_t>(offset));
        if (bytes_read < 0) {
            fprintf(stderr, "[nexus] Prefetch read error at offset %llu\n", offset);
            return;
        }

        // Decompress/dequantize on CPU
        // For FP32/FP16 passthrough codecs, data is ready as-is.
        // For INT4/INT8, we dequantize in-place or to the same buffer.
        // For ANS entropy-coded data, we decompress first.
        //
        // Phase 2 codec dispatch:
        switch (codec) {
            case Codec::FP32:
            case Codec::FP16:
                // Passthrough — data is ready
                break;

            case Codec::INT4:
            case Codec::GPTQ:
            case Codec::AWQ: {
                // INT4 data: first portion is packed weights, followed by scales/zeros
                // The actual layout depends on how the NXF chunk was written.
                // For now, leave packed — dequant happens at compute time.
                break;
            }

            case Codec::ANS: {
                // ANS entropy-coded: decompress first, then handle inner codec
                // TODO: Integrate ANS decompressor here
                break;
            }

            default:
                break;
        }

        // Notify completion
        if (callback) {
            callback(dst, static_cast<size_t>(bytes_read));
        }
    });
}

void Prefetcher::drain() {
    if (impl_->group) {
        dispatch_group_wait(impl_->group, DISPATCH_TIME_FOREVER);
    }
}

void Prefetcher::swap() {
    // Wait for prefetch to complete before swapping
    drain();
    current_ = 1 - current_;
}

void* Prefetcher::current_buffer() {
    return impl_->buffers[current_];
}

void* Prefetcher::next_buffer() {
    return impl_->buffers[1 - current_];
}

}  // namespace nexus
