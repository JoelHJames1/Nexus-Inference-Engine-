#pragma once
/// NEXUS Inference Engine — Metal GPU context.
///
/// Manages the MTLDevice, command queue, compiled shader libraries, and
/// pipeline state cache.  Uses an opaque pointer (pimpl) so callers only
/// need C++ headers.

#include <cstddef>
#include <memory>
#include <string>

namespace nexus::compute {

/// Opaque wrapper around Metal runtime objects.
/// The actual Objective-C++ implementation lives in metal_context.mm.
struct MetalContextImpl;

class MetalContext {
public:
    MetalContext();
    ~MetalContext();

    // Non-copyable, movable.
    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;
    MetalContext(MetalContext&&) noexcept;
    MetalContext& operator=(MetalContext&&) noexcept;

    /// Returns true if a usable Metal GPU was found.
    bool is_available() const;

    /// Human-readable name of the GPU (e.g. "Apple M4 Max").
    std::string device_name() const;

    /// Maximum recommended working-set size in bytes.
    size_t recommended_max_working_set_size() const;

    /// Load a .metallib shader library from the given path.
    /// Returns true on success.
    bool load_library(const std::string& path);

    /// Create (or retrieve cached) compute pipeline state for a named kernel.
    /// Returns true if the pipeline is ready.
    bool make_pipeline(const std::string& function_name);

    /// Allocate a storageModeShared MTLBuffer (UMA zero-copy).
    /// Returns an opaque handle (cast of id<MTLBuffer>), or 0 on failure.
    uint64_t create_buffer(size_t size);

    /// Create a new command buffer from the command queue.
    /// The returned handle must be committed via commit_and_wait().
    /// Returns 0 on failure.
    uint64_t new_command_buffer();

    /// Commit a command buffer and block until it completes.
    void commit_and_wait(uint64_t command_buffer_handle);

    /// Access the raw implementation (used by MetalBackend).
    MetalContextImpl* impl() const;

private:
    std::unique_ptr<MetalContextImpl> impl_;
};

}  // namespace nexus::compute
