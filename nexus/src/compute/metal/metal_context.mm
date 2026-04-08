/// NEXUS Inference Engine — Metal GPU context implementation (Phase 2).

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_context.h"
#include "metal_context_impl.h"

namespace nexus::compute {

// ─── MetalContext public API ────────────────────────────────────────────────

MetalContext::MetalContext()
    : impl_(std::make_unique<MetalContextImpl>())
{
}

MetalContext::~MetalContext() = default;

MetalContext::MetalContext(MetalContext&&) noexcept = default;
MetalContext& MetalContext::operator=(MetalContext&&) noexcept = default;

bool MetalContext::is_available() const {
    return impl_ && impl_->device != nil;
}

std::string MetalContext::device_name() const {
    if (!is_available()) return "(no Metal device)";
    NSString* name = [impl_->device name];
    return std::string([name UTF8String]);
}

size_t MetalContext::recommended_max_working_set_size() const {
    if (!is_available()) return 0;
    return static_cast<size_t>([impl_->device recommendedMaxWorkingSetSize]);
}

bool MetalContext::load_library(const std::string& path) {
    if (!is_available()) return false;

    NSError* error = nil;
    NSString* ns_path = [NSString stringWithUTF8String:path.c_str()];
    NSURL* url = [NSURL fileURLWithPath:ns_path];

    impl_->library = [impl_->device newLibraryWithURL:url error:&error];

    if (error) {
        NSLog(@"NEXUS: Failed to load Metal library at %@: %@",
              ns_path, [error localizedDescription]);
        return false;
    }

    // Clear pipeline cache when loading a new library.
    impl_->pipeline_cache.clear();

    NSLog(@"NEXUS: Loaded Metal library from %@", ns_path);
    return impl_->library != nil;
}

bool MetalContext::make_pipeline(const std::string& function_name) {
    if (!is_available()) return false;
    return impl_->build_pipeline(function_name) != nil;
}

uint64_t MetalContext::create_buffer(size_t size) {
    if (!is_available()) return 0;
    id<MTLBuffer> buf = impl_->create_shared_buffer(size);
    if (!buf) return 0;
    return reinterpret_cast<uint64_t>((__bridge_retained void*)buf);
}

uint64_t MetalContext::new_command_buffer() {
    if (!is_available()) return 0;
    id<MTLCommandBuffer> cb = impl_->make_command_buffer();
    if (!cb) return 0;
    return reinterpret_cast<uint64_t>((__bridge_retained void*)cb);
}

void MetalContext::commit_and_wait(uint64_t command_buffer_handle) {
    if (command_buffer_handle == 0) return;
    id<MTLCommandBuffer> cb =
        (__bridge_transfer id<MTLCommandBuffer>)(void*)command_buffer_handle;
    [cb commit];
    [cb waitUntilCompleted];
}

MetalContextImpl* MetalContext::impl() const {
    return impl_.get();
}

}  // namespace nexus::compute
