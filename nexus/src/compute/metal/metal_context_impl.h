#pragma once
/// NEXUS Inference Engine — MetalContextImpl definition (internal header).
///
/// Shared between metal_context.mm and metal_backend.mm so both can access
/// the Metal runtime objects directly without duplicating the struct.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <string>
#include <unordered_map>

namespace nexus::compute {

struct MetalContextImpl {
    id<MTLDevice>        device        = nil;
    id<MTLCommandQueue>  command_queue = nil;
    id<MTLLibrary>       library       = nil;

    /// Cached compute pipeline states, keyed by kernel function name.
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipeline_cache;

    MetalContextImpl() {
        device = MTLCreateSystemDefaultDevice();
        if (device) {
            command_queue = [device newCommandQueue];
        }
    }

    ~MetalContextImpl() {
        pipeline_cache.clear();
        library       = nil;
        command_queue  = nil;
        device         = nil;
    }

    // ── Pipeline cache ──────────────────────────────────────────────────

    /// Get an existing pipeline or return nil.
    id<MTLComputePipelineState> get_pipeline(const std::string& name) const {
        auto it = pipeline_cache.find(name);
        if (it != pipeline_cache.end()) return it->second;
        return nil;
    }

    /// Build a pipeline for the named function and cache it.
    /// Returns nil on failure.
    id<MTLComputePipelineState> build_pipeline(const std::string& name) {
        if (!library) {
            NSLog(@"NEXUS: Cannot build pipeline '%s' — no library loaded.",
                  name.c_str());
            return nil;
        }

        // Check cache first
        auto it = pipeline_cache.find(name);
        if (it != pipeline_cache.end()) return it->second;

        NSString* ns_name = [NSString stringWithUTF8String:name.c_str()];
        id<MTLFunction> function = [library newFunctionWithName:ns_name];
        if (!function) {
            NSLog(@"NEXUS: Kernel function '%@' not found in library.", ns_name);
            return nil;
        }

        NSError* error = nil;
        id<MTLComputePipelineState> pso =
            [device newComputePipelineStateWithFunction:function error:&error];

        if (error || !pso) {
            NSLog(@"NEXUS: Failed to create pipeline for '%@': %@",
                  ns_name, [error localizedDescription]);
            return nil;
        }

        pipeline_cache[name] = pso;
        NSLog(@"NEXUS: Pipeline '%@' created (maxThreads=%lu).",
              ns_name, (unsigned long)[pso maxTotalThreadsPerThreadgroup]);
        return pso;
    }

    // ── Buffer creation ─────────────────────────────────────────────────

    id<MTLBuffer> create_shared_buffer(size_t size) {
        if (!device) return nil;
        return [device newBufferWithLength:size
                                  options:MTLResourceStorageModeShared];
    }

    // ── Command buffer ──────────────────────────────────────────────────

    id<MTLCommandBuffer> make_command_buffer() {
        if (!command_queue) return nil;
        return [command_queue commandBuffer];
    }
};

}  // namespace nexus::compute
