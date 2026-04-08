/// NEXUS Inference Engine — Metal compute backend (Phase 2 implementation).
///
/// Each operation:
///   1. Retrieves (or builds) the pipeline state from the context cache
///   2. Creates a command buffer and compute encoder
///   3. Binds buffers/constants and dispatches with appropriate threadgroup sizes
///   4. Commits synchronously (Phase 3 will pipeline)
///
/// All buffers use storageModeShared for UMA zero-copy on Apple Silicon.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_backend.h"
#include "metal_context.h"
#include "metal_context_impl.h"

#include <cstring>
#include <algorithm>

namespace nexus::compute {

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Cast an opaque handle back to an MTLBuffer without transferring ownership.
static inline id<MTLBuffer> handle_to_buffer(uint64_t h) {
    return (__bridge id<MTLBuffer>)(void*)h;
}

/// Compute 1D threadgroup size: min(maxThreads, 1024), clamped to pipeline max.
static inline NSUInteger threadgroup_1d(id<MTLComputePipelineState> pso) {
    return std::min<NSUInteger>([pso maxTotalThreadsPerThreadgroup], 1024);
}

/// Ceiling division.
static inline uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

// ─── Opaque implementation ──────────────────────────────────────────────────

struct MetalBackendImpl {
    MetalContext* ctx = nullptr;

    // ── Persistent encoder state for batch mode ────────────────────────
    bool persistent_mode = false;
    id<MTLCommandBuffer> persistent_cb = nil;
    id<MTLComputeCommandEncoder> persistent_enc = nil;

    explicit MetalBackendImpl(MetalContext& context) : ctx(&context) {}

    /// Convenience: get the context impl.
    MetalContextImpl* ci() const { return ctx->impl(); }

    /// Get or build a pipeline for the given kernel name.
    id<MTLComputePipelineState> pipeline(const std::string& name) {
        auto* c = ci();
        auto pso = c->get_pipeline(name);
        if (!pso) pso = c->build_pipeline(name);
        return pso;
    }

    /// Begin a batch — create one command buffer + encoder that persists
    /// across multiple dispatches.
    void begin_batch() {
        if (persistent_mode) return;  // Already in batch
        auto* c = ci();
        persistent_cb = c->make_command_buffer();
        if (persistent_cb) {
            persistent_enc = [persistent_cb computeCommandEncoder];
            persistent_mode = true;
        }
    }

    /// End the batch — commit and wait for all dispatched work.
    void end_batch() {
        if (!persistent_mode) return;
        if (persistent_enc) {
            [persistent_enc endEncoding];
            persistent_enc = nil;
        }
        if (persistent_cb) {
            [persistent_cb commit];
            [persistent_cb waitUntilCompleted];
            if (persistent_cb.status == MTLCommandBufferStatusError) {
                NSError* err = persistent_cb.error;
                fprintf(stderr, "[metal] Batch command buffer error: %s\n",
                        err ? [[err localizedDescription] UTF8String] : "unknown");
            }
            persistent_cb = nil;
        }
        persistent_mode = false;
    }

    /// Create a command buffer + compute encoder pair.  In persistent mode,
    /// returns the existing encoder (inserting a memory barrier for
    /// correctness between dependent dispatches).  Returns nil encoder
    /// on failure.
    std::pair<id<MTLCommandBuffer>, id<MTLComputeCommandEncoder>>
    begin_compute() {
        if (persistent_mode && persistent_enc) {
            // Reuse the persistent encoder — insert a barrier so that the
            // previous dispatch's writes are visible to the next dispatch.
            [persistent_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            return {persistent_cb, persistent_enc};
        }
        // Non-batch mode: create a fresh command buffer + encoder.
        auto* c = ci();
        id<MTLCommandBuffer> cb = c->make_command_buffer();
        if (!cb) return {nil, nil};
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        return {cb, enc};
    }

    /// End encode, commit, and wait.  In persistent mode, this is a no-op
    /// (the batch's end_batch() handles the commit).
    void end_compute(id<MTLCommandBuffer> cb,
                     id<MTLComputeCommandEncoder> enc) {
        if (persistent_mode) {
            // Don't end encoding or commit — the batch owns the encoder.
            // The memory barrier in the next begin_compute() ensures
            // correct ordering between dependent dispatches.
            return;
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            NSError* err = cb.error;
            fprintf(stderr, "[metal] Command buffer error: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "unknown");
        }
    }
};

// ─── MetalBackend ───────────────────────────────────────────────────────────

MetalBackend::MetalBackend(MetalContext& ctx)
    : impl_(std::make_unique<MetalBackendImpl>(ctx))
{
}

MetalBackend::~MetalBackend() = default;

MetalBackend::MetalBackend(MetalBackend&&) noexcept = default;
MetalBackend& MetalBackend::operator=(MetalBackend&&) noexcept = default;

bool MetalBackend::is_ready() const {
    return impl_ && impl_->ctx && impl_->ctx->is_available();
}

// ─── GEMM FP32 ─────────────────────────────────────────────────────────────

bool MetalBackend::gemm_f32(buffer_id buf_A, buffer_id buf_B, buffer_id buf_C,
                            uint32_t M, uint32_t N, uint32_t K)
{
    if (!is_ready()) return false;

    id<MTLComputePipelineState> pso = impl_->pipeline("gemm_f32_tiled");
    if (!pso) return false;

    auto [cb, enc] = impl_->begin_compute();
    if (!enc) return false;

    [enc setComputePipelineState:pso];

    // Bind buffers
    [enc setBuffer:handle_to_buffer(buf_A) offset:0 atIndex:0];
    [enc setBuffer:handle_to_buffer(buf_B) offset:0 atIndex:1];
    [enc setBuffer:handle_to_buffer(buf_C) offset:0 atIndex:2];

    // Bind dimension constants
    [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&N length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&K length:sizeof(uint32_t) atIndex:5];

    // 32x32 tile dispatch
    constexpr uint32_t TILE = 32;
    MTLSize threadgroup_size = MTLSizeMake(TILE, TILE, 1);
    MTLSize grid_size = MTLSizeMake(ceil_div(N, TILE), ceil_div(M, TILE), 1);

    [enc dispatchThreadgroups:grid_size
        threadsPerThreadgroup:threadgroup_size];

    impl_->end_compute(cb, enc);
    return true;
}

// ─── Dequantized GEMV (INT4) ───────────────────────────────────────────────

bool MetalBackend::gemv_dequant_int4(buffer_id buf_activations,
                                     buffer_id buf_weights_q,
                                     buffer_id buf_scales, buffer_id buf_zeros,
                                     buffer_id buf_output,
                                     uint32_t M, uint32_t N, uint32_t K,
                                     uint32_t group_size)
{
    if (!is_ready()) return false;

    id<MTLComputePipelineState> pso = impl_->pipeline("gemv_dequant_int4");
    if (!pso) return false;

    auto [cb, enc] = impl_->begin_compute();
    if (!enc) return false;

    [enc setComputePipelineState:pso];

    [enc setBuffer:handle_to_buffer(buf_activations) offset:0 atIndex:0];
    [enc setBuffer:handle_to_buffer(buf_weights_q)   offset:0 atIndex:1];
    [enc setBuffer:handle_to_buffer(buf_scales)      offset:0 atIndex:2];
    [enc setBuffer:handle_to_buffer(buf_zeros)       offset:0 atIndex:3];
    [enc setBuffer:handle_to_buffer(buf_output)      offset:0 atIndex:4];

    // DequantParams struct (matches the Metal shader layout)
    struct {
        uint32_t rows;
        uint32_t cols;
        uint32_t K;
        uint32_t group_size;
    } params = { M, N, K, group_size };

    [enc setBytes:&params length:sizeof(params) atIndex:5];

    // 1D dispatch: one thread per output column
    NSUInteger tg = threadgroup_1d(pso);
    MTLSize threadgroup_size = MTLSizeMake(tg, 1, 1);
    MTLSize grid_size = MTLSizeMake(ceil_div(N, (uint32_t)tg), 1, 1);

    [enc dispatchThreadgroups:grid_size
        threadsPerThreadgroup:threadgroup_size];

    impl_->end_compute(cb, enc);
    return true;
}

// ─── Fused INT4 Uniform GEMV ────────────────────────────────────────────────

bool MetalBackend::gemv_int4_uniform(buffer_id buf_activations,
                                      buffer_id buf_weights_q,
                                      buffer_id buf_output,
                                      uint32_t N, uint32_t K)
{
    if (!is_ready()) return false;

    // Use proven uniform GEMV shader (correct results verified at 5.9 tok/s)
    // TODO: Debug gemv_int4_fast shader memory access patterns
    id<MTLComputePipelineState> pso = impl_->pipeline("gemv_int4_uniform");
    if (!pso) return false;

    auto [cb, enc] = impl_->begin_compute();
    if (!enc) return false;

    [enc setComputePipelineState:pso];
    [enc setBuffer:handle_to_buffer(buf_activations) offset:0 atIndex:0];
    [enc setBuffer:handle_to_buffer(buf_weights_q) offset:0 atIndex:1];
    [enc setBuffer:handle_to_buffer(buf_output) offset:0 atIndex:2];

    // Params struct matching the shader
    struct { uint32_t N; uint32_t K; } params = { N, K };
    [enc setBytes:&params length:sizeof(params) atIndex:3];

    // 1D dispatch: one thread per output column
    NSUInteger tg = threadgroup_1d(pso);
    MTLSize threadgroup_size = MTLSizeMake(tg, 1, 1);
    MTLSize grid_size = MTLSizeMake((N + tg - 1) / tg, 1, 1);
    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:threadgroup_size];

    impl_->end_compute(cb, enc);
    return true;
}

// ─── RMSNorm ────────────────────────────────────────────────────────────────

bool MetalBackend::rmsnorm(buffer_id buf_input, buffer_id buf_weight,
                           buffer_id buf_output,
                           uint32_t dim, float eps)
{
    if (!is_ready()) return false;

    id<MTLComputePipelineState> pso = impl_->pipeline("rmsnorm");
    if (!pso) return false;

    auto [cb, enc] = impl_->begin_compute();
    if (!enc) return false;

    [enc setComputePipelineState:pso];

    [enc setBuffer:handle_to_buffer(buf_input)  offset:0 atIndex:0];
    [enc setBuffer:handle_to_buffer(buf_weight) offset:0 atIndex:1];
    [enc setBuffer:handle_to_buffer(buf_output) offset:0 atIndex:2];
    [enc setBytes:&dim length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&eps length:sizeof(float)    atIndex:4];

    // One threadgroup per row.  threadgroup size = min(dim, maxThreads, 1024).
    NSUInteger tg = std::min<NSUInteger>(dim, threadgroup_1d(pso));
    // Round down to power of 2 for the tree reduction to work correctly.
    NSUInteger tg_pow2 = 1;
    while (tg_pow2 * 2 <= tg) tg_pow2 *= 2;

    MTLSize threadgroup_size = MTLSizeMake(tg_pow2, 1, 1);
    // Single row for now (Phase 2 processes one row at a time).
    MTLSize grid_size = MTLSizeMake(1, 1, 1);

    [enc dispatchThreadgroups:grid_size
        threadsPerThreadgroup:threadgroup_size];

    impl_->end_compute(cb, enc);
    return true;
}

// ─── SiLU activation ────────────────────────────────────────────────────────

bool MetalBackend::silu(buffer_id buf_input, buffer_id buf_output, uint32_t n)
{
    if (!is_ready()) return false;

    id<MTLComputePipelineState> pso = impl_->pipeline("silu");
    if (!pso) return false;

    auto [cb, enc] = impl_->begin_compute();
    if (!enc) return false;

    [enc setComputePipelineState:pso];

    [enc setBuffer:handle_to_buffer(buf_input)  offset:0 atIndex:0];
    [enc setBuffer:handle_to_buffer(buf_output) offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];

    NSUInteger tg = threadgroup_1d(pso);
    MTLSize threadgroup_size = MTLSizeMake(tg, 1, 1);
    MTLSize grid_size = MTLSizeMake(ceil_div(n, (uint32_t)tg), 1, 1);

    [enc dispatchThreadgroups:grid_size
        threadsPerThreadgroup:threadgroup_size];

    impl_->end_compute(cb, enc);
    return true;
}

// ─── Fused SwiGLU ───────────────────────────────────────────────────────────

bool MetalBackend::swiglu_fused(buffer_id buf_gate, buffer_id buf_up,
                                buffer_id buf_output, uint32_t n)
{
    if (!is_ready()) return false;

    id<MTLComputePipelineState> pso = impl_->pipeline("swiglu_fused");
    if (!pso) return false;

    auto [cb, enc] = impl_->begin_compute();
    if (!enc) return false;

    [enc setComputePipelineState:pso];

    [enc setBuffer:handle_to_buffer(buf_gate)   offset:0 atIndex:0];
    [enc setBuffer:handle_to_buffer(buf_up)     offset:0 atIndex:1];
    [enc setBuffer:handle_to_buffer(buf_output) offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(uint32_t) atIndex:3];

    NSUInteger tg = threadgroup_1d(pso);
    MTLSize threadgroup_size = MTLSizeMake(tg, 1, 1);
    MTLSize grid_size = MTLSizeMake(ceil_div(n, (uint32_t)tg), 1, 1);

    [enc dispatchThreadgroups:grid_size
        threadsPerThreadgroup:threadgroup_size];

    impl_->end_compute(cb, enc);
    return true;
}

// ─── Attention decode ───────────────────────────────────────────────────────

bool MetalBackend::attention_decode(buffer_id buf_Q, buffer_id buf_K,
                                    buffer_id buf_V, buffer_id buf_O,
                                    const AttentionDecodeParams& params)
{
    if (!is_ready()) return false;

    id<MTLComputePipelineState> pso =
        impl_->pipeline("attention_decode_single_head");
    if (!pso) return false;

    auto [cb, enc] = impl_->begin_compute();
    if (!enc) return false;

    [enc setComputePipelineState:pso];

    [enc setBuffer:handle_to_buffer(buf_Q) offset:0 atIndex:0];
    [enc setBuffer:handle_to_buffer(buf_K) offset:0 atIndex:1];
    [enc setBuffer:handle_to_buffer(buf_V) offset:0 atIndex:2];
    [enc setBuffer:handle_to_buffer(buf_O) offset:0 atIndex:3];

    // AttentionParams matches the shader struct layout.
    struct {
        uint32_t seq_len;
        uint32_t num_heads;
        uint32_t num_kv_heads;
        uint32_t head_dim;
        float    scale;
    } gpu_params = {
        params.seq_len,
        params.num_heads,
        params.num_kv_heads,
        params.head_dim,
        params.scale
    };
    [enc setBytes:&gpu_params length:sizeof(gpu_params) atIndex:4];

    // One threadgroup per head.  The kernel uses tid within the threadgroup
    // to iterate over KV positions.
    NSUInteger tg = threadgroup_1d(pso);
    MTLSize threadgroup_size = MTLSizeMake(tg, 1, 1);
    // Grid: one threadgroup per query head.
    MTLSize grid_size = MTLSizeMake(params.num_heads, 1, 1);

    [enc dispatchThreadgroups:grid_size
        threadsPerThreadgroup:threadgroup_size];

    impl_->end_compute(cb, enc);
    return true;
}

// ─── Buffer management (UMA zero-copy) ──────────────────────────────────────

MetalBackend::buffer_id MetalBackend::alloc_shared_buffer(size_t bytes)
{
    if (!is_ready()) return 0;
    auto* ci = impl_->ci();
    id<MTLBuffer> buf = ci->create_shared_buffer(bytes);
    if (!buf) return 0;
    // Transfer ownership out via bridging cast.
    return reinterpret_cast<uint64_t>((__bridge_retained void*)buf);
}

// ─── Batched Expert FFN ─────────────────────────────────────────────────────

bool MetalBackend::batched_expert_ffn(buffer_id buf_activations,
                                       buffer_id buf_w1, buffer_id buf_w3, buffer_id buf_w2,
                                       buffer_id buf_expert_ids, buffer_id buf_gate_weights,
                                       buffer_id buf_output,
                                       uint32_t hidden_dim, uint32_t expert_ffn_dim,
                                       uint32_t num_active, uint32_t num_experts)
{
    if (!is_ready()) return false;

    id<MTLComputePipelineState> pso = impl_->pipeline("batched_expert_gemv");
    if (!pso) return false;

    auto [cb, enc] = impl_->begin_compute();
    if (!enc) return false;

    [enc setComputePipelineState:pso];
    [enc setBuffer:handle_to_buffer(buf_activations) offset:0 atIndex:0];
    [enc setBuffer:handle_to_buffer(buf_w1) offset:0 atIndex:1];
    [enc setBuffer:handle_to_buffer(buf_w3) offset:0 atIndex:2];
    [enc setBuffer:handle_to_buffer(buf_w2) offset:0 atIndex:3];
    [enc setBuffer:handle_to_buffer(buf_expert_ids) offset:0 atIndex:4];
    [enc setBuffer:handle_to_buffer(buf_gate_weights) offset:0 atIndex:5];
    [enc setBuffer:handle_to_buffer(buf_output) offset:0 atIndex:6];

    struct {
        uint32_t hidden_dim;
        uint32_t expert_ffn_dim;
        uint32_t num_active;
        uint32_t num_experts;
    } params = { hidden_dim, expert_ffn_dim, num_active, num_experts };
    [enc setBytes:&params length:sizeof(params) atIndex:7];

    // One threadgroup per active expert
    NSUInteger tg = std::min<NSUInteger>([pso maxTotalThreadsPerThreadgroup], 256);
    MTLSize threadgroup_size = MTLSizeMake(tg, 1, 1);
    MTLSize grid_size = MTLSizeMake(num_active, 1, 1);
    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:threadgroup_size];

    impl_->end_compute(cb, enc);
    return true;
}

MetalBackend::buffer_id MetalBackend::wrap_pointer(void* ptr, size_t bytes)
{
    if (!is_ready() || !ptr || bytes == 0) return 0;
    auto* ci = impl_->ci();
    // newBufferWithBytesNoCopy: wraps existing memory as an MTLBuffer.
    // On Apple Silicon UMA, this is true zero-copy — the GPU accesses
    // the SAME physical pages as the CPU, no allocation or memcpy.
    // The pointer must be page-aligned (16KB on arm64).
    id<MTLBuffer> buf = [ci->device newBufferWithBytesNoCopy:ptr
                                                      length:bytes
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
    if (!buf) {
        // Fallback: pointer might not be page-aligned. Allocate and copy.
        buf = [ci->device newBufferWithBytes:ptr
                                      length:bytes
                                     options:MTLResourceStorageModeShared];
    }
    if (!buf) return 0;
    return reinterpret_cast<uint64_t>((__bridge_retained void*)buf);
}

bool MetalBackend::copy_to_buffer(buffer_id buffer_handle, const void* data,
                                  size_t size)
{
    if (buffer_handle == 0 || !data || size == 0) return false;
    id<MTLBuffer> buf = handle_to_buffer(buffer_handle);
    if ([buf length] < size) {
        NSLog(@"NEXUS: copy_to_buffer: buffer length %lu < requested %zu",
              (unsigned long)[buf length], size);
        return false;
    }
    // On Apple Silicon with storageModeShared this is a direct write into
    // unified memory — the GPU will see it with no additional copy.
    std::memcpy([buf contents], data, size);
    return true;
}

void* MetalBackend::buffer_contents(buffer_id buffer_handle)
{
    if (buffer_handle == 0) return nullptr;
    id<MTLBuffer> buf = handle_to_buffer(buffer_handle);
    return [buf contents];
}

void MetalBackend::free_buffer(buffer_id buffer_handle)
{
    if (buffer_handle == 0) return;
    // Transfer ownership back to ARC, which will release immediately.
    (void)(__bridge_transfer id<MTLBuffer>)(void*)buffer_handle;
}

// ─── Command buffer pipelining (batch mode) ────────────────────────────────

void MetalBackend::begin_batch()
{
    if (impl_) impl_->begin_batch();
}

void MetalBackend::end_batch()
{
    if (impl_) impl_->end_batch();
}

bool MetalBackend::in_batch() const
{
    return impl_ && impl_->persistent_mode;
}

}  // namespace nexus::compute
