#pragma once
/// NEXUS Inference Engine — Metal compute backend (Phase 2).
///
/// Provides GPU-accelerated GEMM, attention, normalization, activations,
/// and dequantization via Metal compute shaders on Apple Silicon.

#include <cstddef>
#include <cstdint>
#include <memory>

namespace nexus::compute {

class MetalContext;
struct MetalBackendImpl;

/// Parameters for the flash-attention decode kernel.
struct AttentionDecodeParams {
    uint32_t seq_len;       // Total KV sequence length
    uint32_t num_heads;     // Number of query heads
    uint32_t num_kv_heads;  // Number of KV heads (GQA)
    uint32_t head_dim;      // Dimension per head
    float    scale;         // 1/sqrt(head_dim)
};

class MetalBackend {
public:
    /// Opaque buffer handle (cast of id<MTLBuffer>).
    using buffer_id = uint64_t;

    /// Construct with an existing Metal context (must outlive this object).
    explicit MetalBackend(MetalContext& ctx);
    ~MetalBackend();

    // Non-copyable, movable.
    MetalBackend(const MetalBackend&) = delete;
    MetalBackend& operator=(const MetalBackend&) = delete;
    MetalBackend(MetalBackend&&) noexcept;
    MetalBackend& operator=(MetalBackend&&) noexcept;

    /// Returns true if the backend is ready to dispatch GPU work.
    bool is_ready() const;

    // ─── GEMM ──────────────────────────────────────────────────────────────

    /// GPU FP32 tiled GEMM:  C[M,N] = A[M,K] x B[K,N]
    /// A, B, C must be handles to storageModeShared MTLBuffers.
    bool gemm_f32(buffer_id buf_A, buffer_id buf_B, buffer_id buf_C,
                  uint32_t M, uint32_t N, uint32_t K);

    // ─── Dequantized GEMV ──────────────────────────────────────────────────

    /// Fused INT4 dequant + GEMV:  output[1,N] = activations[1,K] x W_q[K,N]
    bool gemv_dequant_int4(buffer_id buf_activations, buffer_id buf_weights_q,
                           buffer_id buf_scales, buffer_id buf_zeros,
                           buffer_id buf_output,
                           uint32_t M, uint32_t N, uint32_t K,
                           uint32_t group_size);

    // ─── Normalization ─────────────────────────────────────────────────────

    /// RMSNorm:  output = input * weight / sqrt(mean(input^2) + eps)
    bool rmsnorm(buffer_id buf_input, buffer_id buf_weight,
                 buffer_id buf_output, uint32_t dim, float eps);

    // ─── Activations ───────────────────────────────────────────────────────

    /// SiLU (Swish) activation:  output = x * sigmoid(x)
    bool silu(buffer_id buf_input, buffer_id buf_output, uint32_t n);

    /// Fused SwiGLU:  output = silu(gate) * up
    bool swiglu_fused(buffer_id buf_gate, buffer_id buf_up,
                      buffer_id buf_output, uint32_t n);

    // ─── Attention ─────────────────────────────────────────────────────────

    /// Single-query decode attention over all KV.
    /// Q[num_heads, head_dim], K[num_kv_heads, seq_len, head_dim], etc.
    bool attention_decode(buffer_id buf_Q, buffer_id buf_K,
                          buffer_id buf_V, buffer_id buf_O,
                          const AttentionDecodeParams& params);

    // ─── Buffer management (UMA zero-copy) ─────────────────────────────────

    /// Allocate a storageModeShared MTLBuffer.  Returns handle or 0.
    buffer_id alloc_shared_buffer(size_t bytes);

    /// Wrap an existing mmap'd pointer as an MTLBuffer (TRUE zero-copy on UMA).
    /// The pointer must be page-aligned and the memory must stay valid while
    /// the buffer is in use. This is the fastest path — no allocation, no copy.
    buffer_id wrap_pointer(void* ptr, size_t bytes);

    /// Copy CPU data into a shared buffer (memcpy — zero-copy on UMA means
    /// this writes directly to GPU-visible memory).
    bool copy_to_buffer(buffer_id buffer_handle, const void* data, size_t size);

    /// Get the raw CPU pointer for a shared buffer (UMA zero-copy).
    void* buffer_contents(buffer_id buffer_handle);

    /// Release a buffer previously allocated with alloc_shared_buffer.
    void free_buffer(buffer_id buffer_handle);

private:
    std::unique_ptr<MetalBackendImpl> impl_;
};

}  // namespace nexus::compute
