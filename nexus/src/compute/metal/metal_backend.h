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

/// Parameters for the GPU-resident attention kernel.
/// K/V cache lives entirely on GPU — no CPU round-trip needed.
struct GPUAttentionParams {
    uint32_t seq_len;       // Number of KV entries AFTER writing the new token
    uint32_t num_heads;     // Number of query heads
    uint32_t num_kv_heads;  // Number of KV heads (GQA)
    uint32_t head_dim_q;    // Dimension per query head
    uint32_t head_dim_kv;   // Dimension per KV head
    float    scale;         // 1/sqrt(head_dim_q)
    uint32_t seq_pos;       // Position to write new K/V in cache
    uint32_t kv_stride;     // Stride between sequence positions in KV cache (= kv_dim)
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

    /// Fused INT4 uniform dequant + GEMV (no scales/zeros).
    /// Maps nibbles via (nibble - 8) * 0.125. Fastest path for NXF INT4 data.
    bool gemv_int4_uniform(buffer_id buf_activations, buffer_id buf_weights_q,
                           buffer_id buf_output,
                           uint32_t N, uint32_t K);

    /// GPU-resident INT4 GEMV: all buffers are already on-GPU.
    /// No CPU→GPU or GPU→CPU copies — activations stay resident between GEMMs.
    /// This is the zero-copy fast path for the GPU-resident activation pipeline.
    bool gemm_int4_gpu(buffer_id buf_in, buffer_id buf_weight,
                       buffer_id buf_out, uint32_t N, uint32_t K);

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

    /// Element-wise residual add:  output[i] = a[i] + b[i]
    bool residual_add(buffer_id buf_a, buffer_id buf_b,
                      buffer_id buf_output, uint32_t n);

    /// GPU buffer-to-buffer copy via blit encoder.
    /// Works correctly inside a persistent batch (inserts a barrier, uses
    /// a blit pass, then resumes the compute encoder).
    /// On UMA storageModeShared, this is a simple memcpy via the GPU command
    /// stream, keeping the copy ordered with respect to prior dispatches.
    bool buffer_copy(buffer_id src, buffer_id dst, size_t bytes);

    // ─── Fused Multi-Head Attention ──────────────────────────────────────

    struct FusedAttentionParams {
        uint32_t num_heads;
        uint32_t num_kv_heads;
        uint32_t head_dim_q;
        uint32_t head_dim_kv;
        uint32_t dot_dim;
        uint32_t out_head_dim;
        uint32_t seq_len;
        uint32_t kv_dim;
        float    scale;
    };

    /// ALL heads in ONE dispatch. KV cache is FP16 (uint16 buffers).
    bool fused_attention_decode(buffer_id buf_Q, buffer_id buf_K_cache,
                                buffer_id buf_V_cache, buffer_id buf_output,
                                const FusedAttentionParams& params);

    // ─── Attention ─────────────────────────────────────────────────────────

    /// Single-query decode attention over all KV.
    /// Q[num_heads, head_dim], K[num_kv_heads, seq_len, head_dim], etc.
    bool attention_decode(buffer_id buf_Q, buffer_id buf_K,
                          buffer_id buf_V, buffer_id buf_O,
                          const AttentionDecodeParams& params);

    /// GPU-resident attention: reads KV from persistent GPU cache buffers.
    /// Writes the new K/V at seq_pos, then computes GQA attention.
    /// All buffers stay on GPU — no flush/resume needed.
    bool attention_gpu(buffer_id buf_q, buffer_id buf_k_cache,
                       buffer_id buf_v_cache,
                       buffer_id buf_new_k, buffer_id buf_new_v,
                       buffer_id buf_output,
                       const GPUAttentionParams& params);

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

    // ─── Batched MoE expert GEMV ────────────────────────────────────────────

    /// Execute all active experts' SwiGLU in ONE dispatch.
    bool batched_expert_ffn(buffer_id buf_activations,
                            buffer_id buf_w1, buffer_id buf_w3, buffer_id buf_w2,
                            buffer_id buf_expert_ids, buffer_id buf_gate_weights,
                            buffer_id buf_output,
                            uint32_t hidden_dim, uint32_t expert_ffn_dim,
                            uint32_t num_active, uint32_t num_experts);

    // ─── Command buffer pipelining (batch mode) ───────────────────────────

    /// Begin a batch: create a single command buffer + encoder that will be
    /// reused across multiple dispatches.  While a batch is active,
    /// begin_compute()/end_compute() inside each operation will reuse the
    /// persistent encoder and insert memory barriers instead of
    /// commit+wait.
    void begin_batch();

    /// End the current batch: commit the command buffer and wait for all
    /// dispatched work to complete.  Must be paired with begin_batch().
    void end_batch();

    /// Returns true if a batch is currently active.
    bool in_batch() const;

private:
    std::unique_ptr<MetalBackendImpl> impl_;
};

}  // namespace nexus::compute
