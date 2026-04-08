#pragma once
/// NEXUS Compute Dispatch — Unified CPU/GPU compute interface.
///
/// Automatically routes operations to Metal GPU when available,
/// falling back to Accelerate/NEON on CPU. Handles UMA buffer
/// management so callers don't need to know which backend is active.

#include "core/config.h"
#include "compute/metal/metal_context.h"
#include "compute/metal/metal_backend.h"
#include <vector>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace nexus::compute {

/// Unified compute dispatcher.
/// Call init() once at startup. Then use gemm/rmsnorm/silu methods
/// which auto-route to GPU or CPU.
class ComputeDispatch {
public:
    ComputeDispatch();
    ~ComputeDispatch();

    /// Initialize Metal GPU. Returns true if GPU is available.
    /// shader_path: path to nexus_shaders.metallib
    bool init_gpu(const std::string& shader_path);

    /// Is GPU available?
    bool has_gpu() const;

    /// GPU device name.
    std::string gpu_name() const;

    // ─── Matrix operations ──────────────────────────────────────────────

    /// GEMM: C[M,N] = A[M,K] × B[K,N]
    /// Same signature as compute::gemm_f32 for drop-in replacement.
    /// If GPU available, dispatches Metal. Otherwise Accelerate cblas_sgemm.
    void gemm(const float* A, const float* B, float* C,
              int M, int N, int K);

    /// Fused INT4 dequant + GEMV on GPU (falls back to CPU dequant+GEMM).
    /// weights_q is packed INT4, scales/zeros are FP32 per-group.
    void gemv_int4(float* out, const float* activations,
                   const uint8_t* weights_q, const float* scales, const float* zeros,
                   int M, int N, int K, int group_size);

    /// GEMM with raw INT4 weights — UMA zero-copy path.
    /// Wraps the mmap'd INT4 pointer directly as an MTLBuffer, dispatches
    /// fused dequant+GEMV on GPU. Falls back to CPU dequant+GEMM.
    /// This is the FAST PATH — no CPU dequant, no memcpy, no allocation.
    /// @param activations FP32 input [M, K]
    /// @param weights_int4 Raw mmap'd INT4 packed data (2 values per byte)
    /// @param weights_bytes Size of the INT4 data in bytes
    /// @param out FP32 output [M, N]
    void gemm_int4(const float* activations, const void* weights_int4,
                   size_t weights_bytes, float* out,
                   int M, int N, int K);

    // ─── Element-wise operations ────────────────────────────────────────

    /// RMSNorm: out = x * weight / sqrt(mean(x^2) + eps)
    void rmsnorm(float* out, const float* x, const float* weight, int dim, float eps);

    /// SiLU activation: out = x * sigmoid(x)
    void silu(float* out, const float* x, int n);

    /// Fused SwiGLU: out = silu(gate) * up
    void swiglu(float* out, const float* gate, const float* up, int n);

    /// Softmax in-place over dim elements
    void softmax(float* data, int dim);

    // ─── Batched MoE expert execution ─────────────────────────────────

    /// Execute all active experts' SwiGLU FFN in ONE GPU dispatch.
    /// expert_ids: array of active expert indices
    /// gate_weights: corresponding gate scores (normalized)
    /// w1/w2/w3_raw: full packed INT4 expert weight tensors
    /// output: accumulated weighted result [hidden_dim]
    void batched_moe_ffn(const float* activations, int hidden_dim, int expert_ffn_dim,
                         const int* expert_ids, const float* gate_weights, int num_active,
                         int num_experts,
                         const void* w1_raw, size_t w1_bytes,
                         const void* w2_raw, size_t w2_bytes,
                         const void* w3_raw, size_t w3_bytes,
                         float* output);

    // ─── Batch mode (command buffer pipelining) ────────────────────────

    /// Begin a GPU batch: all subsequent GPU dispatches share a single
    /// command buffer. Call end_gpu_batch() when done to commit and wait.
    void begin_gpu_batch();

    /// End a GPU batch: commit the shared command buffer and synchronize.
    void end_gpu_batch();

    // ─── Token-level batching (single commit per token) ──────────────

    /// Begin a full-token batch: creates ONE Metal command buffer for the
    /// entire decode step (all layers, all dispatches). Every GPU operation
    /// between begin_token() and end_token() encodes without committing.
    void begin_token();

    /// End the full-token batch: commit the single command buffer and wait.
    /// This is the ONLY Metal commit per token.
    void end_token();

    /// Returns true if we're inside a begin_token()/end_token() block
    /// AND a command buffer is actively encoding (not between flush/resume).
    bool in_token_batch() const { return token_batch_active_ && token_batch_encoding_; }

    /// Flush the current batch mid-token: commit+wait so GPU results become
    /// readable on CPU (e.g., for attention that needs CPU softmax + KV cache).
    /// After flush, call resume_token() to start a new command buffer for
    /// the remaining dispatches.
    void flush_token();

    /// Resume batching after a flush_token(). Creates a new command buffer
    /// for the remaining GPU work in this token.
    void resume_token();

    /// Upload a small CPU tensor (e.g., RMSNorm weights) as a persistent
    /// GPU buffer. Returns a cached buffer_id. Thread-safe for repeated calls
    /// with the same pointer (returns cached ID).
    MetalBackend::buffer_id upload_small_buffer(const void* cpu_ptr, size_t bytes);

    // ─── GPU-Resident Activation Pipeline ─────────────────────────────────
    // Keeps activations ON the GPU between GEMMs within a layer.
    // Eliminates 420 CPU↔GPU round-trips per token (7 GEMMs × 60 layers).

    /// Pre-allocated persistent GPU buffers for intermediate activations.
    /// Allocated once at model load, reused every token. Activations NEVER
    /// leave the GPU between GEMMs within a layer.
    struct GPUBufferPool {
        MetalBackend::buffer_id hidden;      // [hidden_dim] float
        MetalBackend::buffer_id residual;    // [hidden_dim] float
        MetalBackend::buffer_id norm_buf;    // [hidden_dim] float
        MetalBackend::buffer_id q_buf;       // [max_q_dim] float
        MetalBackend::buffer_id k_buf;       // [max_kv_dim] float
        MetalBackend::buffer_id v_buf;       // [max_kv_dim] float
        MetalBackend::buffer_id attn_out;    // [max_q_dim] float
        MetalBackend::buffer_id ffn_gate;    // [max_ffn_dim] float
        MetalBackend::buffer_id ffn_up;      // [max_ffn_dim] float
        MetalBackend::buffer_id ffn_out;     // [hidden_dim] float
        MetalBackend::buffer_id logits;      // [vocab_size] float

        // GPU-resident KV cache: keeps K/V on GPU so attention never
        // needs a flush/resume. Indexed as [layer * max_seq * kv_dim + pos * kv_dim].
        MetalBackend::buffer_id kv_keys;     // [num_layers * max_seq * kv_dim] float
        MetalBackend::buffer_id kv_values;   // [num_layers * max_seq * kv_dim] float
        int kv_max_seq = 0;
        int kv_dim = 0;
        int kv_num_layers = 0;
    };

    /// Allocate the GPU buffer pool once at model load time.
    /// All buffers are persistent storageModeShared MTLBuffers that stay
    /// allocated for the lifetime of the engine.
    bool init_gpu_pool(uint32_t hidden_dim, uint32_t max_q_dim,
                       uint32_t max_kv_dim, uint32_t max_ffn_dim,
                       uint32_t vocab_size);

    /// Allocate the GPU-resident KV cache. Call after init_gpu_pool().
    /// For decode (short sequences), this is manageable:
    ///   e.g. 60 layers * 100 seq * 8192 kv_dim * 4 bytes = ~188 MB.
    bool init_kv_cache(int num_layers, int max_seq, int kv_dim);

    /// GPU-resident attention: Q/K/V and KV cache all stay on GPU.
    /// No flush/resume needed — the entire attention runs as GPU dispatches.
    /// Writes new K/V into the GPU KV cache at seq_pos, then computes
    /// GQA attention over all cached KV, writing output to buf_output.
    bool attention_gpu(MetalBackend::buffer_id buf_q,
                       MetalBackend::buffer_id buf_new_k,
                       MetalBackend::buffer_id buf_new_v,
                       MetalBackend::buffer_id buf_output,
                       int layer_idx, int seq_pos,
                       int num_heads, int num_kv_heads,
                       int head_dim_q, int head_dim_kv);

    /// Fused multi-head attention: ALL heads in 1 dispatch.
    /// Q is FP32 on CPU. KV cache is FP16 (uint16*) on CPU.
    /// Uploads to GPU, dispatches fused kernel, downloads result.
    bool fused_attention(const float* q, int q_dim,
                         const float* k_new, const float* v_new, int kv_dim,
                         uint16_t* k_cache, uint16_t* v_cache,
                         int seq_pos, int max_seq,
                         int num_heads, int num_kv_heads,
                         int head_dim_q, int head_dim_kv,
                         float* output, int out_dim);

    /// GPU-resident INT4 GEMV: input and output are GPU buffer IDs.
    /// No CPU↔GPU copies — dispatches directly on resident buffers.
    bool gemm_int4_gpu(MetalBackend::buffer_id buf_in,
                       MetalBackend::buffer_id buf_weight,
                       MetalBackend::buffer_id buf_out,
                       uint32_t M, uint32_t N, uint32_t K);

    /// Copy CPU data to a GPU pool buffer (called once at start of layer).
    bool upload_to_gpu(const void* cpu_ptr, MetalBackend::buffer_id gpu_buf,
                       size_t size);

    /// Copy GPU pool buffer to CPU (called once at end of layer).
    bool download_from_gpu(MetalBackend::buffer_id gpu_buf, void* cpu_ptr,
                           size_t size);

    /// Dispatch SwiGLU on GPU-resident buffers. Eliminates CPU SiLU+multiply.
    bool gpu_swiglu(MetalBackend::buffer_id buf_gate,
                    MetalBackend::buffer_id buf_up, uint32_t n);

    /// Dispatch RMSNorm on GPU-resident buffers.
    bool gpu_rmsnorm(MetalBackend::buffer_id buf_in,
                     MetalBackend::buffer_id buf_weight,
                     MetalBackend::buffer_id buf_out,
                     uint32_t dim, float eps);

    /// Dispatch residual add on GPU-resident buffers.
    bool gpu_residual_add(MetalBackend::buffer_id buf_a,
                          MetalBackend::buffer_id buf_b,
                          MetalBackend::buffer_id buf_out, uint32_t n);

    /// GPU buffer-to-buffer copy (uses blit encoder, works inside token batch).
    /// Correctly ordered with prior compute dispatches.
    bool gpu_buffer_copy(MetalBackend::buffer_id src,
                         MetalBackend::buffer_id dst, size_t bytes);

    /// Fused FFN: encode W1 GEMV + W3 GEMV + SwiGLU + W2 GEMV as 4 dispatches
    /// on ONE command buffer with memory barriers. Only 1 commit for the whole FFN.
    /// Input: activations on CPU. Output: result on CPU.
    /// Weight buffers must be pre-cached MTLBuffer IDs.
    bool gpu_ffn_fused(const float* input, int hidden_dim,
                       MetalBackend::buffer_id buf_w1,
                       MetalBackend::buffer_id buf_w3,
                       MetalBackend::buffer_id buf_w2,
                       int ffn_dim, float* output);

    /// Access the GPU buffer pool (for callers that need specific buffer IDs).
    const GPUBufferPool& gpu_pool() const { return gpu_pool_; }

    /// Returns true if the GPU buffer pool has been initialized.
    bool has_gpu_pool() const { return gpu_pool_ready_; }

    // ─── Buffer cache management ────────────────────────────────────────

    /// Clear the wrapped-pointer buffer cache. Call when model weights are
    /// unmapped or at shutdown.
    void clear_buffer_cache();

    /// Get a cached wrapped buffer ID for a raw pointer. Returns 0 if not cached.
    MetalBackend::buffer_id get_cached_buffer(const void* ptr) const;

    /// Pre-allocate the activation buffer to the maximum size needed.
    /// In resident mode this avoids reallocation on the per-token path.
    void pre_allocate_activation_buffer(size_t max_act_bytes, size_t max_out_bytes);

    /// Touch every page of every cached MTLBuffer to fault them into
    /// physical RAM. Call after preloading all weights in resident mode.
    void preload_all_buffers();

private:
    std::unique_ptr<MetalContext> ctx_;
    std::unique_ptr<MetalBackend> gpu_;
    bool gpu_ready_ = false;

    // Cached GPU buffers for reuse (avoid re-allocating per call)
    struct BufferCache {
        MetalBackend::buffer_id buf = 0;
        size_t size = 0;
    };
    BufferCache buf_a_, buf_b_, buf_c_;

    MetalBackend::buffer_id ensure_buffer(BufferCache& cache, size_t needed);

    // Token-batch state
    bool token_batch_active_ = false;    // true between begin_token/end_token
    bool token_batch_encoding_ = false;  // true when command buffer is actively encoding

    // Cache for small uploaded buffers (norm weights, etc.)
    std::unordered_map<const void*, MetalBackend::buffer_id> small_buffer_cache_;

    // Cached wrapped-pointer MTLBuffers for mmap'd weight data.
    // Avoids recreating MTLBuffers for the same persistent pointers
    // across tokens (288 wrap_pointer + free_buffer calls per token).
    std::unordered_map<const void*, MetalBackend::buffer_id> wrapped_buffer_cache_;

    // Pre-cached per-layer KV slice buffers for GPU-resident attention.
    // Avoids wrap_pointer + free_buffer per layer per token inside the batch.
    // Indexed by layer_idx. Allocated once in init_kv_cache.
    std::vector<MetalBackend::buffer_id> kv_layer_k_bufs_;
    std::vector<MetalBackend::buffer_id> kv_layer_v_bufs_;

    // GPU-resident activation buffer pool
    GPUBufferPool gpu_pool_{};
    bool gpu_pool_ready_ = false;
    void free_gpu_pool();
};

/// Global compute dispatcher (singleton-ish, set by Engine).
ComputeDispatch& global_compute();
void set_global_compute(ComputeDispatch* dispatch);

}  // namespace nexus::compute
