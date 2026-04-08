#pragma once
/// NEXUS Compute Dispatch — Unified CPU/GPU compute interface.
///
/// Automatically routes operations to Metal GPU when available,
/// falling back to Accelerate/NEON on CPU. Handles UMA buffer
/// management so callers don't need to know which backend is active.

#include "core/config.h"
#include "compute/metal/metal_context.h"
#include "compute/metal/metal_backend.h"
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
    };

    /// Allocate the GPU buffer pool once at model load time.
    /// All buffers are persistent storageModeShared MTLBuffers that stay
    /// allocated for the lifetime of the engine.
    bool init_gpu_pool(uint32_t hidden_dim, uint32_t max_q_dim,
                       uint32_t max_kv_dim, uint32_t max_ffn_dim,
                       uint32_t vocab_size);

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

    // Cached wrapped-pointer MTLBuffers for mmap'd weight data.
    // Avoids recreating MTLBuffers for the same persistent pointers
    // across tokens (288 wrap_pointer + free_buffer calls per token).
    std::unordered_map<const void*, MetalBackend::buffer_id> wrapped_buffer_cache_;

    // GPU-resident activation buffer pool
    GPUBufferPool gpu_pool_{};
    bool gpu_pool_ready_ = false;
    void free_gpu_pool();
};

/// Global compute dispatcher (singleton-ish, set by Engine).
ComputeDispatch& global_compute();
void set_global_compute(ComputeDispatch* dispatch);

}  // namespace nexus::compute
