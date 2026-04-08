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

    // ─── Buffer cache management ────────────────────────────────────────

    /// Clear the wrapped-pointer buffer cache. Call when model weights are
    /// unmapped or at shutdown.
    void clear_buffer_cache();

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
};

/// Global compute dispatcher (singleton-ish, set by Engine).
ComputeDispatch& global_compute();
void set_global_compute(ComputeDispatch* dispatch);

}  // namespace nexus::compute
