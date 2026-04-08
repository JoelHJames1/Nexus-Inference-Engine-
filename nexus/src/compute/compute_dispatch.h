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

    // ─── Element-wise operations ────────────────────────────────────────

    /// RMSNorm: out = x * weight / sqrt(mean(x^2) + eps)
    void rmsnorm(float* out, const float* x, const float* weight, int dim, float eps);

    /// SiLU activation: out = x * sigmoid(x)
    void silu(float* out, const float* x, int n);

    /// Fused SwiGLU: out = silu(gate) * up
    void swiglu(float* out, const float* gate, const float* up, int n);

    /// Softmax in-place over dim elements
    void softmax(float* data, int dim);

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
};

/// Global compute dispatcher (singleton-ish, set by Engine).
ComputeDispatch& global_compute();
void set_global_compute(ComputeDispatch* dispatch);

}  // namespace nexus::compute
