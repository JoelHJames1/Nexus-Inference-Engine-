#pragma once
/// NEXUS Inference Engine — Accelerate BLAS / vDSP compute kernels.
///
/// Wraps Apple's Accelerate framework for GEMM, normalization, and
/// activation functions.  FP16 GEMM converts through FP32 for now;
/// Phase 2 will route FP16 work to Metal compute shaders.

#include <cstddef>
#include <cstdint>

namespace nexus::compute {

// ─── General matrix multiply ────────────────────────────────────────────────

/// C = alpha * op(A) * op(B) + beta * C   (alpha=1, beta=0)
/// A : M x K  (or K x M if transA)
/// B : K x N  (or N x K if transB)
/// C : M x N
void gemm_f32(const float* A, const float* B, float* C,
              int M, int N, int K,
              bool transA = false, bool transB = false);

/// FP16 GEMM — accepts __fp16 buffers packed as void*.
/// Phase 1: promotes to FP32, calls cblas_sgemm, demotes result.
/// Phase 2: dispatches to Metal compute shader.
void gemm_f16(const void* A, const void* B, void* C,
              int M, int N, int K);

// ─── Normalization ──────────────────────────────────────────────────────────

/// RMSNorm:  out[i] = x[i] / rms(x) * weight[i]
///           rms(x) = sqrt( (1/dim) * sum(x^2) + eps )
void rms_norm(float* out, const float* x, const float* weight,
              int dim, float eps = 1e-5f);

// ─── Activations ────────────────────────────────────────────────────────────

/// SiLU (Swish):  out[i] = x[i] * sigmoid(x[i])
void silu(float* out, const float* x, int n);

}  // namespace nexus::compute
