/// NEXUS Inference Engine — Accelerate BLAS / vDSP compute kernels.

#include "gemm.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace nexus::compute {

// ─── gemm_f32 ───────────────────────────────────────────────────────────────

void gemm_f32(const float* A, const float* B, float* C,
              int M, int N, int K,
              bool transA, bool transB)
{
    // cblas_sgemm expects column-major by default.  We use CblasRowMajor so
    // the leading dimensions follow C-style row-major layout.
    const enum CBLAS_TRANSPOSE tA = transA ? CblasTrans : CblasNoTrans;
    const enum CBLAS_TRANSPOSE tB = transB ? CblasTrans : CblasNoTrans;

    // Leading dimensions in row-major:
    //   A (M x K): lda = K  if no transpose, lda = M if transposed
    //   B (K x N): ldb = N  if no transpose, ldb = K if transposed
    //   C (M x N): ldc = N
    const int lda = transA ? M : K;
    const int ldb = transB ? K : N;
    const int ldc = N;

    cblas_sgemm(CblasRowMajor,
                tA, tB,
                M, N, K,
                1.0f,           // alpha
                A, lda,
                B, ldb,
                0.0f,           // beta
                C, ldc);
}

// ─── gemm_f16 ───────────────────────────────────────────────────────────────

void gemm_f16(const void* A, const void* B, void* C,
              int M, int N, int K)
{
    // Phase 1 fallback: promote FP16 -> FP32, compute, demote.
    // Phase 2 will route this to a Metal compute shader.

    const __fp16* srcA = static_cast<const __fp16*>(A);
    const __fp16* srcB = static_cast<const __fp16*>(B);
    __fp16*       dstC = static_cast<__fp16*>(C);

    const size_t sizeA = static_cast<size_t>(M) * K;
    const size_t sizeB = static_cast<size_t>(K) * N;
    const size_t sizeC = static_cast<size_t>(M) * N;

    // Use vDSP to convert FP16 <-> FP32 in bulk.
    std::vector<float> fA(sizeA);
    std::vector<float> fB(sizeB);
    std::vector<float> fC(sizeC);

    // FP16 -> FP32 conversion via Accelerate.
    vImage_Buffer src_buf_a = { const_cast<__fp16*>(srcA), 1,
                                static_cast<vImagePixelCount>(sizeA),
                                sizeA * sizeof(__fp16) };
    vImage_Buffer dst_buf_a = { fA.data(), 1,
                                static_cast<vImagePixelCount>(sizeA),
                                sizeA * sizeof(float) };
    vImageConvert_Planar16FtoPlanarF(&src_buf_a, &dst_buf_a, kvImageNoFlags);

    vImage_Buffer src_buf_b = { const_cast<__fp16*>(srcB), 1,
                                static_cast<vImagePixelCount>(sizeB),
                                sizeB * sizeof(__fp16) };
    vImage_Buffer dst_buf_b = { fB.data(), 1,
                                static_cast<vImagePixelCount>(sizeB),
                                sizeB * sizeof(float) };
    vImageConvert_Planar16FtoPlanarF(&src_buf_b, &dst_buf_b, kvImageNoFlags);

    // Run FP32 GEMM.
    gemm_f32(fA.data(), fB.data(), fC.data(), M, N, K, false, false);

    // FP32 -> FP16 conversion.
    vImage_Buffer src_buf_c = { fC.data(), 1,
                                static_cast<vImagePixelCount>(sizeC),
                                sizeC * sizeof(float) };
    vImage_Buffer dst_buf_c = { dstC, 1,
                                static_cast<vImagePixelCount>(sizeC),
                                sizeC * sizeof(__fp16) };
    vImageConvert_PlanarFtoPlanar16F(&src_buf_c, &dst_buf_c, kvImageNoFlags);
}

// ─── rms_norm ───────────────────────────────────────────────────────────────

void rms_norm(float* out, const float* x, const float* weight,
              int dim, float eps)
{
    // rms = sqrt( mean(x^2) + eps )
    //
    // 1. Compute sum of squares via vDSP_dotpr (dot product of x with itself).
    float sum_sq = 0.0f;
    vDSP_dotpr(x, 1, x, 1, &sum_sq, static_cast<vDSP_Length>(dim));

    // 2. mean = sum_sq / dim,  rms = sqrt(mean + eps)
    const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(dim) + eps);

    // 3. out = x * inv_rms  (element-wise scalar multiply)
    vDSP_vsmul(x, 1, &inv_rms, out, 1, static_cast<vDSP_Length>(dim));

    // 4. out = out * weight  (element-wise vector multiply)
    vDSP_vmul(out, 1, weight, 1, out, 1, static_cast<vDSP_Length>(dim));
}

// ─── silu ───────────────────────────────────────────────────────────────────

void silu(float* out, const float* x, int n)
{
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    //
    // We use vDSP to vectorize:
    //   1. negate x
    //   2. exp(-x) via vvexpf
    //   3. add 1
    //   4. reciprocal
    //   5. multiply by x

    const vDSP_Length len = static_cast<vDSP_Length>(n);

    // out = -x
    const float neg_one = -1.0f;
    vDSP_vsmul(x, 1, &neg_one, out, 1, len);

    // out = exp(-x)   — vvexpf(dst, src, &count)
    int count = n;
    vvexpf(out, out, &count);

    // out = 1 + exp(-x)
    const float one = 1.0f;
    vDSP_vsadd(out, 1, &one, out, 1, len);

    // out = x / (1 + exp(-x))
    vDSP_vdiv(out, 1, x, 1, out, 1, len);
    // Note: vDSP_vdiv computes B[i]/A[i] -> C[i], so A=denominator, B=numerator.
}

}  // namespace nexus::compute
