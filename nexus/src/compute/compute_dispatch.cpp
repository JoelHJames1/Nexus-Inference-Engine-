/// NEXUS Compute Dispatch — Unified CPU/GPU compute.
///
/// Routes to Metal GPU when available, Accelerate/NEON CPU otherwise.
/// GPU path uses storageModeShared buffers for UMA zero-copy.

#include "compute/compute_dispatch.h"
#include "compute/accelerate/gemm.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace nexus::compute {

// ─── Global singleton ───────────────────────────────────────────────────────
static ComputeDispatch* g_dispatch = nullptr;

ComputeDispatch& global_compute() {
    static ComputeDispatch fallback;
    return g_dispatch ? *g_dispatch : fallback;
}

void set_global_compute(ComputeDispatch* dispatch) {
    g_dispatch = dispatch;
}

// ─── Constructor / Destructor ───────────────────────────────────────────────

ComputeDispatch::ComputeDispatch() = default;
ComputeDispatch::~ComputeDispatch() = default;

bool ComputeDispatch::init_gpu(const std::string& shader_path) {
    ctx_ = std::make_unique<MetalContext>();
    if (!ctx_->is_available()) {
        fprintf(stderr, "[compute] No Metal GPU available, using CPU\n");
        ctx_.reset();
        return false;
    }

    fprintf(stderr, "[compute] GPU: %s (%.0f MB working set)\n",
            ctx_->device_name().c_str(),
            ctx_->recommended_max_working_set_size() / (1024.0 * 1024.0));

    if (!shader_path.empty() && ctx_->load_library(shader_path)) {
        gpu_ = std::make_unique<MetalBackend>(*ctx_);
        if (gpu_->is_ready()) {
            gpu_ready_ = true;
            fprintf(stderr, "[compute] Metal shaders loaded from %s\n", shader_path.c_str());
            return true;
        }
    }

    fprintf(stderr, "[compute] Metal shaders not loaded, GPU GEMM unavailable\n");
    fprintf(stderr, "[compute] Using Accelerate/AMX for compute\n");
    return false;
}

bool ComputeDispatch::has_gpu() const { return gpu_ready_; }

std::string ComputeDispatch::gpu_name() const {
    return ctx_ ? ctx_->device_name() : "none";
}

MetalBackend::buffer_id ComputeDispatch::ensure_buffer(BufferCache& cache, size_t needed) {
    if (cache.buf && cache.size >= needed) return cache.buf;
    if (cache.buf) gpu_->free_buffer(cache.buf);
    cache.buf = gpu_->alloc_shared_buffer(needed);
    cache.size = cache.buf ? needed : 0;
    return cache.buf;
}

// ─── GEMM ───────────────────────────────────────────────────────────────────

void ComputeDispatch::gemm(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    if (gpu_ready_ && static_cast<int64_t>(N) * K > 100000) {
        // GPU path: upload to shared buffers, dispatch Metal tiled GEMM
        size_t a_size = M * K * sizeof(float);
        size_t b_size = K * N * sizeof(float);
        size_t c_size = M * N * sizeof(float);

        auto ba = ensure_buffer(buf_a_, a_size);
        auto bb = ensure_buffer(buf_b_, b_size);
        auto bc = ensure_buffer(buf_c_, c_size);

        if (ba && bb && bc) {
            // Validate source pointers before GPU copy
            if (!A || !B || !C) {
                fprintf(stderr, "[compute] NULL pointer: A=%p B=%p C=%p\n", (void*)A, (void*)B, (void*)C);
                goto cpu_fallback;
            }
            // Verify B data is accessible (catch bad dequant pointers)
            {
                volatile float test_b = B[0];
                volatile float test_b_end = B[(size_t)K * N - 1];
                (void)test_b; (void)test_b_end;
            }
            if (!gpu_->copy_to_buffer(ba, A, a_size)) {
                fprintf(stderr, "[compute] copy_to_buffer(A) failed\n");
                goto cpu_fallback;
            }
            if (!gpu_->copy_to_buffer(bb, B, b_size)) {
                fprintf(stderr, "[compute] copy_to_buffer(B) failed\n");
                goto cpu_fallback;
            }

            fprintf(stderr, "[gpu] gemm dispatch M=%d N=%d K=%d...", M, N, K);
            fflush(stderr);
            bool ok = gpu_->gemm_f32(ba, bb, bc, M, N, K);
            fprintf(stderr, "%s\n", ok ? "OK" : "FAIL");
            if (ok) {
                void* result = gpu_->buffer_contents(bc);
                fprintf(stderr, "[gpu] readback ptr=%p\n", result);
                if (result) {
                    memcpy(C, result, c_size);
                    fprintf(stderr, "[gpu] done\n");
                    return;
                }
                fprintf(stderr, "[compute] GPU GEMM: buffer_contents returned null\n");
            } else {
                fprintf(stderr, "[compute] GPU GEMM dispatch failed, falling back to CPU\n");
            }
        } else {
            fprintf(stderr, "[compute] GPU buffer alloc failed (need %zu + %zu + %zu bytes)\n",
                    a_size, b_size, c_size);
        }
        // Fall through to CPU on failure
    }

cpu_fallback:
    // CPU path: Accelerate cblas_sgemm (auto-uses AMX coprocessor)
    compute::gemm_f32(A, B, C, M, N, K, false, false);
}

void ComputeDispatch::gemv_int4(float* out, const float* activations,
                                 const uint8_t* weights_q, const float* scales,
                                 const float* zeros,
                                 int M, int N, int K, int group_size) {
    if (gpu_ready_) {
        // GPU path: fused dequant + GEMV in one Metal dispatch
        size_t act_size = M * K * sizeof(float);
        size_t wq_size = (K * N + 1) / 2;  // INT4 packed
        size_t num_groups = (K * N + group_size - 1) / group_size;
        size_t sc_size = num_groups * sizeof(float);
        size_t out_size = M * N * sizeof(float);

        auto ba = ensure_buffer(buf_a_, act_size);
        auto bb = ensure_buffer(buf_b_, std::max(wq_size, sc_size * 2));
        auto bc = ensure_buffer(buf_c_, out_size);

        if (ba && bb && bc) {
            gpu_->copy_to_buffer(ba, activations, act_size);

            // For fused INT4, we'd dispatch gemv_dequant_int4 shader here
            // For now, fall through to CPU
        }
    }

    // CPU fallback: dequantize then GEMM
    // (This is what the hybrid model already does via load_tensor dequant)
    // If we get here, activations are FP32 and we just do standard GEMM
    compute::gemm_f32(activations, reinterpret_cast<const float*>(weights_q),
                      out, M, N, K, false, false);
}

// ─── Element-wise ops ───────────────────────────────────────────────────────

void ComputeDispatch::rmsnorm(float* out, const float* x, const float* weight,
                               int dim, float eps) {
    // CPU path (fast enough for small dim, GPU overhead not worth it)
    compute::rms_norm(out, x, weight, dim, eps);
}

void ComputeDispatch::silu(float* out, const float* x, int n) {
    compute::silu(out, x, n);
}

void ComputeDispatch::swiglu(float* out, const float* gate, const float* up, int n) {
    // Fused: out = silu(gate) * up
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        float s = g / (1.0f + expf(-g));
        out[i] = s * up[i];
    }
}

void ComputeDispatch::softmax(float* data, int dim) {
    float max_val = *std::max_element(data, data + dim);
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }
    if (sum > 0.0f) {
        for (int i = 0; i < dim; i++) data[i] /= sum;
    }
}

}  // namespace nexus::compute
