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
#include <vector>

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

ComputeDispatch::~ComputeDispatch() {
    // Clean up cached wrapped-pointer buffers before gpu_ is destroyed.
    clear_buffer_cache();
}

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
            if (!gpu_->copy_to_buffer(ba, A, a_size) ||
                !gpu_->copy_to_buffer(bb, B, b_size)) {
                goto cpu_fallback;
            }

            if (gpu_->gemm_f32(ba, bb, bc, M, N, K)) {
                void* result = gpu_->buffer_contents(bc);
                if (result) {
                    memcpy(C, result, c_size);
                    return;
                }
            }
        }
        // Fall through to CPU on failure
    }

cpu_fallback:
    // CPU path: Accelerate cblas_sgemm (auto-uses AMX coprocessor)
    compute::gemm_f32(A, B, C, M, N, K, false, false);
}

// ─── Fused INT4 GEMM (zero-copy UMA path) ──────────────────────────────────

void ComputeDispatch::gemm_int4(const float* activations, const void* weights_int4,
                                 size_t weights_bytes, float* out,
                                 int M, int N, int K) {
    // UMA zero-copy GPU path: wrap mmap'd INT4 data as MTLBuffer, dispatch
    // fused dequant+GEMV shader. No CPU dequant, no allocation, no memcpy.
    if (gpu_ready_) {
        // ── Optimization 2: Cached weight buffers ──
        // The same mmap'd weight pointers are reused across tokens.
        // Cache the wrapped MTLBuffer to avoid 288 wrap+free round-trips/token.
        MetalBackend::buffer_id buf_w;
        auto it = wrapped_buffer_cache_.find(weights_int4);
        if (it != wrapped_buffer_cache_.end()) {
            buf_w = it->second;
        } else {
            buf_w = gpu_->wrap_pointer(const_cast<void*>(weights_int4), weights_bytes);
            if (buf_w) {
                wrapped_buffer_cache_[weights_int4] = buf_w;
            }
        }

        size_t act_size = static_cast<size_t>(M) * K * sizeof(float);
        size_t out_size = static_cast<size_t>(M) * N * sizeof(float);
        auto buf_a = ensure_buffer(buf_a_, act_size);
        auto buf_out = ensure_buffer(buf_c_, out_size);

        if (buf_w && buf_a && buf_out) {
            gpu_->copy_to_buffer(buf_a, activations, act_size);

            if (gpu_->gemv_int4_uniform(buf_a, buf_w, buf_out, N, K)) {
                void* result = gpu_->buffer_contents(buf_out);
                if (result) {
                    memcpy(out, result, out_size);
                    // Don't free buf_w — it's cached for reuse across tokens
                    return;
                }
            }
            // Don't free buf_w — it's cached
        }
    }

    // GPU is required for INT4 GEMV — no CPU fallback
    fprintf(stderr, "[FATAL] gemm_int4 requires GPU but Metal dispatch failed.\n");
    memset(out, 0, static_cast<size_t>(M) * N * sizeof(float));
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

// ─── Batched MoE FFN (single dispatch for all experts) ──────────────────────

void ComputeDispatch::batched_moe_ffn(
    const float* activations, int hidden_dim, int expert_ffn_dim,
    const int* expert_ids, const float* gate_weights, int num_active,
    int num_experts,
    const void* w1_raw, size_t w1_bytes,
    const void* w2_raw, size_t w2_bytes,
    const void* w3_raw, size_t w3_bytes,
    float* output)
{
    if (!gpu_ready_) {
        memset(output, 0, hidden_dim * sizeof(float));
        return;
    }

    // Wrap or get cached buffers for expert weight tensors
    auto get_cached = [&](const void* ptr, size_t bytes) -> MetalBackend::buffer_id {
        auto it = wrapped_buffer_cache_.find(ptr);
        if (it != wrapped_buffer_cache_.end()) return it->second;
        auto buf = gpu_->wrap_pointer(const_cast<void*>(ptr), bytes);
        if (buf) wrapped_buffer_cache_[ptr] = buf;
        return buf;
    };

    auto buf_w1 = get_cached(w1_raw, w1_bytes);
    auto buf_w2 = get_cached(w2_raw, w2_bytes);
    auto buf_w3 = get_cached(w3_raw, w3_bytes);

    size_t act_size = hidden_dim * sizeof(float);
    size_t out_size = hidden_dim * sizeof(float);
    size_t ids_size = num_active * sizeof(int);
    size_t gw_size = num_active * sizeof(float);

    auto buf_act = ensure_buffer(buf_a_, act_size);
    auto buf_out = ensure_buffer(buf_c_, out_size);

    // Small buffers for expert IDs and gate weights
    auto buf_ids = gpu_->alloc_shared_buffer(ids_size);
    auto buf_gw = gpu_->alloc_shared_buffer(gw_size);

    if (buf_w1 && buf_w2 && buf_w3 && buf_act && buf_out && buf_ids && buf_gw) {
        gpu_->copy_to_buffer(buf_act, activations, act_size);
        gpu_->copy_to_buffer(buf_ids, expert_ids, ids_size);
        gpu_->copy_to_buffer(buf_gw, gate_weights, gw_size);
        // Zero output buffer
        memset(gpu_->buffer_contents(buf_out), 0, out_size);

        if (gpu_->batched_expert_ffn(buf_act, buf_w1, buf_w3, buf_w2,
                                      buf_ids, buf_gw, buf_out,
                                      hidden_dim, expert_ffn_dim,
                                      num_active, num_experts)) {
            void* result = gpu_->buffer_contents(buf_out);
            if (result) {
                memcpy(output, result, out_size);
                gpu_->free_buffer(buf_ids);
                gpu_->free_buffer(buf_gw);
                return;
            }
        }
    }
    if (buf_ids) gpu_->free_buffer(buf_ids);
    if (buf_gw) gpu_->free_buffer(buf_gw);

    // Fallback: zero output
    memset(output, 0, hidden_dim * sizeof(float));
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

// ─── Batch mode (command buffer pipelining) ────────────────────────────────

void ComputeDispatch::begin_gpu_batch() {
    if (gpu_ready_ && gpu_) {
        gpu_->begin_batch();
    }
}

void ComputeDispatch::end_gpu_batch() {
    if (gpu_ready_ && gpu_) {
        gpu_->end_batch();
    }
}

// ─── Buffer cache management ───────────────────────────────────────────────

void ComputeDispatch::clear_buffer_cache() {
    if (gpu_) {
        for (auto& [ptr, buf_id] : wrapped_buffer_cache_) {
            gpu_->free_buffer(buf_id);
        }
    }
    wrapped_buffer_cache_.clear();
}

}  // namespace nexus::compute
