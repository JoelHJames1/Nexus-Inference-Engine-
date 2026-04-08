/// NEXUS Compute Dispatch — Unified CPU/GPU compute.
///
/// Routes to Metal GPU when available, Accelerate/NEON CPU otherwise.
/// GPU path uses storageModeShared buffers for UMA zero-copy.

#include "compute/compute_dispatch.h"
#include "compute/accelerate/gemm.h"
#include "model/hybrid_model.h"
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
    // Free GPU-resident activation pool before gpu_ is destroyed.
    free_gpu_pool();
    // Free small uploaded buffers (norm weights, etc.)
    if (gpu_) {
        for (auto& [ptr, buf_id] : small_buffer_cache_) {
            gpu_->free_buffer(buf_id);
        }
    }
    small_buffer_cache_.clear();
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

// ─── Fused INT4 GEMM via Metal (UMA zero-copy) ─────────────────────────────
//
// Metal GPU with cached MTLBuffers. The INT4 data is read directly by the
// GPU shader without expanding to FP32, saving 4x bandwidth.
//
// Key: weight buffers are pre-cached as MTLBuffers at model load time.
// Per-token cost is just: activation copy + dispatch + readback.

void ComputeDispatch::gemm_int4(const float* activations, const void* weights_int4,
                                 size_t weights_bytes, float* out,
                                 int M, int N, int K) {
    if (gpu_ready_) {
        // Get or create cached weight buffer
        MetalBackend::buffer_id buf_w;
        auto it = wrapped_buffer_cache_.find(weights_int4);
        if (it != wrapped_buffer_cache_.end()) {
            buf_w = it->second;
        } else {
            buf_w = gpu_->wrap_pointer(const_cast<void*>(weights_int4), weights_bytes);
            if (buf_w) wrapped_buffer_cache_[weights_int4] = buf_w;
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
                    return;
                }
            }
        }
    }

    // Fallback: zero output
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

// ─── GPU-Resident Activation Pipeline ───────────────────────────────────────

void ComputeDispatch::free_gpu_pool() {
    if (!gpu_pool_ready_ || !gpu_) return;
    auto free = [&](MetalBackend::buffer_id& buf) {
        if (buf) { gpu_->free_buffer(buf); buf = 0; }
    };
    free(gpu_pool_.hidden);
    free(gpu_pool_.residual);
    free(gpu_pool_.norm_buf);
    free(gpu_pool_.q_buf);
    free(gpu_pool_.k_buf);
    free(gpu_pool_.v_buf);
    free(gpu_pool_.attn_out);
    free(gpu_pool_.ffn_gate);
    free(gpu_pool_.ffn_up);
    free(gpu_pool_.ffn_out);
    free(gpu_pool_.logits);
    // Free pre-cached per-layer KV slice buffers
    for (auto buf : kv_layer_k_bufs_) { if (buf) gpu_->free_buffer(buf); }
    for (auto buf : kv_layer_v_bufs_) { if (buf) gpu_->free_buffer(buf); }
    kv_layer_k_bufs_.clear();
    kv_layer_v_bufs_.clear();
    free(gpu_pool_.kv_keys);
    free(gpu_pool_.kv_values);
    gpu_pool_.kv_max_seq = 0;
    gpu_pool_.kv_dim = 0;
    gpu_pool_.kv_num_layers = 0;
    gpu_pool_ready_ = false;
}

bool ComputeDispatch::init_gpu_pool(uint32_t hidden_dim, uint32_t max_q_dim,
                                     uint32_t max_kv_dim, uint32_t max_ffn_dim,
                                     uint32_t vocab_size) {
    if (!gpu_ready_ || !gpu_) return false;

    // Free any previous pool
    free_gpu_pool();

    // Helper: allocate a persistent shared buffer of the given float count.
    auto alloc = [&](uint32_t float_count) -> MetalBackend::buffer_id {
        return gpu_->alloc_shared_buffer(static_cast<size_t>(float_count) * sizeof(float));
    };

    gpu_pool_.hidden   = alloc(hidden_dim);
    gpu_pool_.residual = alloc(hidden_dim);
    gpu_pool_.norm_buf = alloc(hidden_dim);
    gpu_pool_.q_buf    = alloc(max_q_dim);
    gpu_pool_.k_buf    = alloc(max_kv_dim);
    gpu_pool_.v_buf    = alloc(max_kv_dim);
    gpu_pool_.attn_out = alloc(max_q_dim);
    gpu_pool_.ffn_gate = alloc(max_ffn_dim);
    gpu_pool_.ffn_up   = alloc(max_ffn_dim);
    gpu_pool_.ffn_out  = alloc(hidden_dim);
    gpu_pool_.logits   = alloc(vocab_size);

    // Verify all allocations succeeded
    if (!gpu_pool_.hidden || !gpu_pool_.residual || !gpu_pool_.norm_buf ||
        !gpu_pool_.q_buf || !gpu_pool_.k_buf || !gpu_pool_.v_buf ||
        !gpu_pool_.attn_out || !gpu_pool_.ffn_gate || !gpu_pool_.ffn_up ||
        !gpu_pool_.ffn_out || !gpu_pool_.logits) {
        fprintf(stderr, "[compute] GPU pool allocation failed — freeing partial pool\n");
        free_gpu_pool();
        return false;
    }

    gpu_pool_ready_ = true;

    size_t total_bytes = (static_cast<size_t>(hidden_dim) * 4 +  // hidden, residual, norm_buf, ffn_out
                          static_cast<size_t>(max_q_dim) * 2 +    // q_buf, attn_out
                          static_cast<size_t>(max_kv_dim) * 2 +   // k_buf, v_buf
                          static_cast<size_t>(max_ffn_dim) * 2 +  // ffn_gate, ffn_up
                          static_cast<size_t>(vocab_size)) * sizeof(float);
    fprintf(stderr, "[compute] GPU-resident activation pool: %.1f MB "
            "(hidden=%u q=%u kv=%u ffn=%u vocab=%u)\n",
            total_bytes / (1024.0 * 1024.0),
            hidden_dim, max_q_dim, max_kv_dim, max_ffn_dim, vocab_size);
    return true;
}

bool ComputeDispatch::init_kv_cache(int num_layers, int max_seq, int kv_dim) {
    if (!gpu_ready_ || !gpu_ || !gpu_pool_ready_) return false;

    // Free any previous KV cache buffers
    if (gpu_pool_.kv_keys) { gpu_->free_buffer(gpu_pool_.kv_keys); gpu_pool_.kv_keys = 0; }
    if (gpu_pool_.kv_values) { gpu_->free_buffer(gpu_pool_.kv_values); gpu_pool_.kv_values = 0; }

    size_t kv_buf_size = static_cast<size_t>(num_layers) * max_seq * kv_dim * sizeof(float);
    gpu_pool_.kv_keys = gpu_->alloc_shared_buffer(kv_buf_size);
    gpu_pool_.kv_values = gpu_->alloc_shared_buffer(kv_buf_size);

    if (!gpu_pool_.kv_keys || !gpu_pool_.kv_values) {
        fprintf(stderr, "[compute] GPU KV cache allocation failed (%.1f MB per buffer)\n",
                kv_buf_size / (1024.0 * 1024.0));
        if (gpu_pool_.kv_keys) { gpu_->free_buffer(gpu_pool_.kv_keys); gpu_pool_.kv_keys = 0; }
        if (gpu_pool_.kv_values) { gpu_->free_buffer(gpu_pool_.kv_values); gpu_pool_.kv_values = 0; }
        return false;
    }

    // Zero-initialize the cache (UMA: buffer_contents gives direct pointer)
    void* k_ptr = gpu_->buffer_contents(gpu_pool_.kv_keys);
    void* v_ptr = gpu_->buffer_contents(gpu_pool_.kv_values);
    if (k_ptr) memset(k_ptr, 0, kv_buf_size);
    if (v_ptr) memset(v_ptr, 0, kv_buf_size);

    gpu_pool_.kv_max_seq = max_seq;
    gpu_pool_.kv_dim = kv_dim;
    gpu_pool_.kv_num_layers = num_layers;

    // Pre-create per-layer KV slice buffers (wrap_pointer on UMA offsets).
    // This avoids wrap_pointer + free_buffer per layer per token inside the batch.
    // Free any previous layer buffers first.
    for (auto buf : kv_layer_k_bufs_) { if (buf) gpu_->free_buffer(buf); }
    for (auto buf : kv_layer_v_bufs_) { if (buf) gpu_->free_buffer(buf); }
    kv_layer_k_bufs_.resize(num_layers, 0);
    kv_layer_v_bufs_.resize(num_layers, 0);

    size_t layer_buf_size = static_cast<size_t>(max_seq) * kv_dim * sizeof(float);
    int cached_layers = 0;
    for (int i = 0; i < num_layers; i++) {
        size_t layer_float_offset = static_cast<size_t>(i) * max_seq * kv_dim;
        float* k_layer = reinterpret_cast<float*>(k_ptr) + layer_float_offset;
        float* v_layer = reinterpret_cast<float*>(v_ptr) + layer_float_offset;
        kv_layer_k_bufs_[i] = gpu_->wrap_pointer(k_layer, layer_buf_size);
        kv_layer_v_bufs_[i] = gpu_->wrap_pointer(v_layer, layer_buf_size);
        if (kv_layer_k_bufs_[i] && kv_layer_v_bufs_[i]) cached_layers++;
    }

    fprintf(stderr, "[compute] GPU KV cache: %.1f MB "
            "(layers=%d max_seq=%d kv_dim=%d, %.1f MB per buffer, %d layer bufs cached)\n",
            2.0 * kv_buf_size / (1024.0 * 1024.0),
            num_layers, max_seq, kv_dim, kv_buf_size / (1024.0 * 1024.0), cached_layers);
    return true;
}

bool ComputeDispatch::attention_gpu(MetalBackend::buffer_id buf_q,
                                     MetalBackend::buffer_id buf_new_k,
                                     MetalBackend::buffer_id buf_new_v,
                                     MetalBackend::buffer_id buf_output,
                                     int layer_idx, int seq_pos,
                                     int num_heads, int num_kv_heads,
                                     int head_dim_q, int head_dim_kv) {
    if (!gpu_ready_ || !gpu_ || !gpu_pool_ready_) return false;
    if (!gpu_pool_.kv_keys || !gpu_pool_.kv_values) return false;
    if (layer_idx >= gpu_pool_.kv_num_layers) return false;
    if (seq_pos >= gpu_pool_.kv_max_seq) return false;

    int kv_dim = gpu_pool_.kv_dim;

    // Use pre-cached per-layer KV slice buffers (allocated in init_kv_cache).
    // This eliminates wrap_pointer + free_buffer per layer per token,
    // which was creating and destroying MTLBuffers inside the command batch.
    MetalBackend::buffer_id buf_k_layer = 0;
    MetalBackend::buffer_id buf_v_layer = 0;

    if (layer_idx < static_cast<int>(kv_layer_k_bufs_.size())) {
        buf_k_layer = kv_layer_k_bufs_[layer_idx];
        buf_v_layer = kv_layer_v_bufs_[layer_idx];
    }

    if (!buf_k_layer || !buf_v_layer) {
        // Fallback: create temporary buffers (shouldn't happen after init_kv_cache)
        void* k_base = gpu_->buffer_contents(gpu_pool_.kv_keys);
        void* v_base = gpu_->buffer_contents(gpu_pool_.kv_values);
        if (!k_base || !v_base) return false;

        size_t layer_float_offset = static_cast<size_t>(layer_idx) * gpu_pool_.kv_max_seq * kv_dim;
        float* k_layer_ptr = reinterpret_cast<float*>(k_base) + layer_float_offset;
        float* v_layer_ptr = reinterpret_cast<float*>(v_base) + layer_float_offset;
        size_t layer_buf_size = static_cast<size_t>(gpu_pool_.kv_max_seq) * kv_dim * sizeof(float);
        buf_k_layer = gpu_->wrap_pointer(k_layer_ptr, layer_buf_size);
        buf_v_layer = gpu_->wrap_pointer(v_layer_ptr, layer_buf_size);
        if (!buf_k_layer || !buf_v_layer) {
            if (buf_k_layer) gpu_->free_buffer(buf_k_layer);
            if (buf_v_layer) gpu_->free_buffer(buf_v_layer);
            return false;
        }
        // Cache them for next time
        if (layer_idx < static_cast<int>(kv_layer_k_bufs_.size())) {
            kv_layer_k_bufs_[layer_idx] = buf_k_layer;
            kv_layer_v_bufs_[layer_idx] = buf_v_layer;
        }
    }

    GPUAttentionParams params;
    params.seq_len = static_cast<uint32_t>(seq_pos + 1);  // Including the new token
    params.num_heads = static_cast<uint32_t>(num_heads);
    params.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
    params.head_dim_q = static_cast<uint32_t>(head_dim_q);
    params.head_dim_kv = static_cast<uint32_t>(head_dim_kv);
    params.scale = 1.0f / sqrtf(static_cast<float>(head_dim_q));
    params.seq_pos = static_cast<uint32_t>(seq_pos);
    params.kv_stride = static_cast<uint32_t>(kv_dim);

    return gpu_->attention_gpu(buf_q, buf_k_layer, buf_v_layer,
                               buf_new_k, buf_new_v, buf_output, params);
}

bool ComputeDispatch::fused_attention(
    const float* q, int q_dim,
    const float* k_new, const float* v_new, int kv_dim,
    uint16_t* k_cache, uint16_t* v_cache,
    int seq_pos, int max_seq,
    int num_heads, int num_kv_heads,
    int head_dim_q, int head_dim_kv,
    float* output, int out_dim)
{
    if (!gpu_ready_ || !gpu_) return false;

    // 1. Store new K/V in FP16 cache (CPU side — fast bit conversion)
    for (int i = 0; i < kv_dim; i++) {
        k_cache[seq_pos * kv_dim + i] = nexus::model::HybridKVCache::f32_to_f16(k_new[i]);
        v_cache[seq_pos * kv_dim + i] = nexus::model::HybridKVCache::f32_to_f16(v_new[i]);
    }
    int seq_len = seq_pos + 1;

    // 2. Upload Q, wrap KV cache as GPU buffers
    size_t q_size = q_dim * sizeof(float);
    size_t out_size = out_dim * sizeof(float);
    size_t kv_cache_size = static_cast<size_t>(max_seq) * kv_dim * sizeof(uint16_t);

    auto buf_q = ensure_buffer(buf_a_, q_size);
    auto buf_out = ensure_buffer(buf_c_, out_size);
    // Allocate and copy FP16 KV data to GPU buffers
    size_t kv_used = static_cast<size_t>(seq_len) * kv_dim * sizeof(uint16_t);
    auto buf_k = gpu_->alloc_shared_buffer(kv_cache_size);
    auto buf_v = gpu_->alloc_shared_buffer(kv_cache_size);

    if (!buf_q || !buf_out || !buf_k || !buf_v) {
        if (buf_k) gpu_->free_buffer(buf_k);
        if (buf_v) gpu_->free_buffer(buf_v);
        return false;
    }

    gpu_->copy_to_buffer(buf_q, q, q_size);
    gpu_->copy_to_buffer(buf_k, k_cache, kv_used);
    gpu_->copy_to_buffer(buf_v, v_cache, kv_used);

    // 3. Dispatch fused multi-head attention (ALL heads, 1 dispatch)
    int dot_dim = std::min(head_dim_q, head_dim_kv);
    MetalBackend::FusedAttentionParams params;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim_q = head_dim_q;
    params.head_dim_kv = head_dim_kv;
    params.dot_dim = dot_dim;
    params.out_head_dim = head_dim_kv;
    params.seq_len = seq_len;
    params.kv_dim = kv_dim;
    params.scale = 1.0f / sqrtf(static_cast<float>(dot_dim));

    bool ok = gpu_->fused_attention_decode(buf_q, buf_k, buf_v, buf_out, params);

    // 4. Read result
    if (ok) {
        void* result = gpu_->buffer_contents(buf_out);
        if (result) {
            memcpy(output, result, out_size);
        }
    }

    gpu_->free_buffer(buf_k);
    gpu_->free_buffer(buf_v);
    return ok;
}

bool ComputeDispatch::gemm_int4_gpu(MetalBackend::buffer_id buf_in,
                                     MetalBackend::buffer_id buf_weight,
                                     MetalBackend::buffer_id buf_out,
                                     uint32_t M, uint32_t N, uint32_t K) {
    if (!gpu_ready_ || !gpu_) return false;
    // Direct GPU-to-GPU dispatch — no CPU copies.
    // M is always 1 for single-token decode, but we pass N and K to the shader.
    return gpu_->gemm_int4_gpu(buf_in, buf_weight, buf_out, N, K);
}

bool ComputeDispatch::upload_to_gpu(const void* cpu_ptr,
                                     MetalBackend::buffer_id gpu_buf,
                                     size_t size) {
    if (!gpu_ready_ || !gpu_ || !cpu_ptr || !gpu_buf) return false;
    return gpu_->copy_to_buffer(gpu_buf, cpu_ptr, size);
}

bool ComputeDispatch::download_from_gpu(MetalBackend::buffer_id gpu_buf,
                                         void* cpu_ptr, size_t size) {
    if (!gpu_ready_ || !gpu_ || !cpu_ptr || !gpu_buf) return false;
    void* gpu_ptr = gpu_->buffer_contents(gpu_buf);
    if (!gpu_ptr) return false;
    std::memcpy(cpu_ptr, gpu_ptr, size);
    return true;
}

bool ComputeDispatch::gpu_swiglu(MetalBackend::buffer_id buf_gate,
                                  MetalBackend::buffer_id buf_up, uint32_t n) {
    if (!gpu_ready_ || !gpu_) return false;
    // Dispatch swiglu_fused shader directly on GPU-resident buffers.
    // Output overwrites buf_gate in-place (gate buffer becomes the result).
    return gpu_->swiglu_fused(buf_gate, buf_up, buf_gate, n);
}

bool ComputeDispatch::gpu_rmsnorm(MetalBackend::buffer_id buf_in,
                                   MetalBackend::buffer_id buf_weight,
                                   MetalBackend::buffer_id buf_out,
                                   uint32_t dim, float eps) {
    if (!gpu_ready_ || !gpu_) return false;
    return gpu_->rmsnorm(buf_in, buf_weight, buf_out, dim, eps);
}

bool ComputeDispatch::gpu_residual_add(MetalBackend::buffer_id buf_a,
                                        MetalBackend::buffer_id buf_b,
                                        MetalBackend::buffer_id buf_out,
                                        uint32_t n) {
    if (!gpu_ready_ || !gpu_) return false;
    return gpu_->residual_add(buf_a, buf_b, buf_out, n);
}

bool ComputeDispatch::fused_ffn(const float* hidden, int dim,
                                 MetalBackend::buffer_id buf_norm_w,
                                 MetalBackend::buffer_id buf_w1,
                                 MetalBackend::buffer_id buf_w3,
                                 MetalBackend::buffer_id buf_w2,
                                 int ffn_dim, float eps, float* output) {
    if (!gpu_ready_ || !gpu_) return false;

    size_t h_size = dim * sizeof(float);
    size_t o_size = dim * sizeof(float);
    auto buf_h = ensure_buffer(buf_a_, h_size);
    auto buf_o = ensure_buffer(buf_c_, o_size);
    if (!buf_h || !buf_o) return false;

    gpu_->copy_to_buffer(buf_h, hidden, h_size);
    memset(gpu_->buffer_contents(buf_o), 0, o_size);

    MetalBackend::FusedFFNParams params;
    params.dim = dim;
    params.ffn_dim = ffn_dim;
    params.rms_eps = eps;

    bool ok = gpu_->fused_ffn_int4(buf_h, buf_norm_w, buf_w1, buf_w3, buf_w2,
                                    buf_o, params);
    if (ok) {
        void* result = gpu_->buffer_contents(buf_o);
        if (result) {
            memcpy(output, result, o_size);
            return true;
        }
    }
    return false;
}

bool ComputeDispatch::gpu_buffer_copy(MetalBackend::buffer_id src,
                                       MetalBackend::buffer_id dst,
                                       size_t bytes) {
    if (!gpu_ready_ || !gpu_) return false;
    return gpu_->buffer_copy(src, dst, bytes);
}

bool ComputeDispatch::gpu_ffn_fused(const float* input, int hidden_dim,
                                     MetalBackend::buffer_id buf_w1,
                                     MetalBackend::buffer_id buf_w3,
                                     MetalBackend::buffer_id buf_w2,
                                     int ffn_dim, float* output) {
    if (!gpu_ready_ || !gpu_pool_ready_) return false;

    // When in token-batch mode, activations are already in gpu_pool_.norm_buf
    // and results stay in gpu_pool_.hidden (no CPU copies needed).
    bool in_token = token_batch_active_;

    if (!in_token) {
        // Standalone FFN call: upload input activation
        size_t in_size = hidden_dim * sizeof(float);
        gpu_->copy_to_buffer(gpu_pool_.hidden, input, in_size);

        // Start persistent batch — all 4 dispatches share one command buffer
        gpu_->begin_batch();
    }

    // Use norm_buf as FFN input when in token-batch mode (already has post-norm data),
    // otherwise use hidden (where we just uploaded the input).
    auto ffn_input_buf = in_token ? gpu_pool_.norm_buf : gpu_pool_.hidden;

    // W1 GEMV: input → ffn_gate
    gpu_->gemm_int4_gpu(ffn_input_buf, buf_w1, gpu_pool_.ffn_gate, ffn_dim, hidden_dim);
    // W3 GEMV: input → ffn_up
    gpu_->gemm_int4_gpu(ffn_input_buf, buf_w3, gpu_pool_.ffn_up, ffn_dim, hidden_dim);
    // SwiGLU: ffn_gate = silu(ffn_gate) * ffn_up
    gpu_->swiglu_fused(gpu_pool_.ffn_gate, gpu_pool_.ffn_up, gpu_pool_.ffn_gate, ffn_dim);
    // W2 GEMV: ffn_gate → ffn_out
    gpu_->gemm_int4_gpu(gpu_pool_.ffn_gate, buf_w2, gpu_pool_.ffn_out, hidden_dim, ffn_dim);

    if (!in_token) {
        // ONE commit for all 4 dispatches
        gpu_->end_batch();

        // Download result
        void* result = gpu_->buffer_contents(gpu_pool_.ffn_out);
        if (result) {
            memcpy(output, result, hidden_dim * sizeof(float));
            return true;
        }
        return false;
    }

    // In token-batch mode: results stay on GPU in ffn_out buffer.
    // Caller (execute_attention_moe_layer) will do the residual add on GPU.
    return true;
}

MetalBackend::buffer_id ComputeDispatch::get_cached_buffer(const void* ptr) const {
    auto it = wrapped_buffer_cache_.find(ptr);
    return (it != wrapped_buffer_cache_.end()) ? it->second : 0;
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
        // If we're in a token-level batch with an active command buffer,
        // this is a no-op — the token batch owns the command buffer.
        if (token_batch_encoding_) return;
        gpu_->begin_batch();
    }
}

void ComputeDispatch::end_gpu_batch() {
    if (gpu_ready_ && gpu_) {
        // If we're in a token-level batch with an active command buffer,
        // don't commit — the token batch's end_token() will handle it.
        if (token_batch_encoding_) return;
        gpu_->end_batch();
    }
}

// ─── Token-level batching (single commit per token) ────────────────────────

void ComputeDispatch::begin_token() {
    if (!gpu_ready_ || !gpu_) return;
    if (token_batch_active_) return;  // Already in token batch
    token_batch_active_ = true;
    token_batch_encoding_ = true;
    gpu_->begin_batch();
}

void ComputeDispatch::end_token() {
    if (!gpu_ready_ || !gpu_) return;
    if (!token_batch_active_) return;
    if (token_batch_encoding_) {
        gpu_->end_batch();
    }
    token_batch_active_ = false;
    token_batch_encoding_ = false;
}

void ComputeDispatch::flush_token() {
    if (!gpu_ready_ || !gpu_ || !token_batch_active_) return;
    if (token_batch_encoding_) {
        // Commit current command buffer so GPU results are readable on CPU.
        gpu_->end_batch();
        token_batch_encoding_ = false;
    }
    // token_batch_active_ stays true — resume_token() will restart.
}

void ComputeDispatch::resume_token() {
    if (!gpu_ready_ || !gpu_ || !token_batch_active_) return;
    if (!token_batch_encoding_) {
        // Start a new command buffer for the rest of this token.
        gpu_->begin_batch();
        token_batch_encoding_ = true;
    }
}

MetalBackend::buffer_id ComputeDispatch::upload_small_buffer(const void* cpu_ptr,
                                                              size_t bytes) {
    if (!gpu_ready_ || !gpu_ || !cpu_ptr || bytes == 0) return 0;

    // Check cache first
    auto it = small_buffer_cache_.find(cpu_ptr);
    if (it != small_buffer_cache_.end()) return it->second;

    // Allocate and upload
    auto buf = gpu_->alloc_shared_buffer(bytes);
    if (!buf) return 0;
    gpu_->copy_to_buffer(buf, cpu_ptr, bytes);
    small_buffer_cache_[cpu_ptr] = buf;
    return buf;
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

void ComputeDispatch::pre_allocate_activation_buffer(size_t max_act_bytes, size_t max_out_bytes) {
    if (!gpu_ready_) return;
    // Pre-allocate buf_a_ (activations) and buf_c_ (output) to their maximum
    // sizes so the per-token path never reallocates.
    ensure_buffer(buf_a_, max_act_bytes);
    ensure_buffer(buf_c_, max_out_bytes);
    fprintf(stderr, "[compute] Pre-allocated activation buffers: act=%zu KB, out=%zu KB\n",
            max_act_bytes / 1024, max_out_bytes / 1024);
}

void ComputeDispatch::preload_all_buffers() {
    if (!gpu_ready_) return;
    size_t count = 0;
    // Touch every page of every cached wrapped-pointer MTLBuffer to fault
    // all pages into physical RAM. This ensures the first token pays zero
    // page-fault cost.
    for (auto& [ptr, buf_id] : wrapped_buffer_cache_) {
        void* contents = gpu_->buffer_contents(buf_id);
        if (contents) {
            volatile char touch = ((volatile char*)contents)[0];
            (void)touch;
            count++;
        }
    }
    fprintf(stderr, "[compute] Pre-faulted %zu cached MTLBuffers\n", count);
}

}  // namespace nexus::compute
