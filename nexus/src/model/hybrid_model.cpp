/// NEXUS HybridModel — Qwen3-Coder-Next hybrid SSM+Attention execution.
///
/// Layer-by-layer streaming execution for the hybrid architecture.
/// Type A layers (SSM+MoE) use a simplified linear-attention approximation
/// of the Gated DeltaNet — full SSM support will follow in a later phase.
/// Type B layers (Attention+MoE) use standard GQA with Q/K RMSNorm.
/// Both types share the MoE FFN path: top-10 expert routing with SwiGLU.

#include "model/hybrid_model.h"
#include "memory/memory_manager.h"
#include "compute/compute_dispatch.h"
#include "compute/accelerate/gemm.h"
#include "quant/gptq.h"
#include "core/scheduler.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <unordered_map>
#include <sys/mman.h>
#include <sys/sysctl.h>

namespace nexus::model {

// ─── Implementation ────────────────────────────────────────────────────────────

struct HybridModel::Impl {
    format::ModelManifest manifest;
    format::NXFReader* reader = nullptr;
    MemoryManager* memory = nullptr;
    std::unique_ptr<Scheduler> scheduler;

    // Per-layer weights (loaded on demand, evicted after use)
    std::vector<HybridLayerWeights> layer_weights;

    // KV cache (persistent across tokens) — only used for attention layers
    // and for the simplified attention path on SSM layers
    std::vector<HybridKVCache> kv_cache;

    // Embedding table
    float* token_embeddings = nullptr;   // [vocab_size, hidden_dim]
    float* output_norm = nullptr;        // [hidden_dim]
    float* output_weight = nullptr;      // [vocab_size, hidden_dim]
    HybridLayerWeights::RawWeight output_weight_raw;  // Raw INT4 for fused GPU logits

    // Scratch buffers for intermediate activations
    float* hidden_state = nullptr;       // [hidden_dim]
    float* residual = nullptr;           // [hidden_dim]
    float* attn_output = nullptr;        // [max(q_dim, hidden_dim)]
    float* ffn_output = nullptr;         // [hidden_dim]
    float* logits = nullptr;             // [vocab_size]
    float* norm_buf = nullptr;           // [hidden_dim] scratch for norms

    // Large scratch for QKV projections and MoE computation
    float* qkv_buf = nullptr;            // [max_qkv_dim] for fused QKV output
    float* moe_scratch = nullptr;        // [hidden_dim] for MoE accumulation
    float* gate_logits_buf = nullptr;    // [max_num_experts] for router

    // Dequantized weight buffers — per-layer tracking for cleanup on eviction.
    struct DequantBuf { float* ptr; size_t size; };
    std::unordered_map<uint32_t, std::vector<DequantBuf>> layer_dequant_buffers;
    uint32_t current_loading_layer = 0;  // Set during load_layer_weights

    // Sequence state
    int current_seq_len = 0;
    int max_seq_len = 0;

    // Resident mode: all weights pre-loaded as MTLBuffers, zero per-token loading.
    // Enabled when total model size fits in 80% of available RAM.
    bool resident_mode = false;
    size_t total_model_size_bytes = 0;

    // Detected dimensions (resolved from tensor shapes at load time)
    int max_qkv_dim = 0;

    // ─── Layer execution ──────────────────────────────────────────────────
    void execute_layer(uint32_t layer_idx, float* x, int seq_pos);
    void execute_ssm_moe_layer(uint32_t layer_idx, float* x, int seq_pos);
    void execute_attention_moe_layer(uint32_t layer_idx, float* x, int seq_pos);

    // ─── Attention helpers ────────────────────────────────────────────────
    void gqa_attention(const float* q, int q_dim,
                       const float* k_new, const float* v_new, int kv_dim,
                       HybridKVCache& kv, int seq_pos,
                       int n_heads, int n_kv_heads, int head_dim,
                       float* out);

    // ─── MoE FFN ─────────────────────────────────────────────────────────
    void moe_ffn(uint32_t layer_idx, const float* x, float* out);

    // ─── Weight loading ───────────────────────────────────────────────────
    HybridLayerType detect_layer_type(uint32_t layer_idx);
    bool load_layer_weights(uint32_t layer_idx);
    void evict_layer_weights(uint32_t layer_idx);
    bool load_embedding_weights();

    // Load tensor and also store raw INT4 pointer for fused GPU path
    float* load_tensor_raw(const char* name, HybridLayerWeights::RawWeight& raw,
                           std::vector<int64_t>* shape_out = nullptr);

    // Fused GEMM: uses INT4 GPU path if raw data available, else FP32 GEMM
    void fused_gemm(const float* input, const float* weight,
                    const HybridLayerWeights::RawWeight& raw,
                    float* output, int M, int N, int K) {
        if (raw.data && raw.bytes > 0) {
            compute::global_compute().gemm_int4(input, raw.data, raw.bytes,
                                                 output, M, N, K);
        } else if (weight) {
            compute::global_compute().gemm(input, weight, output, M, N, K);
        }
    }

    // ─── Tensor loading helpers ───────────────────────────────────────────
    /// Load a tensor by name, returning the mapped pointer and optionally
    /// filling in shape dimensions. Returns nullptr if tensor not found.
    float* load_tensor(const char* name, std::vector<int64_t>* shape_out = nullptr);

    /// Build layer tensor name: "layers.<idx>.<suffix>"
    static std::string layer_tensor_name(uint32_t layer_idx, const char* suffix);

    // ─── Sampling ─────────────────────────────────────────────────────────
    int32_t sample_token(const float* logits_ptr, const SamplingParams& params);
};

// ─── Utility: build tensor name ─────────────────────────────────────────────

std::string HybridModel::Impl::layer_tensor_name(uint32_t layer_idx, const char* suffix) {
    char buf[256];
    snprintf(buf, sizeof(buf), "layers.%u.%s", layer_idx, suffix);
    return std::string(buf);
}

float* HybridModel::Impl::load_tensor(const char* name,
                                        std::vector<int64_t>* shape_out) {
    const auto* info = reader->get_tensor(name);
    if (!info || info->chunks.empty()) { fprintf(stderr, "not found\n"); return nullptr; }

    if (shape_out) {
        *shape_out = info->shape;
    }

    const auto& chunk = info->chunks[0];
    const void* mapped = reader->map_chunk(chunk);
    if (!mapped) return nullptr;

    // Compute total number of elements from shape
    int64_t num_elements = 1;
    for (auto s : info->shape) num_elements *= s;
    if (num_elements <= 0) return nullptr;

    Codec codec = chunk.codec;

    // If data is already FP32, return directly
    if (codec == Codec::FP32) {
        return const_cast<float*>(static_cast<const float*>(mapped));
    }

    // For small tensors (norms, biases < 64KB), the NXF writer stores them
    // as FP32 regardless of the target codec. Check if the chunk size matches
    // FP32 expectations.
    if (chunk.decompressed_size == static_cast<uint32_t>(num_elements * 4)) {
        // Data is actually FP32 (small tensor passthrough)
        return const_cast<float*>(static_cast<const float*>(mapped));
    }

    // Dequantize: allocate FP32 buffer and convert from quantized format
    size_t fp32_bytes = num_elements * sizeof(float);
    // Round up to page alignment
    size_t alloc_size = (fp32_bytes + kPageSize - 1) & ~(kPageSize - 1);
    float* fp32_buf = static_cast<float*>(memory->alloc_pages(alloc_size));
    if (!fp32_buf) {
        fprintf(stderr, "[nexus] WARNING: Failed to allocate dequant buffer for %s (%lld elements)\n",
                name, (long long)num_elements);
        return nullptr;
    }
    layer_dequant_buffers[current_loading_layer].push_back({fp32_buf, alloc_size});

    const uint8_t* src = static_cast<const uint8_t*>(mapped);

    switch (codec) {
        case Codec::FP16: {
            // FP16 → FP32
            const uint16_t* fp16 = reinterpret_cast<const uint16_t*>(src);
            for (int64_t i = 0; i < num_elements; i++) {
                uint32_t h = fp16[i];
                uint32_t sign = (h & 0x8000) << 16;
                uint32_t exp = (h >> 10) & 0x1F;
                uint32_t mant = h & 0x3FF;
                uint32_t f;
                if (exp == 0) {
                    f = sign;  // zero/denorm → zero
                } else if (exp == 31) {
                    f = sign | 0x7F800000 | (mant << 13);  // inf/nan
                } else {
                    f = sign | ((exp + 112) << 23) | (mant << 13);
                }
                memcpy(&fp32_buf[i], &f, 4);
            }
            break;
        }
        case Codec::INT4:
        case Codec::GPTQ:
        case Codec::AWQ: {
            // Fast INT4 dequant: unpack nibbles to FP32 using vectorized ops.
            // Maps [0,15] → [-1.0, 0.875] via (nibble - 8) / 8.
            size_t avail_bytes = chunk.compressed_size;
            int64_t max_pairs = static_cast<int64_t>(avail_bytes);
            int64_t safe_elements = std::min(num_elements, max_pairs * 2);
            int64_t safe_bytes = safe_elements / 2;

            // Process 8 bytes (16 elements) at a time
            int64_t i_byte = 0;
            for (; i_byte + 7 < safe_bytes; i_byte += 8) {
                for (int b = 0; b < 8; b++) {
                    uint8_t packed = src[i_byte + b];
                    int64_t idx = (i_byte + b) * 2;
                    fp32_buf[idx]     = (static_cast<float>(packed & 0x0F) - 8.0f) * 0.125f;
                    fp32_buf[idx + 1] = (static_cast<float>(packed >> 4) - 8.0f) * 0.125f;
                }
            }
            // Scalar tail
            for (; i_byte < safe_bytes; i_byte++) {
                uint8_t packed = src[i_byte];
                int64_t idx = i_byte * 2;
                fp32_buf[idx] = (static_cast<float>(packed & 0x0F) - 8.0f) * 0.125f;
                if (idx + 1 < safe_elements) {
                    fp32_buf[idx + 1] = (static_cast<float>(packed >> 4) - 8.0f) * 0.125f;
                }
            }
            // Zero-fill any remaining elements beyond available data
            for (int64_t i = safe_elements; i < num_elements; i++) {
                fp32_buf[i] = 0.0f;
            }
            break;
        }
        case Codec::INT8: {
            const int8_t* i8 = reinterpret_cast<const int8_t*>(src);
            // Simple symmetric dequant
            for (int64_t i = 0; i < num_elements; i++) {
                fp32_buf[i] = static_cast<float>(i8[i]) / 127.0f;
            }
            break;
        }
        default:
            // Unknown codec — zero fill
            memset(fp32_buf, 0, fp32_bytes);
            fprintf(stderr, "[nexus] WARNING: unknown codec %d for %s, zero-filling\n",
                    (int)codec, name);
            break;
    }

    return fp32_buf;
}

float* HybridModel::Impl::load_tensor_raw(const char* name,
                                            HybridLayerWeights::RawWeight& raw,
                                            std::vector<int64_t>* shape_out) {
    // First, store the raw mapped pointer for fused GPU path
    const auto* info = reader->get_tensor(name);
    if (info && !info->chunks.empty()) {
        const auto& chunk = info->chunks[0];
        const void* mapped = reader->map_chunk(chunk);
        if (mapped) {
            raw.data = mapped;
            raw.bytes = chunk.compressed_size;
        }
    }
    // Then do the normal load (with CPU dequant as fallback)
    return load_tensor(name, shape_out);
}

// ─── Layer type detection ───────────────────────────────────────────────────

HybridLayerType HybridModel::Impl::detect_layer_type(uint32_t layer_idx) {
    // Type A (SSM+MoE) has "attn_qkv.weight" (fused QKV)
    // Type B (Attention+MoE) has "attention.wq.weight" (separate Q)
    std::string fused_name = layer_tensor_name(layer_idx, "attn_qkv.weight");
    std::string sep_name = layer_tensor_name(layer_idx, "attention.wq.weight");

    if (reader->get_tensor(fused_name)) {
        return HybridLayerType::SSM_MoE;
    }
    if (reader->get_tensor(sep_name)) {
        return HybridLayerType::Attention_MoE;
    }
    return HybridLayerType::Unknown;
}

// ─── Creation ──────────────────────────────────────────────────────────────

HybridModel::~HybridModel() = default;

std::unique_ptr<HybridModel> HybridModel::create(
    const format::ModelManifest& manifest,
    format::NXFReader& reader,
    MemoryManager& memory) {

    auto model = std::unique_ptr<HybridModel>(new HybridModel());
    model->impl_ = std::make_unique<Impl>();
    auto& m = *model->impl_;

    m.manifest = manifest;
    m.reader = &reader;
    m.memory = &memory;

    // Initialize scheduler for layer streaming
    SchedulerConfig sched_config;
    sched_config.prefetch_window = 2;
    m.scheduler = std::make_unique<Scheduler>(memory, manifest.num_layers, sched_config);

    // Allocate per-layer structures
    m.layer_weights.resize(manifest.num_layers);
    m.kv_cache.resize(manifest.num_layers);

    m.max_seq_len = manifest.max_seq_len;

    // Detect layer types and determine max dimensions needed for buffers
    int ssm_count = 0, attn_count = 0, unknown_count = 0;
    m.max_qkv_dim = 0;

    for (uint32_t i = 0; i < manifest.num_layers; i++) {
        HybridLayerType lt = m.detect_layer_type(i);
        m.layer_weights[i].type = lt;

        switch (lt) {
            case HybridLayerType::SSM_MoE: ssm_count++; break;
            case HybridLayerType::Attention_MoE: attn_count++; break;
            default: unknown_count++; break;
        }

        // Probe QKV dimensions from tensor shapes
        if (lt == HybridLayerType::SSM_MoE) {
            std::string tn = Impl::layer_tensor_name(i, "attn_qkv.weight");
            const auto* info = reader.get_tensor(tn);
            if (info && info->shape.size() >= 2) {
                int qkv_dim = static_cast<int>(info->shape[1]);
                if (qkv_dim > m.max_qkv_dim) m.max_qkv_dim = qkv_dim;
            }
        } else if (lt == HybridLayerType::Attention_MoE) {
            std::string tn = Impl::layer_tensor_name(i, "attention.wq.weight");
            const auto* info = reader.get_tensor(tn);
            if (info && info->shape.size() >= 2) {
                int q_dim = static_cast<int>(info->shape[1]);
                if (q_dim > m.max_qkv_dim) m.max_qkv_dim = q_dim;
            }
        }
    }

    // Fallback: if we couldn't detect dimensions, use reasonable defaults
    if (m.max_qkv_dim == 0) {
        m.max_qkv_dim = manifest.hidden_dim * 4;  // Conservative estimate
    }

    // Infer vocab_size from tensor shapes if manifest has 0
    if (m.manifest.vocab_size == 0) {
        const auto* emb = reader.get_tensor("tok_embeddings.weight");
        if (emb && emb->shape.size() >= 2) {
            m.manifest.vocab_size = static_cast<uint32_t>(emb->shape[1]);
            fprintf(stderr, "[nexus] Inferred vocab_size=%u from tok_embeddings shape\n",
                    m.manifest.vocab_size);
        } else {
            const auto* out = reader.get_tensor("output.weight");
            if (out && out->shape.size() >= 2) {
                m.manifest.vocab_size = static_cast<uint32_t>(out->shape[1]);
                fprintf(stderr, "[nexus] Inferred vocab_size=%u from output.weight shape\n",
                        m.manifest.vocab_size);
            } else {
                m.manifest.vocab_size = 151936;  // Qwen default
                fprintf(stderr, "[nexus] WARNING: Could not infer vocab_size, using default %u\n",
                        m.manifest.vocab_size);
            }
        }
    }

    fprintf(stderr, "[nexus] HybridModel: %d SSM+MoE layers, %d Attn+MoE layers",
            ssm_count, attn_count);
    if (unknown_count > 0) {
        fprintf(stderr, ", %d unknown (will use fallback)", unknown_count);
    }
    fprintf(stderr, "\n");

    // Allocate scratch buffers
    size_t hidden_bytes = manifest.hidden_dim * sizeof(float);
    m.hidden_state = static_cast<float*>(memory.alloc_pages(hidden_bytes));
    m.residual = static_cast<float*>(memory.alloc_pages(hidden_bytes));
    m.attn_output = static_cast<float*>(memory.alloc_pages(
        std::max(hidden_bytes, static_cast<size_t>(m.max_qkv_dim) * sizeof(float))));
    m.ffn_output = static_cast<float*>(memory.alloc_pages(hidden_bytes));
    m.logits = static_cast<float*>(memory.alloc_pages(m.manifest.vocab_size * sizeof(float)));
    m.norm_buf = static_cast<float*>(memory.alloc_pages(hidden_bytes));
    m.qkv_buf = static_cast<float*>(memory.alloc_pages(
        static_cast<size_t>(m.max_qkv_dim) * sizeof(float)));
    m.moe_scratch = static_cast<float*>(memory.alloc_pages(hidden_bytes));

    // Gate logits buffer — size based on max expected experts
    int max_experts = manifest.num_experts > 0 ? manifest.num_experts : 512;
    m.gate_logits_buf = static_cast<float*>(
        memory.alloc_pages(max_experts * sizeof(float)));

    // Allocate KV cache for each layer
    // For Type A (SSM) layers using simplified attention, we derive kv_dim from
    // the fused QKV tensor. For Type B layers, we use the manifest's kv config.
    for (uint32_t i = 0; i < manifest.num_layers; i++) {
        auto& lw = m.layer_weights[i];
        int kv_dim = 0;

        if (lw.type == HybridLayerType::Attention_MoE) {
            // Separate K projection determines KV dim
            std::string tn = Impl::layer_tensor_name(i, "attention.wk.weight");
            const auto* info = reader.get_tensor(tn);
            if (info && info->shape.size() >= 2) {
                kv_dim = static_cast<int>(info->shape[1]);
            }
            if (kv_dim == 0) {
                kv_dim = manifest.num_kv_heads * manifest.head_dim;
            }
        } else if (lw.type == HybridLayerType::SSM_MoE) {
            // For simplified attention on SSM layers, we'll use a portion of
            // the fused QKV output as K and V. We need to figure out the split.
            // The fused QKV output dim is known; we'll determine K/V dim at
            // weight-load time. For now, allocate based on manifest KV heads.
            kv_dim = manifest.num_kv_heads * manifest.head_dim;
            // If head_dim is not set reasonably, use a fallback
            if (kv_dim == 0) kv_dim = 512;
        } else {
            // Unknown layer — allocate a conservative KV cache
            kv_dim = manifest.num_kv_heads * manifest.head_dim;
            if (kv_dim == 0) kv_dim = 512;
        }

        m.kv_cache[i].kv_dim = kv_dim;
        // Cap max_seq_len for KV cache to avoid OOM (use 4096 for initial testing)
        int effective_seq = std::min(static_cast<int>(m.manifest.max_seq_len), 4096);
        m.max_seq_len = effective_seq;
        size_t kv_bytes = static_cast<size_t>(effective_seq) * kv_dim * sizeof(float);
        m.kv_cache[i].keys = static_cast<float*>(memory.alloc_pages(kv_bytes));
        m.kv_cache[i].values = static_cast<float*>(memory.alloc_pages(kv_bytes));
        m.kv_cache[i].seq_len = 0;
    }

    // Load embedding weights (these stay resident)
    fprintf(stderr, "[nexus] Loading embeddings (vocab=%u, dim=%u, ~%.0f MB FP32)...\n",
            m.manifest.vocab_size, manifest.hidden_dim,
            (double)m.manifest.vocab_size * manifest.hidden_dim * 4 / (1024*1024));
    if (!m.load_embedding_weights()) {
        fprintf(stderr, "[nexus] WARNING: Could not load embeddings, using zero initialization\n");
    }
    fprintf(stderr, "[nexus] Embeddings: tok=%p norm=%p out=%p\n",
            (void*)m.token_embeddings, (void*)m.output_norm, (void*)m.output_weight);

    // ── Resident mode detection ──────────────────────────────────────────
    // If the entire model fits in 80% of available RAM, preload ALL weights
    // and wrap them as MTLBuffers once. This eliminates per-token weight
    // loading entirely — the decode loop becomes pure GPU dispatch.
    {
        m.total_model_size_bytes = reader.file_size();
        // Get available RAM (macOS sysctl)
        uint64_t ram_bytes = 0;
        size_t ram_size = sizeof(ram_bytes);
        if (sysctlbyname("hw.memsize", &ram_bytes, &ram_size, nullptr, 0) == 0) {
            uint64_t ram_limit = static_cast<uint64_t>(ram_bytes * 0.8);
            if (m.total_model_size_bytes < ram_limit) {
                m.resident_mode = true;
            }
        } else {
            // Fallback: assume 48 GB if sysctl fails
            if (m.total_model_size_bytes < static_cast<uint64_t>(48ULL * 1024 * 1024 * 1024 * 0.8)) {
                m.resident_mode = true;
            }
        }

        if (m.resident_mode) {
            fprintf(stderr, "[nexus] Resident mode: model (%zu MB) fits in RAM, preloading ALL weights...\n",
                    m.total_model_size_bytes / (1024 * 1024));

            // Pre-load ALL layer weights upfront
            for (uint32_t i = 0; i < manifest.num_layers; i++) {
                m.load_layer_weights(i);
            }

            // The wrapped_buffer_cache_ in ComputeDispatch gets populated on
            // first gemm_int4 call per weight pointer during the first token.
            // Since all layer weights are now loaded, that first pass populates
            // the cache. Every subsequent token reuses cached MTLBuffers with
            // zero overhead. preload_all_buffers() is called after prefill to
            // fault all pages into physical RAM.
            //
            // Pre-allocate activation buffers to max size to avoid per-token realloc.
            auto& gpu = compute::global_compute();
            size_t max_act_bytes = static_cast<size_t>(1) * std::max(m.max_qkv_dim, (int)manifest.hidden_dim) * sizeof(float);
            size_t max_out_bytes = static_cast<size_t>(std::max((int)manifest.vocab_size, m.max_qkv_dim)) * sizeof(float);
            gpu.pre_allocate_activation_buffer(max_act_bytes, max_out_bytes);

            size_t total_mb = m.total_model_size_bytes / (1024 * 1024);
            fprintf(stderr, "[nexus] Resident mode: ALL weights preloaded (%zu MB)\n", total_mb);
        }
    }

    fprintf(stderr, "[nexus] HybridModel initialized: %u layers, %s mode\n",
            manifest.num_layers, m.resident_mode ? "resident" : "streaming");
    return model;
}

// ─── Prefill ───────────────────────────────────────────────────────────────

void HybridModel::prefill(const std::vector<int32_t>& tokens) {
    auto& m = *impl_;
    int num_tokens = static_cast<int>(tokens.size());

    for (int t = 0; t < num_tokens; t++) {
        
        // Lookup embedding
        if (m.token_embeddings && tokens[t] >= 0 &&
            tokens[t] < static_cast<int32_t>(m.manifest.vocab_size)) {
            int tid = tokens[t];
            int dim = m.manifest.hidden_dim;
            int vocab = m.manifest.vocab_size;
            for (int i = 0; i < dim; i++) {
                m.hidden_state[i] = m.token_embeddings[i * vocab + tid];
            }
        } else {
            memset(m.hidden_state, 0, m.manifest.hidden_dim * sizeof(float));
        }

        // Stream through all layers
        m.scheduler->execute_layers([&](uint32_t layer_idx) {
            if (!m.resident_mode) m.load_layer_weights(layer_idx);
            // Batch all GPU dispatches within this layer into one command buffer
            compute::global_compute().begin_gpu_batch();
            m.execute_layer(layer_idx, m.hidden_state, m.current_seq_len);
            compute::global_compute().end_gpu_batch();
        });

        m.current_seq_len++;
    }

    // In resident mode, after the first full pass all weight MTLBuffers are
    // cached. Fault every page into physical RAM so decode tokens pay zero
    // page-fault cost.
    if (m.resident_mode) {
        compute::global_compute().preload_all_buffers();
    }

    fprintf(stderr, "[nexus] Prefill complete: %d tokens, seq_len=%d\n",
            num_tokens, m.current_seq_len);
}

// ─── Decode step ───────────────────────────────────────────────────────────

int32_t HybridModel::decode_step(const SamplingParams& params) {
    auto& m = *impl_;

    if (m.current_seq_len >= m.max_seq_len) {
        fprintf(stderr, "[nexus] Max sequence length reached (%d)\n", m.max_seq_len);
        return -1;
    }

    // Stream through all layers
    m.scheduler->execute_layers([&](uint32_t layer_idx) {
        if (!m.resident_mode) m.load_layer_weights(layer_idx);
        m.execute_layer(layer_idx, m.hidden_state, m.current_seq_len);
    });

    // Apply output norm
    if (m.output_norm) {
        compute::global_compute().rmsnorm(m.hidden_state, m.hidden_state, m.output_norm,
                         m.manifest.hidden_dim, m.manifest.rms_norm_eps);
    }

    // Compute logits: hidden_state @ output_weight^T -> logits
    if (m.output_weight) {
        m.fused_gemm(m.hidden_state, m.output_weight, m.output_weight_raw,
                     m.logits, 1, m.manifest.vocab_size, m.manifest.hidden_dim);
    }

    int32_t next_token = m.sample_token(m.logits, params);
    m.current_seq_len++;
    return next_token;
}

void HybridModel::reset_kv_cache() {
    for (auto& kv : impl_->kv_cache) {
        kv.seq_len = 0;
    }
    impl_->current_seq_len = 0;
}

int HybridModel::seq_len() const {
    return impl_->current_seq_len;
}

// ─── Layer execution dispatch ──────────────────────────────────────────────

void HybridModel::Impl::execute_layer(uint32_t layer_idx, float* x, int seq_pos) {
    auto& lw = layer_weights[layer_idx];
    switch (lw.type) {
        case HybridLayerType::SSM_MoE:
            execute_ssm_moe_layer(layer_idx, x, seq_pos);
            break;
        case HybridLayerType::Attention_MoE:
            execute_attention_moe_layer(layer_idx, x, seq_pos);
            break;
        default:
            // Unknown layer type — skip (pass through)
            fprintf(stderr, "[nexus] WARNING: Unknown layer type at layer %u, skipping\n",
                    layer_idx);
            break;
    }
}

// ─── Type A: SSM + MoE layer (simplified as linear attention) ──────────────

void HybridModel::Impl::execute_ssm_moe_layer(uint32_t layer_idx, float* x,
                                                 int seq_pos) {
    auto& lw = layer_weights[layer_idx];
    int dim = manifest.hidden_dim;

    // ── Save residual ──
    memcpy(residual, x, dim * sizeof(float));

    // ── Pre-attention RMSNorm ──
    if (lw.attention_norm) {
        compute::global_compute().rmsnorm(norm_buf, x, lw.attention_norm, dim, manifest.rms_norm_eps);
    } else {
        memcpy(norm_buf, x, dim * sizeof(float));
    }

    // ── Fused QKV projection ──
    // attn_qkv: [hidden_dim, qkv_out_dim] — single matmul
    int qkv_dim = lw.qkv_out_dim;
    if (lw.attn_qkv && qkv_dim > 0) {
        fused_gemm(norm_buf, lw.attn_qkv, lw.attn_qkv_raw, qkv_buf, 1, qkv_dim, dim);
    } else {
        // No QKV weights — zero out and skip attention
        memset(qkv_buf, 0, dim * sizeof(float));
        // Residual passthrough
        goto ssm_residual;
    }

    {
        // ── Split fused QKV output ──
        // Heuristic split: given qkv_dim and manifest head config, determine
        // Q, K, V sizes. The separate attention layer has wq=[hidden,q_dim],
        // wk=[hidden,k_dim], wv=[hidden,v_dim]. For the fused form, we assume
        // the output is packed as [Q, K, V] contiguously.
        //
        // We try to derive head_dim from the K projection size in any attention
        // layer, then use: K_dim = V_dim = num_kv_heads * head_dim,
        // Q_dim = qkv_dim - K_dim - V_dim.
        //
        // Fallback: if we can't determine, split proportionally.
        int k_dim = lw.k_out_dim;
        int v_dim = lw.v_out_dim;
        int q_dim;

        if (k_dim > 0 && v_dim > 0) {
            q_dim = qkv_dim - k_dim - v_dim;
        } else {
            // Use manifest kv_heads * head_dim as K/V dim
            k_dim = manifest.num_kv_heads * manifest.head_dim;
            v_dim = k_dim;
            q_dim = qkv_dim - k_dim - v_dim;

            // Safety: if the split doesn't make sense, fall back to even split
            if (q_dim <= 0 || k_dim <= 0) {
                // Can't determine split — use entire output as Q and skip KV
                q_dim = qkv_dim;
                k_dim = 0;
                v_dim = 0;
            }
        }

        float* q_ptr = qkv_buf;
        float* k_ptr = qkv_buf + q_dim;
        float* v_ptr = qkv_buf + q_dim + k_dim;

        // ── Simplified attention (approximation of DeltaNet) ──
        // We use standard softmax attention as a working approximation.
        // The gate weight is applied as an element-wise gate on the output.
        if (k_dim > 0 && v_dim > 0) {
            auto& kv = kv_cache[layer_idx];
            // Determine head dimensions for this layer
            int n_heads = manifest.num_heads;
            int n_kv_heads = manifest.num_kv_heads;
            int head_dim_q = (n_heads > 0) ? (q_dim / n_heads) : q_dim;
            int head_dim_kv = (n_kv_heads > 0) ? (k_dim / n_kv_heads) : k_dim;

            // If head dims don't match, use the smaller one for attention
            // and truncate Q heads accordingly. This handles the case where
            // Q has a larger per-head dimension than KV.
            int attn_head_dim = std::min(head_dim_q, head_dim_kv);

            if (attn_head_dim > 0 && n_heads > 0 && n_kv_heads > 0) {
                gqa_attention(q_ptr, q_dim, k_ptr, v_ptr, k_dim,
                              kv, seq_pos, n_heads, n_kv_heads,
                              attn_head_dim, attn_output);
            } else {
                // Fallback: just use the Q projection as output
                memcpy(attn_output, q_ptr,
                       std::min(q_dim, dim) * sizeof(float));
                if (q_dim < dim) {
                    memset(attn_output + q_dim, 0, (dim - q_dim) * sizeof(float));
                }
            }
        } else {
            // No K/V — use Q directly (truncated/padded to hidden_dim)
            int copy_dim = std::min(q_dim, dim);
            memcpy(attn_output, q_ptr, copy_dim * sizeof(float));
            if (copy_dim < dim) {
                memset(attn_output + copy_dim, 0, (dim - copy_dim) * sizeof(float));
            }
        }

        // ── Apply SSM output projection if available ──
        if (lw.ssm_out && lw.ssm_out_dim > 0) {
            // ssm_out: [ssm_out_dim, hidden_dim]
            // We project attn_output (ssm_out_dim) -> hidden_dim
            int in_dim = lw.ssm_out_dim;
            fused_gemm(attn_output, lw.ssm_out, lw.ssm_out_raw, ffn_output, 1, dim, in_dim);
            memcpy(attn_output, ffn_output, dim * sizeof(float));
        }

        // ── Apply gate if present ──
        // The attn_gate provides a sigmoid gate on the attention output.
        // gate_output = sigmoid(x @ attn_gate) * attn_output
        if (lw.attn_gate && lw.gate_out_dim > 0) {
            // Compute gate: norm_buf @ attn_gate -> gate values
            std::vector<float> gate_vals(lw.gate_out_dim);
            fused_gemm(norm_buf, lw.attn_gate, lw.attn_gate_raw, gate_vals.data(), 1, lw.gate_out_dim, dim);
            // Apply sigmoid gate element-wise (up to min of gate_dim and dim)
            int gate_dim = std::min(lw.gate_out_dim, dim);
            for (int i = 0; i < gate_dim; i++) {
                float sig = 1.0f / (1.0f + expf(-gate_vals[i]));
                attn_output[i] *= sig;
            }
        }
    }

ssm_residual:
    // ── Residual connection after attention/SSM ──
    for (int i = 0; i < dim; i++) {
        x[i] = residual[i] + attn_output[i];
    }

    // ── Save residual for post-FFN ──
    memcpy(residual, x, dim * sizeof(float));

    // ── Post-attention RMSNorm ──
    if (lw.post_attention_norm) {
        compute::global_compute().rmsnorm(norm_buf, x, lw.post_attention_norm, dim, manifest.rms_norm_eps);
    } else {
        memcpy(norm_buf, x, dim * sizeof(float));
    }

    // ── MoE FFN ──
    moe_ffn(layer_idx, norm_buf, ffn_output);

    // ── Residual connection after FFN ──
    for (int i = 0; i < dim; i++) {
        x[i] = residual[i] + ffn_output[i];
    }
}

// ─── Type B: Full Attention + MoE layer ────────────────────────────────────

void HybridModel::Impl::execute_attention_moe_layer(uint32_t layer_idx, float* x,
                                                       int seq_pos) {
    auto& lw = layer_weights[layer_idx];
    int dim = manifest.hidden_dim;

    // ── Save residual ──
    memcpy(residual, x, dim * sizeof(float));

    // ── Pre-attention RMSNorm ──
    if (lw.attention_norm) {
        compute::global_compute().rmsnorm(norm_buf, x, lw.attention_norm, dim, manifest.rms_norm_eps);
    } else {
        memcpy(norm_buf, x, dim * sizeof(float));
    }

    // ── Separate Q/K/V projections ──
    int q_dim = lw.q_out_dim > 0 ? lw.q_out_dim : dim;
    int k_dim = lw.k_out_dim > 0 ? lw.k_out_dim : (manifest.num_kv_heads * manifest.head_dim);
    int v_dim = lw.v_out_dim > 0 ? lw.v_out_dim : k_dim;

    std::vector<float> q(q_dim, 0.0f);
    std::vector<float> k(k_dim, 0.0f);
    std::vector<float> v(v_dim, 0.0f);

    if (lw.wq) fused_gemm(norm_buf, lw.wq, lw.wq_raw, q.data(), 1, q_dim, dim);
    if (lw.wk) fused_gemm(norm_buf, lw.wk, lw.wk_raw, k.data(), 1, k_dim, dim);
    if (lw.wv) fused_gemm(norm_buf, lw.wv, lw.wv_raw, v.data(), 1, v_dim, dim);

    // ── Apply Q/K RMSNorm if present ──
    // Q norm and K norm are applied per-head. The norm weight dimension tells
    // us the granularity. For simplicity, we apply a single RMSNorm across
    // the entire Q/K vector if the weight exists.
    if (lw.q_norm) {
        // q_norm weight might be [head_dim] or [num_heads * head_dim] or [norm_dim].
        // We apply it in chunks matching the norm weight size across Q.
        // For Qwen3-Coder-Next, the norm is [256], and Q might be [8192].
        // We apply the norm in repeating blocks of 256 across Q.
        int norm_dim = 256;  // From the tensor shape [256]
        // Try to get actual dim from tensor shape
        std::string qn_name = layer_tensor_name(layer_idx, "attn_q_norm.weight");
        const auto* qn_info = reader->get_tensor(qn_name);
        if (qn_info && !qn_info->shape.empty()) {
            norm_dim = static_cast<int>(qn_info->shape[0]);
        }

        // Apply norm in blocks of norm_dim across the Q vector
        for (int offset = 0; offset + norm_dim <= q_dim; offset += norm_dim) {
            compute::global_compute().rmsnorm(q.data() + offset, q.data() + offset,
                             lw.q_norm, norm_dim, manifest.rms_norm_eps);
        }
    }

    if (lw.k_norm) {
        int norm_dim = 256;
        std::string kn_name = layer_tensor_name(layer_idx, "attn_k_norm.weight");
        const auto* kn_info = reader->get_tensor(kn_name);
        if (kn_info && !kn_info->shape.empty()) {
            norm_dim = static_cast<int>(kn_info->shape[0]);
        }

        for (int offset = 0; offset + norm_dim <= k_dim; offset += norm_dim) {
            compute::global_compute().rmsnorm(k.data() + offset, k.data() + offset,
                             lw.k_norm, norm_dim, manifest.rms_norm_eps);
        }
    }

    // ── GQA Attention with KV cache ──
    auto& kv = kv_cache[layer_idx];
    int n_heads = manifest.num_heads;
    int n_kv_heads = manifest.num_kv_heads;
    // Derive head_dim from K projection: head_dim = k_dim / n_kv_heads
    int head_dim = (n_kv_heads > 0) ? (k_dim / n_kv_heads) : manifest.head_dim;
    // For Q, head_dim_q = q_dim / n_heads might differ from head_dim_kv.
    // Use the KV head_dim for the attention computation.
    int attn_head_dim = head_dim;

    if (attn_head_dim > 0 && n_heads > 0 && n_kv_heads > 0) {
        gqa_attention(q.data(), q_dim, k.data(), v.data(), k_dim,
                      kv, seq_pos, n_heads, n_kv_heads, attn_head_dim,
                      attn_output);
    } else {
        // Fallback: just use Q as attention output
        int copy_dim = std::min(q_dim, dim);
        memcpy(attn_output, q.data(), copy_dim * sizeof(float));
        if (copy_dim < dim) {
            memset(attn_output + copy_dim, 0, (dim - copy_dim) * sizeof(float));
        }
    }

    // ── Output projection ──
    // wo shape is [wo_input_dim, hidden_dim]. For Qwen3: [4096, 2048].
    // attn_output contains n_heads * head_dim_kv = 4096 elements.
    if (lw.wo) {
        // Determine wo input dimension from tensor shape
        int wo_in_dim = n_heads * head_dim;  // 16 * 256 = 4096
        if (wo_in_dim <= 0) wo_in_dim = q_dim;  // fallback
        fused_gemm(attn_output, lw.wo, lw.wo_raw, ffn_output, 1, dim, wo_in_dim);
        memcpy(attn_output, ffn_output, dim * sizeof(float));
    } else {
        // No output projection — truncate/pad to hidden_dim
        if (q_dim > dim) {
            // attn_output already has the first `dim` values
        } else {
            memset(attn_output + q_dim, 0, (dim - q_dim) * sizeof(float));
        }
    }

    // ── Residual connection after attention ──
    for (int i = 0; i < dim; i++) {
        x[i] = residual[i] + attn_output[i];
    }

    // ── Save residual for post-FFN ──
    memcpy(residual, x, dim * sizeof(float));

    // ── Post-attention RMSNorm ──
    if (lw.post_attention_norm) {
        compute::global_compute().rmsnorm(norm_buf, x, lw.post_attention_norm, dim, manifest.rms_norm_eps);
    } else {
        memcpy(norm_buf, x, dim * sizeof(float));
    }

    // ── MoE FFN ──
    moe_ffn(layer_idx, norm_buf, ffn_output);

    // ── Residual connection after FFN ──
    for (int i = 0; i < dim; i++) {
        x[i] = residual[i] + ffn_output[i];
    }
}

// ─── GQA Attention (shared by both layer types) ────────────────────────────

void HybridModel::Impl::gqa_attention(const float* q, int q_dim,
                                        const float* k_new, const float* v_new,
                                        int kv_dim, HybridKVCache& kv,
                                        int seq_pos, int n_heads, int n_kv_heads,
                                        int head_dim, float* out) {
    int dim = manifest.hidden_dim;

    // Determine the actual KV dim to store in cache.
    // kv.kv_dim was set at init time. Use the smaller of kv_dim and kv.kv_dim
    // to avoid writing past the allocated cache.
    int store_kv_dim = std::min(kv_dim, kv.kv_dim);

    // Store K, V in cache
    if (kv.keys && kv.values && seq_pos < max_seq_len && store_kv_dim > 0) {
        memcpy(kv.keys + seq_pos * kv.kv_dim, k_new,
               store_kv_dim * sizeof(float));
        memcpy(kv.values + seq_pos * kv.kv_dim, v_new,
               store_kv_dim * sizeof(float));
        kv.seq_len = seq_pos + 1;
    }

    // GQA: n_heads query heads grouped over n_kv_heads KV heads
    int heads_per_group = (n_kv_heads > 0) ? (n_heads / n_kv_heads) : 1;

    // Q head dimension may differ from KV head dimension.
    // head_dim_q = q_dim / n_heads, head_dim_kv = kv_dim / n_kv_heads
    int head_dim_q = (n_heads > 0) ? (q_dim / n_heads) : q_dim;
    int head_dim_kv = (n_kv_heads > 0) ? (store_kv_dim / n_kv_heads) : store_kv_dim;

    // For the dot product, use the minimum of head_dim_q and head_dim_kv.
    // This handles the case where Q and K have different per-head dimensions
    // (e.g., Multi-Head Latent Attention style).
    int dot_dim = std::min(head_dim_q, head_dim_kv);
    dot_dim = std::min(dot_dim, head_dim);  // Also respect the passed head_dim

    // Output accumulator: n_heads × head_dim_kv (V's head dim).
    // For Qwen3: 16 heads × 256 = 4096, matching wo input dim [4096, 2048].
    int out_head_dim = head_dim_kv;
    int total_out_dim = n_heads * out_head_dim;
    std::vector<float> attn_out(total_out_dim, 0.0f);

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_group;
        if (kv_h >= n_kv_heads) kv_h = n_kv_heads - 1;

        const float* q_head = q + h * head_dim_q;

        float scale = 1.0f / sqrtf(static_cast<float>(dot_dim));
        float max_score = -1e30f;
        int seq_len = kv.seq_len;
        if (seq_len <= 0) {
            // No cached KV yet — output zeros for this head
            continue;
        }

        std::vector<float> scores(seq_len);
        for (int t = 0; t < seq_len; t++) {
            const float* k_t = kv.keys + t * kv.kv_dim + kv_h * head_dim_kv;
            float dot = 0.0f;
            for (int d = 0; d < dot_dim; d++) {
                dot += q_head[d] * k_t[d];
            }
            scores[t] = dot * scale;
            if (scores[t] > max_score) max_score = scores[t];
        }

        // Softmax
        float sum_exp = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }
        if (sum_exp > 0.0f) {
            for (int t = 0; t < seq_len; t++) {
                scores[t] /= sum_exp;
            }
        }

        // Weighted sum of values — pack into [h * out_head_dim] slot
        float* out_head = attn_out.data() + h * out_head_dim;
        for (int t = 0; t < seq_len; t++) {
            const float* v_t = kv.values + t * kv.kv_dim + kv_h * head_dim_kv;
            for (int d = 0; d < out_head_dim; d++) {
                out_head[d] += scores[t] * v_t[d];
            }
        }
    }

    // Copy attention output to the output buffer.
    // total_out_dim = n_heads * head_dim_kv (e.g., 16 * 256 = 4096)
    // This feeds into the output projection wo [4096, 2048].
    int out_copy = std::min(total_out_dim, dim);
    memcpy(out, attn_out.data(), out_copy * sizeof(float));
    if (out_copy < dim) {
        memset(out + out_copy, 0, (dim - out_copy) * sizeof(float));
    }
}

// ─── MoE FFN ───────────────────────────────────────────────────────────────

void HybridModel::Impl::moe_ffn(uint32_t layer_idx, const float* x, float* out) {
    auto& lw = layer_weights[layer_idx];
    int dim = manifest.hidden_dim;

    // Initialize output to zero
    memset(out, 0, dim * sizeof(float));

    if (!lw.moe_gate || lw.num_experts <= 0) {
        // No MoE — try to use expert weights as a single dense FFN fallback
        // or just output zeros (layer becomes a skip connection)
        return;
    }

    int num_experts = lw.num_experts;
    int num_active = manifest.num_active_experts > 0 ? manifest.num_active_experts : 10;
    int expert_ffn = lw.expert_ffn_dim;

    if (expert_ffn <= 0 || !lw.expert_w1_raw || !lw.expert_w2_raw || !lw.expert_w3_raw) {
        // Missing expert weight mappings — skip
        return;
    }

    // ── Compute gate logits: x @ moe_gate -> [num_experts] ──
    std::vector<float> gate_logits(num_experts);
    fused_gemm(x, lw.moe_gate, lw.moe_gate_raw, gate_logits.data(),
                     1, num_experts, dim);

    // ── Softmax over gate logits ──
    float max_logit = *std::max_element(gate_logits.begin(), gate_logits.end());
    float sum_exp = 0.0f;
    for (int e = 0; e < num_experts; e++) {
        gate_logits[e] = expf(gate_logits[e] - max_logit);
        sum_exp += gate_logits[e];
    }
    if (sum_exp > 0.0f) {
        for (int e = 0; e < num_experts; e++) {
            gate_logits[e] /= sum_exp;
        }
    }

    // ── Select top-K experts ──
    num_active = std::min(num_active, num_experts);
    std::vector<int> expert_indices(num_experts);
    std::iota(expert_indices.begin(), expert_indices.end(), 0);
    std::partial_sort(expert_indices.begin(),
                      expert_indices.begin() + num_active,
                      expert_indices.end(),
                      [&](int a, int b) {
                          return gate_logits[a] > gate_logits[b];
                      });

    // ── Renormalize selected expert weights ──
    float active_sum = 0.0f;
    for (int i = 0; i < num_active; i++) {
        active_sum += gate_logits[expert_indices[i]];
    }
    std::vector<float> expert_weights(num_active);
    for (int i = 0; i < num_active; i++) {
        expert_weights[i] = (active_sum > 0.0f) ?
            gate_logits[expert_indices[i]] / active_sum : 1.0f / num_active;
    }

    // ── Batched expert execution: ONE GPU dispatch for all 10 experts ──
    // Instead of 30 separate dispatches (10 experts × 3 GEMMs each),
    // send everything to the batched_expert_swiglu shader.
    {
        std::vector<int> active_ids(num_active);
        for (int i = 0; i < num_active; i++) active_ids[i] = expert_indices[i];

        compute::global_compute().batched_moe_ffn(
            x, dim, expert_ffn,
            active_ids.data(), expert_weights.data(), num_active, num_experts,
            lw.expert_w1_raw, lw.expert_w1_bytes,
            lw.expert_w2_raw, lw.expert_w2_bytes,
            lw.expert_w3_raw, lw.expert_w3_bytes,
            out);
        return;
    }

    // ── Fallback: per-expert execution (kept for reference) ──
    // Expert weight tensors are stored as single INT4-packed NXF chunks:
    //   w1: [num_experts, hidden_dim, expert_ffn_dim] — gate projection
    //   w2: [num_experts, expert_ffn_dim, hidden_dim] — down projection
    //   w3: [num_experts, hidden_dim, expert_ffn_dim] — up projection
    //
    // For expert E in INT4 packed format (2 elements per byte):
    //   w1/w3 byte offset = E * dim * expert_ffn / 2  (each slice is [dim, expert_ffn])
    //   w2 byte offset    = E * expert_ffn * dim / 2  (each slice is [expert_ffn, dim])

    size_t w13_slice_bytes = static_cast<size_t>(dim) * expert_ffn / 2;   // INT4 bytes per expert for w1/w3
    size_t w2_slice_bytes  = static_cast<size_t>(expert_ffn) * dim / 2;   // INT4 bytes per expert for w2

    std::vector<float> gate_buf(expert_ffn);
    std::vector<float> up_buf(expert_ffn);
    std::vector<float> expert_out(dim);

    for (int i = 0; i < num_active; i++) {
        int eid = expert_indices[i];
        float weight = expert_weights[i];

        // ── Compute byte offsets for expert E's slice in packed INT4 data ──
        size_t w13_byte_off = static_cast<size_t>(eid) * w13_slice_bytes;
        size_t w2_byte_off  = static_cast<size_t>(eid) * w2_slice_bytes;

        // Bounds check before accessing raw INT4 data
        if (w13_byte_off + w13_slice_bytes > lw.expert_w1_bytes ||
            w2_byte_off  + w2_slice_bytes  > lw.expert_w2_bytes ||
            w13_byte_off + w13_slice_bytes > lw.expert_w3_bytes) {
            fprintf(stderr, "[nexus] WARNING: expert %d slice out of bounds in layer %u, skipping\n",
                    eid, layer_idx);
            continue;
        }

        // Build RawWeight structs pointing to this expert's INT4 slices
        HybridLayerWeights::RawWeight w1_slice_raw;
        w1_slice_raw.data = lw.expert_w1_raw + w13_byte_off;
        w1_slice_raw.bytes = w13_slice_bytes;

        HybridLayerWeights::RawWeight w3_slice_raw;
        w3_slice_raw.data = lw.expert_w3_raw + w13_byte_off;
        w3_slice_raw.bytes = w13_slice_bytes;

        HybridLayerWeights::RawWeight w2_slice_raw;
        w2_slice_raw.data = lw.expert_w2_raw + w2_byte_off;
        w2_slice_raw.bytes = w2_slice_bytes;

        // SwiGLU: out = (silu(x @ W1) * (x @ W3)) @ W2
        // Gate projection: x [1, dim] @ w1 [dim, expert_ffn] -> gate_buf [1, expert_ffn]
        fused_gemm(x, nullptr, w1_slice_raw, gate_buf.data(), 1, expert_ffn, dim);
        // Up projection: x [1, dim] @ w3 [dim, expert_ffn] -> up_buf [1, expert_ffn]
        fused_gemm(x, nullptr, w3_slice_raw, up_buf.data(), 1, expert_ffn, dim);

        // SiLU on gate, elementwise multiply with up
        for (int j = 0; j < expert_ffn; j++) {
            float s = gate_buf[j] / (1.0f + expf(-gate_buf[j]));  // SiLU
            gate_buf[j] = s * up_buf[j];
        }

        // Down projection: gate_buf [1, expert_ffn] @ w2 [expert_ffn, dim] -> expert_out [1, dim]
        fused_gemm(gate_buf.data(), nullptr, w2_slice_raw, expert_out.data(), 1, dim, expert_ffn);

        // Weighted accumulation into output
        for (int j = 0; j < dim; j++) {
            out[j] += weight * expert_out[j];
        }
    }
}

// ─── Weight loading ────────────────────────────────────────────────────────

bool HybridModel::Impl::load_layer_weights(uint32_t layer_idx) {
    auto& lw = layer_weights[layer_idx];
    if (lw.loaded) return true;
    current_loading_layer = layer_idx;

    auto load = [&](const char* suffix, float*& ptr,
                    std::vector<int64_t>* shape = nullptr) -> bool {
        std::string name = layer_tensor_name(layer_idx, suffix);
        ptr = load_tensor(name.c_str(), shape);
        return ptr != nullptr;
    };

    // Load with raw INT4 pointer for fused GPU path
    auto load_raw = [&](const char* suffix, float*& ptr,
                        HybridLayerWeights::RawWeight& raw,
                        std::vector<int64_t>* shape = nullptr) -> bool {
        std::string name = layer_tensor_name(layer_idx, suffix);
        ptr = load_tensor_raw(name.c_str(), raw, shape);
        return ptr != nullptr;
    };

    // ── Shared norms ──
    load("attention_norm.weight", lw.attention_norm);
    load("post_attention_norm.weight", lw.post_attention_norm);

    // ── MoE weights ──
    // NOTE: Expert weights (w1/w2/w3) are HUGE (512 experts × 2048 × 512 each).
    // Loading all of them would require ~1.6 GB FP32 per layer. Instead, we load
    // only the gate weights here and defer expert loading to the MoE forward pass
    // where we know which experts are active.
    {
        std::vector<int64_t> gate_shape;
        load_raw("feed_forward.gate.weight", lw.moe_gate, lw.moe_gate_raw, &gate_shape);
        if (lw.moe_gate && gate_shape.size() >= 2) {
            int d0 = static_cast<int>(gate_shape[0]);
            int d1 = static_cast<int>(gate_shape[1]);
            lw.num_experts = (d1 > d0) ? d1 : d0;
        }
        // Get expert FFN dim from tensor shape metadata without loading data
        std::string w1_name = layer_tensor_name(layer_idx, "feed_forward.experts.w1.weight");
        const auto* w1_info = reader->get_tensor(w1_name);
        if (w1_info && w1_info->shape.size() >= 3) {
            if (lw.num_experts == 0) lw.num_experts = static_cast<int>(w1_info->shape[0]);
            lw.expert_ffn_dim = static_cast<int>(w1_info->shape[2]);
        } else if (w1_info && w1_info->shape.size() >= 2) {
            lw.expert_ffn_dim = static_cast<int>(w1_info->shape[1]);
        }
        if (lw.expert_ffn_dim == 0) lw.expert_ffn_dim = 512;  // Qwen3-Coder-Next default
        if (lw.num_experts == 0) lw.num_experts = 512;
    }
    // Expert w1/w2/w3: map the raw INT4 chunks (but do NOT dequantize the whole thing).
    // Per-expert slicing + dequant happens on-demand in moe_ffn().
    lw.expert_w1 = nullptr;  // FP32 full dequant not used
    lw.expert_w2 = nullptr;
    lw.expert_w3 = nullptr;
    {
        auto map_expert_raw = [&](const char* suffix, const uint8_t*& raw_ptr,
                                   size_t& raw_bytes) {
            std::string tname = layer_tensor_name(layer_idx, suffix);
            const auto* tinfo = reader->get_tensor(tname);
            if (!tinfo || tinfo->chunks.empty()) return;
            const auto& chunk = tinfo->chunks[0];
            const void* mapped = reader->map_chunk(chunk);
            if (mapped) {
                raw_ptr = static_cast<const uint8_t*>(mapped);
                raw_bytes = chunk.compressed_size;
            }
        };
        map_expert_raw("feed_forward.experts.w1.weight",
                       lw.expert_w1_raw, lw.expert_w1_bytes);
        map_expert_raw("feed_forward.experts.w2.weight",
                       lw.expert_w2_raw, lw.expert_w2_bytes);
        map_expert_raw("feed_forward.experts.w3.weight",
                       lw.expert_w3_raw, lw.expert_w3_bytes);
    }
    load("ffn_gate_inp_shexp.weight", lw.shared_expert_gate);

    // ── Type-specific weights ──
    if (lw.type == HybridLayerType::SSM_MoE) {
        // Fused QKV — use load_tensor_raw to get both FP32 + raw INT4 pointer
        std::vector<int64_t> qkv_shape;
        {
            std::string name = layer_tensor_name(layer_idx, "attn_qkv.weight");
            lw.attn_qkv = load_tensor_raw(name.c_str(), lw.attn_qkv_raw, &qkv_shape);
        }
        if (lw.attn_qkv && qkv_shape.size() >= 2) {
            lw.qkv_out_dim = static_cast<int>(qkv_shape[1]);
        }

        // Attn gate
        std::vector<int64_t> gate_shape;
        load_raw("attn_gate.weight", lw.attn_gate, lw.attn_gate_raw, &gate_shape);
        if (lw.attn_gate && gate_shape.size() >= 2) {
            lw.gate_out_dim = static_cast<int>(gate_shape[1]);
        }

        // SSM output projection
        std::vector<int64_t> ssm_out_shape;
        load_raw("ssm_out.weight", lw.ssm_out, lw.ssm_out_raw, &ssm_out_shape);
        if (lw.ssm_out && ssm_out_shape.size() >= 2) {
            lw.ssm_out_dim = static_cast<int>(ssm_out_shape[0]);
        }

        // SSM tensors (loaded for completeness, not used in simplified mode)
        load("ssm_a", lw.ssm_a);
        load("ssm_ba.weight", lw.ssm_ba);
        load("ssm_conv1d.weight", lw.ssm_conv1d);
        load("ssm_dt.bias", lw.ssm_dt_bias);
        load("ssm_norm.weight", lw.ssm_norm);

        // Derive K/V dimensions for the fused QKV split.
        // Strategy: look at a Type B layer's K/V shapes for reference, or use
        // manifest config. We'll use manifest.num_kv_heads * manifest.head_dim.
        lw.k_out_dim = manifest.num_kv_heads * manifest.head_dim;
        lw.v_out_dim = lw.k_out_dim;
        // Validate the split is possible
        if (lw.qkv_out_dim > 0 && lw.k_out_dim > 0 && lw.v_out_dim > 0) {
            int q_dim = lw.qkv_out_dim - lw.k_out_dim - lw.v_out_dim;
            if (q_dim <= 0) {
                // The default K/V dim doesn't work for the split.
                // Try using the actual K projection shape from a Type B layer.
                // Scan for any attention layer's K shape.
                for (uint32_t j = 0; j < manifest.num_layers; j++) {
                    std::string kn = layer_tensor_name(j, "attention.wk.weight");
                    const auto* kinfo = reader->get_tensor(kn);
                    if (kinfo && kinfo->shape.size() >= 2) {
                        lw.k_out_dim = static_cast<int>(kinfo->shape[1]);
                        lw.v_out_dim = lw.k_out_dim;
                        break;
                    }
                }
                // Re-check
                q_dim = lw.qkv_out_dim - lw.k_out_dim - lw.v_out_dim;
                if (q_dim <= 0) {
                    // Still can't split — treat entire output as Q
                    lw.k_out_dim = 0;
                    lw.v_out_dim = 0;
                }
            }
            lw.q_out_dim = lw.qkv_out_dim - lw.k_out_dim - lw.v_out_dim;
        }

    } else if (lw.type == HybridLayerType::Attention_MoE) {
        // Separate Q/K/V/O projections — with raw INT4 for GPU fused path
        std::vector<int64_t> q_shape, k_shape, v_shape;
        load_raw("attention.wq.weight", lw.wq, lw.wq_raw, &q_shape);
        load_raw("attention.wk.weight", lw.wk, lw.wk_raw, &k_shape);
        load_raw("attention.wv.weight", lw.wv, lw.wv_raw, &v_shape);
        load_raw("attention.wo.weight", lw.wo, lw.wo_raw);

        if (lw.wq && q_shape.size() >= 2) lw.q_out_dim = static_cast<int>(q_shape[1]);
        if (lw.wk && k_shape.size() >= 2) lw.k_out_dim = static_cast<int>(k_shape[1]);
        if (lw.wv && v_shape.size() >= 2) lw.v_out_dim = static_cast<int>(v_shape[1]);

        // Q/K norms
        load("attn_q_norm.weight", lw.q_norm);
        load("attn_k_norm.weight", lw.k_norm);
    }

    lw.loaded = true;
    return true;
}

void HybridModel::Impl::evict_layer_weights(uint32_t layer_idx) {
    // In resident mode, weights stay loaded permanently — never evict.
    if (resident_mode) return;

    auto& lw = layer_weights[layer_idx];
    // Mark this layer's dequant buffers as available for reuse.
    // Don't actually free (munmap) — just clear the tracking so
    // the next layer's load_tensor can allocate fresh buffers.
    // The OS will reclaim physical pages via madvise if needed.
    auto it = layer_dequant_buffers.find(layer_idx);
    if (it != layer_dequant_buffers.end()) {
        for (auto& db : it->second) {
            if (db.ptr) {
                // Tell OS these pages are no longer needed (frees physical RAM
                // but keeps the virtual mapping so we don't crash on stale ptrs)
                madvise(db.ptr, db.size, MADV_DONTNEED);
            }
        }
        layer_dequant_buffers.erase(it);
    }
    lw.attention_norm = nullptr;
    lw.post_attention_norm = nullptr;
    lw.attn_gate = nullptr;
    lw.attn_qkv = nullptr;
    lw.ssm_out = nullptr;
    lw.ssm_a = nullptr;
    lw.ssm_ba = nullptr;
    lw.ssm_conv1d = nullptr;
    lw.ssm_dt_bias = nullptr;
    lw.ssm_norm = nullptr;
    lw.wq = nullptr;
    lw.wk = nullptr;
    lw.wv = nullptr;
    lw.wo = nullptr;
    lw.q_norm = nullptr;
    lw.k_norm = nullptr;
    lw.moe_gate = nullptr;
    lw.expert_w1 = nullptr;
    lw.expert_w2 = nullptr;
    lw.expert_w3 = nullptr;
    lw.expert_w1_raw = nullptr;
    lw.expert_w2_raw = nullptr;
    lw.expert_w3_raw = nullptr;
    lw.expert_w1_bytes = 0;
    lw.expert_w2_bytes = 0;
    lw.expert_w3_bytes = 0;
    lw.shared_expert_gate = nullptr;
    lw.loaded = false;
}

bool HybridModel::Impl::load_embedding_weights() {
    // Use a special layer ID (UINT32_MAX) for embeddings so they never get
    // evicted by layer-level eviction. These stay resident for the model lifetime.
    current_loading_layer = UINT32_MAX;
    token_embeddings = load_tensor("tok_embeddings.weight");
    output_norm = load_tensor("norm.weight");
    output_weight = load_tensor_raw("output.weight", output_weight_raw);
    current_loading_layer = 0;
    return token_embeddings != nullptr;
}

// ─── Sampling ──────────────────────────────────────────────────────────────

int32_t HybridModel::Impl::sample_token(const float* logits_ptr,
                                          const SamplingParams& params) {
    int vocab = manifest.vocab_size;

    if (params.temperature <= 0.0f) {
        // Greedy: argmax
        int best = 0;
        float best_val = logits_ptr[0];
        for (int i = 1; i < vocab; i++) {
            if (logits_ptr[i] > best_val) {
                best_val = logits_ptr[i];
                best = i;
            }
        }
        return best;
    }

    // Temperature scaling
    std::vector<float> probs(vocab);
    float max_logit = *std::max_element(logits_ptr, logits_ptr + vocab);
    float sum = 0.0f;
    for (int i = 0; i < vocab; i++) {
        probs[i] = expf((logits_ptr[i] - max_logit) / params.temperature);
        sum += probs[i];
    }
    for (int i = 0; i < vocab; i++) probs[i] /= sum;

    // Top-k filtering
    if (params.top_k > 0 && params.top_k < vocab) {
        std::vector<int> indices(vocab);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + params.top_k,
                          indices.end(),
                          [&](int a, int b) { return probs[a] > probs[b]; });

        std::vector<float> filtered(vocab, 0.0f);
        for (int i = 0; i < params.top_k; i++) {
            filtered[indices[i]] = probs[indices[i]];
        }
        probs = std::move(filtered);
    }

    // Top-p (nucleus) filtering
    if (params.top_p < 1.0f) {
        std::vector<int> indices(vocab);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&](int a, int b) { return probs[a] > probs[b]; });

        float cumsum = 0.0f;
        for (int i = 0; i < vocab; i++) {
            cumsum += probs[indices[i]];
            if (cumsum > params.top_p) {
                for (int j = i + 1; j < vocab; j++) {
                    probs[indices[j]] = 0.0f;
                }
                break;
            }
        }
    }

    // Renormalize and sample
    sum = 0.0f;
    for (int i = 0; i < vocab; i++) sum += probs[i];
    if (sum <= 0.0f) return 0;

    std::mt19937 rng(params.seed ? params.seed : std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, sum);
    float r = dist(rng);

    float cumsum = 0.0f;
    for (int i = 0; i < vocab; i++) {
        cumsum += probs[i];
        if (cumsum >= r) return i;
    }
    return vocab - 1;
}

}  // namespace nexus::model
