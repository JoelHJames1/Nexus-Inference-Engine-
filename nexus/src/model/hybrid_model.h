#pragma once
/// NEXUS HybridModel — Qwen3-Coder-Next hybrid SSM+Attention architecture.
///
/// Supports the "12 x (3 x (Gated DeltaNet -> MoE) -> 1 x (Gated Attention -> MoE))"
/// layout with 48 layers total. Two layer types:
///
///   Type A (SSM+MoE): layers 0,1,2, 4,5,6, 8,9,10, ... (3 of every 4)
///     - Uses fused QKV projection + gated DeltaNet (simplified as linear attention)
///     - MoE FFN with 512 experts, top-10 routing
///
///   Type B (Attention+MoE): layers 3,7,11,15,...,47 (every 4th)
///     - Standard GQA with separate Q/K/V projections and Q/K RMSNorm
///     - MoE FFN with same expert configuration

#include "core/config.h"
#include "format/nxf.h"
#include "model/moe_router.h"
#include <memory>
#include <vector>

namespace nexus {

class MemoryManager;

namespace model {

/// Layer type classification for the hybrid architecture.
enum class HybridLayerType {
    SSM_MoE,         // Type A: Gated DeltaNet + MoE (simplified as linear attention)
    Attention_MoE,   // Type B: Full GQA attention + MoE
    Unknown
};

/// Weights for a hybrid layer — superset of both Type A and Type B tensors.
/// Only the relevant pointers will be non-null for each layer type.
struct HybridLayerWeights {
    HybridLayerType type = HybridLayerType::Unknown;

    // ── Shared: pre/post norms ────────────────────────────────────────────
    float* attention_norm = nullptr;       // [hidden_dim] — pre-attention/SSM RMSNorm
    float* post_attention_norm = nullptr;  // [hidden_dim] — post-attention RMSNorm before MoE

    // ── Type A (SSM+MoE): fused QKV + SSM tensors ────────────────────────
    float* attn_gate = nullptr;            // [hidden_dim, gate_dim] — gating for DeltaNet
    float* attn_qkv = nullptr;            // [hidden_dim, qkv_dim] — fused Q/K/V projection
    float* ssm_out = nullptr;             // [out_dim, hidden_dim] — SSM output projection
    // SSM-specific tensors (loaded but not used in simplified mode):
    float* ssm_a = nullptr;               // [num_ssm_heads] — DeltaNet decay
    float* ssm_ba = nullptr;              // [hidden_dim, ba_dim] — B*A projection
    float* ssm_conv1d = nullptr;          // [conv_width, conv_channels] — 1D conv
    float* ssm_dt_bias = nullptr;         // [num_ssm_heads] — timestep bias
    float* ssm_norm = nullptr;            // [ssm_norm_dim] — SSM output norm

    // ── Type B (Attention+MoE): separate Q/K/V ───────────────────────────
    float* wq = nullptr;                  // [hidden_dim, q_dim] — Query projection
    float* wk = nullptr;                  // [hidden_dim, k_dim] — Key projection
    float* wv = nullptr;                  // [hidden_dim, v_dim] — Value projection
    float* wo = nullptr;                  // [q_dim, hidden_dim] — Output projection
    float* q_norm = nullptr;              // [q_norm_dim] — Q RMSNorm
    float* k_norm = nullptr;              // [k_norm_dim] — K RMSNorm

    // ── MoE FFN (shared by both types) ───────────────────────────────────
    float* moe_gate = nullptr;            // [hidden_dim, num_experts] — router gate
    float* expert_w1 = nullptr;           // [num_experts, hidden_dim, expert_ffn_dim] — gate proj
    float* expert_w2 = nullptr;           // [num_experts, expert_ffn_dim, hidden_dim] — down proj
    float* expert_w3 = nullptr;           // [num_experts, hidden_dim, expert_ffn_dim] — up proj
    float* shared_expert_gate = nullptr;  // [hidden_dim] — shared expert gating

    // Raw mapped INT4 base pointers for on-demand expert slicing.
    // These point to the full [num_experts, ...] packed INT4 data in the NXF.
    // Per-expert slicing and dequant happens in moe_ffn().
    const uint8_t* expert_w1_raw = nullptr;  // mapped INT4 for w1 [num_experts, hidden_dim, ffn_dim]
    const uint8_t* expert_w2_raw = nullptr;  // mapped INT4 for w2 [num_experts, ffn_dim, hidden_dim]
    const uint8_t* expert_w3_raw = nullptr;  // mapped INT4 for w3 [num_experts, hidden_dim, ffn_dim]
    size_t expert_w1_bytes = 0;              // total mapped size for bounds checking
    size_t expert_w2_bytes = 0;
    size_t expert_w3_bytes = 0;

    // Raw INT4 data for main weight tensors (for fused GPU GEMV path).
    // When set, gemm_int4() can skip CPU dequant entirely.
    struct RawWeight {
        const void* data = nullptr;
        size_t bytes = 0;
    };
    RawWeight attn_qkv_raw;              // Raw INT4 for fused QKV
    RawWeight attn_gate_raw;             // Raw INT4 for attn gate
    RawWeight ssm_out_raw;               // Raw INT4 for SSM output projection
    RawWeight wq_raw, wk_raw, wv_raw, wo_raw;  // Raw INT4 for separate attention
    RawWeight moe_gate_raw;              // Raw INT4 for router gate
    RawWeight output_weight_raw;         // Raw INT4 for final output projection

    // ── Dense FFN (for non-MoE models like Gemma, LLaMA) ──────────────────
    float* ffn_w1 = nullptr;              // [hidden_dim, ffn_dim] — gate proj
    float* ffn_w2 = nullptr;              // [ffn_dim, hidden_dim] — down proj
    float* ffn_w3 = nullptr;              // [hidden_dim, ffn_dim] — up proj
    float* ffn_norm = nullptr;            // [hidden_dim] — pre-FFN norm
    RawWeight ffn_w1_raw, ffn_w2_raw, ffn_w3_raw;
    int ffn_dim = 0;

    // Shape metadata read from tensor info (stored per-layer for flexibility)
    int qkv_out_dim = 0;                 // Output dimension of fused QKV
    int q_out_dim = 0;                   // Output dimension of separate Q
    int k_out_dim = 0;                   // Output dimension of separate K
    int v_out_dim = 0;                   // Output dimension of separate V
    int gate_out_dim = 0;               // Output dimension of attn_gate
    int ssm_out_dim = 0;               // First dimension of ssm_out weight
    int expert_ffn_dim = 0;            // FFN intermediate dim per expert
    int num_experts = 0;               // Total experts in this layer

    bool loaded = false;
};

/// KV cache entry for attention layers (Type B) and simplified attention (Type A).
struct HybridKVCache {
    float* keys = nullptr;     // [max_seq, kv_dim]
    float* values = nullptr;   // [max_seq, kv_dim]
    int    seq_len = 0;
    int    kv_dim = 0;         // Actual KV dimension for this layer
};

/// Hybrid model supporting Qwen3-Coder-Next's mixed SSM+Attention architecture.
class HybridModel {
public:
    ~HybridModel();

    /// Create from NXF model file. Detects layer types from available tensors.
    static std::unique_ptr<HybridModel> create(
        const format::ModelManifest& manifest,
        format::NXFReader& reader,
        MemoryManager& memory
    );

    /// Prefill: process all prompt tokens sequentially through all layers.
    void prefill(const std::vector<int32_t>& tokens);

    /// Decode one step: compute next token logits and sample.
    int32_t decode_step(const SamplingParams& params);

    /// Reset KV cache for new conversation.
    void reset_kv_cache();

    /// Current sequence length.
    int seq_len() const;

private:
    HybridModel() = default;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace model
}  // namespace nexus
