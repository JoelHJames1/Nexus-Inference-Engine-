/// NEXUS Metal Shader — Fused Multi-Head Attention (Decode)
///
/// Processes ALL query heads in ONE dispatch for single-token decode.
/// Replaces 32 separate per-head dispatches with 1.
///
/// Each threadgroup handles ONE query head. All threadgroups dispatch
/// together in a single Metal command. This amortizes command buffer
/// overhead across all heads.
///
/// For GQA: multiple query heads share the same KV head.
///   heads_per_group = num_heads / num_kv_heads
///   kv_head_for_query_h = h / heads_per_group
///
/// KV cache is FP16 stored as uint16. Converted to float on the fly.
/// This halves memory bandwidth for KV reads.
///
/// Algorithm per threadgroup (one query head):
///   1. Load Q_head from Q buffer
///   2. For each cached token t in [0, seq_len):
///      a. Load K[t] from FP16 cache, convert to float
///      b. Compute dot(Q_head, K[t]) * scale
///      c. Online softmax: track max and sum
///   3. Second pass: compute weighted sum of V[t]
///   4. Write output[h * out_head_dim : (h+1) * out_head_dim]

#include <metal_stdlib>
using namespace metal;

struct FusedAttentionParams {
    uint num_heads;        // Total query heads (e.g., 32)
    uint num_kv_heads;     // KV heads for GQA (e.g., 32 for MHA, 2 for GQA)
    uint head_dim_q;       // Q head dimension (e.g., 256)
    uint head_dim_kv;      // K/V head dimension (e.g., 256)
    uint dot_dim;          // min(head_dim_q, head_dim_kv) for dot product
    uint out_head_dim;     // Output dimension per head (= head_dim_kv)
    uint seq_len;          // Current sequence length
    uint kv_dim;           // Full KV dimension (kv_heads * head_dim_kv)
    float scale;           // 1/sqrt(dot_dim)
};

/// Convert FP16 bits (stored as uint16) to float
inline float fp16_to_float(uint16_t h) {
    return float(as_type<half>(h));
}

kernel void fused_multihead_attention_decode(
    device const float*    Q           [[buffer(0)]],  // [num_heads * head_dim_q]
    device const uint16_t* K_cache     [[buffer(1)]],  // [seq_len * kv_dim] FP16
    device const uint16_t* V_cache     [[buffer(2)]],  // [seq_len * kv_dim] FP16
    device float*          output      [[buffer(3)]],  // [num_heads * out_head_dim]
    constant FusedAttentionParams& params [[buffer(4)]],
    uint  head_id    [[threadgroup_position_in_grid]],   // Which query head
    uint  tid        [[thread_index_in_threadgroup]],
    uint  tg_size    [[threads_per_threadgroup]])
{
    if (head_id >= params.num_heads) return;

    uint heads_per_group = params.num_heads / params.num_kv_heads;
    uint kv_head = head_id / heads_per_group;
    uint dot_dim = params.dot_dim;
    uint out_hd = params.out_head_dim;
    uint seq_len = params.seq_len;
    float scale = params.scale;

    // Pointer to this query head
    device const float* q_head = Q + head_id * params.head_dim_q;

    // ── Pass 1: Compute attention scores with online softmax ──
    // Each thread handles a subset of the sequence positions
    float thread_max = -1e30f;
    float thread_sum = 0.0f;

    // Temporary score storage in threadgroup memory
    // For short sequences (< 4096), we can store all scores
    threadgroup float scores[4096];

    for (uint t = tid; t < seq_len; t += tg_size) {
        // Load K[t] for this KV head from FP16 cache
        device const uint16_t* k_t = K_cache + t * params.kv_dim + kv_head * params.head_dim_kv;

        // Dot product Q · K
        float dot = 0.0f;
        for (uint d = 0; d < dot_dim; d++) {
            dot += q_head[d] * fp16_to_float(k_t[d]);
        }
        dot *= scale;
        scores[t] = dot;

        if (dot > thread_max) thread_max = dot;
    }

    // Reduce max across threads
    threadgroup float shared_max[256];
    shared_max[tid] = thread_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_max = shared_max[0];

    // Compute exp and sum
    for (uint t = tid; t < seq_len; t += tg_size) {
        scores[t] = exp(scores[t] - global_max);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce sum
    float local_sum = 0.0f;
    for (uint t = tid; t < seq_len; t += tg_size) {
        local_sum += scores[t];
    }
    threadgroup float shared_sum[256];
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total_sum = shared_sum[0];

    // Normalize scores
    if (total_sum > 0.0f) {
        float inv_sum = 1.0f / total_sum;
        for (uint t = tid; t < seq_len; t += tg_size) {
            scores[t] *= inv_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Pass 2: Weighted sum of V ──
    // Each thread computes a subset of the output dimensions
    device float* out_head = output + head_id * out_hd;

    for (uint d = tid; d < out_hd; d += tg_size) {
        float acc = 0.0f;
        for (uint t = 0; t < seq_len; t++) {
            device const uint16_t* v_t = V_cache + t * params.kv_dim + kv_head * params.head_dim_kv;
            acc += scores[t] * fp16_to_float(v_t[d]);
        }
        out_head[d] = acc;
    }
}
