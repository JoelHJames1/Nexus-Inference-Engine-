/// NEXUS Metal Shader — Fused QKV Projection (RMSNorm + Q + K + V in ONE dispatch)
///
/// Replaces 4 separate dispatches (RMSNorm, Q GEMV, K GEMV, V GEMV) with ONE.
/// The normalized input is computed once and shared across all three projections.
///
/// Design: Each threadgroup computes a tile of output elements across Q, K, and V.
/// The RMSNorm reduction is cooperative across the threadgroup, then each thread
/// computes its assigned output elements for all three projections.
///
/// For Gemma 31B: dim=5376, q_dim=8192, k_dim=5376, v_dim=5376

#include <metal_stdlib>
using namespace metal;

struct FusedQKVParams {
    uint dim;       // Input hidden dimension (e.g., 5376)
    uint q_dim;     // Q output dimension (e.g., 8192)
    uint k_dim;     // K output dimension (e.g., 5376)
    uint v_dim;     // V output dimension (e.g., 5376)
    float rms_eps;  // RMSNorm epsilon
};

inline float2 dequant_int4_pair(uint8_t packed) {
    float lo = (float(packed & 0x0F) - 8.0f) * 0.125f;
    float hi = (float(packed >> 4) - 8.0f) * 0.125f;
    return float2(lo, hi);
}

/// Fused QKV projection kernel.
/// Grid: max(q_dim, k_dim, v_dim) threads across all outputs.
/// Each thread computes one output element for whichever projection(s)
/// need that index.
kernel void fused_qkv_int4(
    device const float*   hidden     [[buffer(0)]],  // [dim] input
    device const float*   norm_w     [[buffer(1)]],  // [dim] RMSNorm weight
    device const uint8_t* wq_int4    [[buffer(2)]],  // [dim/2, q_dim] packed INT4
    device const uint8_t* wk_int4    [[buffer(3)]],  // [dim/2, k_dim] packed INT4
    device const uint8_t* wv_int4    [[buffer(4)]],  // [dim/2, v_dim] packed INT4
    device float*         q_out      [[buffer(5)]],  // [q_dim]
    device float*         k_out      [[buffer(6)]],  // [k_dim]
    device float*         v_out      [[buffer(7)]],  // [v_dim]
    constant FusedQKVParams& params  [[buffer(8)]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid     [[thread_position_in_grid]])
{
    uint dim = params.dim;
    float eps = params.rms_eps;

    // ── Step 1: Cooperative RMSNorm ──────────────────────────────────────
    // All threads compute partial sum of squares, reduce, share result.
    threadgroup float shared_rms[1];

    float local_ss = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = hidden[i];
        local_ss += val * val;
    }

    threadgroup float ss_scratch[1024];
    ss_scratch[tid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) ss_scratch[tid] += ss_scratch[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        shared_rms[0] = rsqrt(ss_scratch[0] / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms_inv = shared_rms[0];

    // ── Step 2: Q/K/V GEMV with shared normalized input ─────────────────
    // Each thread handles one output index. If that index is within Q/K/V
    // range, compute the corresponding dot product.

    uint col = gid;  // Output column index

    // Q projection
    if (col < params.q_dim) {
        float acc = 0.0f;
        for (uint k = 0; k < dim; k += 2) {
            float x0 = hidden[k] * rms_inv * norm_w[k];
            float x1 = (k + 1 < dim) ? hidden[k+1] * rms_inv * norm_w[k+1] : 0.0f;
            uint byte_idx = (k / 2) * params.q_dim + col;
            float2 w = dequant_int4_pair(wq_int4[byte_idx]);
            acc += x0 * w[0] + x1 * w[1];
        }
        q_out[col] = acc;
    }

    // K projection (may overlap with Q indices for small models)
    if (col < params.k_dim) {
        float acc = 0.0f;
        for (uint k = 0; k < dim; k += 2) {
            float x0 = hidden[k] * rms_inv * norm_w[k];
            float x1 = (k + 1 < dim) ? hidden[k+1] * rms_inv * norm_w[k+1] : 0.0f;
            uint byte_idx = (k / 2) * params.k_dim + col;
            float2 w = dequant_int4_pair(wk_int4[byte_idx]);
            acc += x0 * w[0] + x1 * w[1];
        }
        k_out[col] = acc;
    }

    // V projection
    if (col < params.v_dim) {
        float acc = 0.0f;
        for (uint k = 0; k < dim; k += 2) {
            float x0 = hidden[k] * rms_inv * norm_w[k];
            float x1 = (k + 1 < dim) ? hidden[k+1] * rms_inv * norm_w[k+1] : 0.0f;
            uint byte_idx = (k / 2) * params.v_dim + col;
            float2 w = dequant_int4_pair(wv_int4[byte_idx]);
            acc += x0 * w[0] + x1 * w[1];
        }
        v_out[col] = acc;
    }
}
