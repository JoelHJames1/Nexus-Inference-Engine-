/// NEXUS Metal Shader — Fused FFN (RMSNorm + W1 + W3 + SiLU×mul + W2 + Residual)
///
/// Replaces 6 separate dispatches with ONE. The entire feed-forward network
/// block executes in a single GPU dispatch:
///   1. RMSNorm on input hidden state
///   2. W1 GEMV (gate projection, INT4)
///   3. W3 GEMV (up projection, INT4)
///   4. SiLU(gate) * up (fused activation)
///   5. W2 GEMV (down projection, INT4)
///   6. Residual add (output += input)
///
/// Design: Tiled across FFN dimension. Each threadgroup handles a TILE of
/// the FFN intermediate dimension. W2 projection accumulates across tiles
/// using atomic adds to the output buffer.
///
/// INT4 dequant inline: (nibble - 8) * 0.125 (no separate dequant dispatch)

#include <metal_stdlib>
using namespace metal;

struct FusedFFNParams {
    uint dim;           // Hidden dimension (e.g., 5376)
    uint ffn_dim;       // FFN intermediate dimension (e.g., 21504)
    float rms_eps;      // RMSNorm epsilon
};

/// Inline INT4 dequant: read packed byte, return two floats
inline float2 dequant_int4_pair(uint8_t packed) {
    float lo = (float(packed & 0x0F) - 8.0f) * 0.125f;
    float hi = (float(packed >> 4) - 8.0f) * 0.125f;
    return float2(lo, hi);
}

/// Fused FFN kernel.
/// Grid: ceil(ffn_dim / TILE_SIZE) threadgroups × 1 × 1
/// Each threadgroup processes TILE_SIZE elements of the FFN intermediate.
/// The W2 down-projection accumulates to output via atomic float add.
constant uint TILE_SIZE = 256;  // FFN elements per threadgroup

kernel void fused_ffn_int4(
    device float*         hidden     [[buffer(0)]],  // [dim] in+out (residual added)
    device const float*   norm_w     [[buffer(1)]],  // [dim] RMSNorm weight
    device const uint8_t* w1_int4    [[buffer(2)]],  // [dim/2, ffn_dim] packed INT4
    device const uint8_t* w3_int4    [[buffer(3)]],  // [dim/2, ffn_dim] packed INT4
    device const uint8_t* w2_int4    [[buffer(4)]],  // [ffn_dim/2, dim] packed INT4
    device float*         output     [[buffer(5)]],  // [dim] accumulated output
    constant FusedFFNParams& params  [[buffer(6)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint dim = params.dim;
    uint ffn_dim = params.ffn_dim;
    float eps = params.rms_eps;

    // Which FFN elements this threadgroup handles
    uint ffn_start = tg_id * TILE_SIZE;
    uint ffn_end = min(ffn_start + TILE_SIZE, ffn_dim);
    uint tile_size = ffn_end - ffn_start;

    // ── Step 1: Cooperative RMSNorm ──────────────────────────────────────
    // All threads in the threadgroup cooperate to compute the norm.
    // This is done ONCE and shared across W1/W3 computations.
    threadgroup float shared_rms[1];
    threadgroup float norm_cache[1024];  // Cache normalized input (up to 1024 elements)

    // Compute sum of squares
    float local_ss = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = hidden[i];
        local_ss += val * val;
    }

    // Reduce sum of squares across threadgroup
    threadgroup float ss_scratch[256];
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

    // ── Step 2 & 3: W1 GEMV + W3 GEMV (gate + up projections) ──────────
    // Each thread computes one or more FFN elements for both W1 and W3.
    // The normalized input is computed on-the-fly (no intermediate buffer).

    // Local storage for gate/up results in this tile
    threadgroup float gate_tile[256];  // SiLU-activated gate values
    threadgroup float up_tile[256];

    for (uint f = tid; f < tile_size; f += tg_size) {
        uint ffn_idx = ffn_start + f;
        float gate_acc = 0.0f;
        float up_acc = 0.0f;

        // Dot product: normalized_input[k] * w1[k, ffn_idx] and w3[k, ffn_idx]
        for (uint k = 0; k < dim; k += 2) {
            // Normalize on-the-fly
            float x0 = hidden[k] * rms_inv * norm_w[k];
            float x1 = (k + 1 < dim) ? hidden[k+1] * rms_inv * norm_w[k+1] : 0.0f;

            // W1 INT4 dequant + multiply
            uint w1_byte = (k / 2) * ffn_dim + ffn_idx;
            uint8_t w1_packed = w1_int4[w1_byte];
            float2 w1_vals = dequant_int4_pair(w1_packed);
            gate_acc += x0 * w1_vals[0] + x1 * w1_vals[1];

            // W3 INT4 dequant + multiply
            uint w3_byte = (k / 2) * ffn_dim + ffn_idx;
            uint8_t w3_packed = w3_int4[w3_byte];
            float2 w3_vals = dequant_int4_pair(w3_packed);
            up_acc += x0 * w3_vals[0] + x1 * w3_vals[1];
        }

        // ── Step 4: SiLU(gate) * up (in register, no memory write) ──
        float silu_gate = gate_acc / (1.0f + exp(-gate_acc));
        gate_tile[f] = silu_gate * up_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Step 5: W2 GEMV (down projection, accumulate to output) ──────────
    // Each thread computes contribution of this FFN tile to output[d].
    // Use atomic add since multiple threadgroups contribute to same output.
    for (uint d = tid; d < dim; d += tg_size) {
        float acc = 0.0f;
        for (uint f = 0; f < tile_size; f += 2) {
            uint ffn_idx = ffn_start + f;
            uint w2_byte = (ffn_idx / 2) * dim + d;
            uint8_t w2_packed = w2_int4[w2_byte];
            float2 w2_vals = dequant_int4_pair(w2_packed);
            acc += gate_tile[f] * w2_vals[0];
            if (f + 1 < tile_size) {
                acc += gate_tile[f + 1] * w2_vals[1];
            }
        }

        // Atomic accumulate (multiple tiles write to same output element)
        // Also add residual (hidden[d] is the pre-norm input = residual)
        if (tg_id == 0) {
            // First tile: initialize output with residual
            output[d] = hidden[d] + acc;
        } else {
            // Subsequent tiles: atomic add
            atomic_fetch_add_explicit((device atomic_float*)&output[d], acc,
                                     memory_order_relaxed);
        }
    }
}
