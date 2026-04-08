/// NEXUS Metal Shader — Batched Expert GEMV (Fused MoE FFN)
///
/// Computes ALL active expert GEMVs in ONE dispatch, eliminating the
/// per-expert dispatch overhead that dominates MoE inference latency.
///
/// For an 80B MoE with 512 experts and 10 active per token, this replaces
/// ~30 separate GPU dispatches (w1/w3/w2 per expert) with a SINGLE dispatch.
///
/// Each threadgroup handles one active expert end-to-end:
///   1. GEMV: gate_buf = x @ w1_expert  (hidden_dim -> ffn_dim)
///   2. GEMV: up_buf   = x @ w3_expert  (hidden_dim -> ffn_dim)
///   3. Fused SiLU + element-wise multiply: inter = silu(gate_buf) * up_buf
///   4. GEMV: out_buf  = inter @ w2_expert (ffn_dim -> hidden_dim)
///   5. Atomic accumulate: output += gate_weight * out_buf
///
/// Weight layout: INT4 uniform dequant, packed as (nibble - 8) * 0.125
/// All 512 expert weight matrices are packed contiguously in one buffer each.
///
/// Dispatch: [num_active, 1, 1] threadgroups, 256 threads per threadgroup.

#include <metal_stdlib>
using namespace metal;

// ─── SiLU activation ───────────────────────────────────────────────────────
inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

// ─── INT4 uniform dequantization ───────────────────────────────────────────
// Matches NXF INT4 encoding: (nibble - 8) * 0.125
inline float dequant_int4(uint8_t nibble) {
    return (float(nibble & 0x0F) - 8.0f) * 0.125f;
}

// ─── Parameters ────────────────────────────────────────────────────────────
struct BatchedExpertParams {
    uint hidden_dim;       // 2048  — model hidden dimension
    uint expert_ffn_dim;   // 512   — per-expert FFN intermediate dimension
    uint num_active;       // 10    — number of active experts this token
    uint num_experts;      // 512   — total experts (for offset calculation)
};

// ─── Constants ─────────────────────────────────────────────────────────────
constant uint THREADS_PER_GROUP = 256;

// ─── Kernel ────────────────────────────────────────────────────────────────
kernel void batched_expert_gemv(
    device const float*    activations       [[buffer(0)]],  // [hidden_dim]
    device const uint8_t*  expert_w1         [[buffer(1)]],  // All experts w1 packed INT4
    device const uint8_t*  expert_w3         [[buffer(2)]],  // All experts w3 packed INT4
    device const uint8_t*  expert_w2         [[buffer(3)]],  // All experts w2 packed INT4
    device const uint*     expert_ids        [[buffer(4)]],  // [num_active]
    device const float*    expert_gate_wts   [[buffer(5)]],  // [num_active]
    device float*          output            [[buffer(6)]],  // [hidden_dim] accumulated
    constant BatchedExpertParams& params     [[buffer(7)]],
    uint  tg_id    [[threadgroup_position_in_grid]],        // Which active expert
    uint  tid      [[thread_index_in_threadgroup]])
{
    // ── Threadgroup memory for intermediate FFN buffers ─────────────────
    // Max ffn_dim we support in threadgroup memory.
    // 2 * 4096 floats = 32 KB — within Apple Silicon 32 KB threadgroup limit.
    // For ffn_dim=512 this uses only 2 * 512 * 4 = 4 KB.
    threadgroup float gate_buf[4096];   // Result of x @ w1 then silu(gate_buf) * up_buf
    threadgroup float up_buf[4096];     // Result of x @ w3

    // ── Identify this expert ───────────────────────────────────────────
    if (tg_id >= params.num_active) return;

    uint expert_id   = expert_ids[tg_id];
    float gate_weight = expert_gate_wts[tg_id];

    uint H = params.hidden_dim;      // e.g. 2048
    uint F = params.expert_ffn_dim;   // e.g. 512

    // Byte offset into packed INT4 weight buffers for this expert.
    // w1, w3 shape per expert: [H, F] stored as [H/2, F] packed bytes (2 H values per byte)
    // Actually: [H * F / 2] bytes per expert, layout: row k contributes k*F/2 bytes
    // Layout: weights stored as [K/2, N] where K=hidden_dim, N=ffn_dim
    // byte_idx = (k/2) * N + col, matching gemv_int4_uniform.metal
    uint expert_w1w3_offset = expert_id * (H * F / 2);  // byte offset for w1/w3
    // w2 shape per expert: [F, H] stored as [F/2, H] packed bytes
    uint expert_w2_offset   = expert_id * (F * H / 2);  // byte offset for w2

    // ════════════════════════════════════════════════════════════════════
    // STEP 1 & 2: Compute gate_buf = x @ w1 and up_buf = x @ w3
    // Each thread computes a subset of the FFN output columns.
    // ════════════════════════════════════════════════════════════════════
    for (uint col = tid; col < F; col += THREADS_PER_GROUP) {
        float acc_w1 = 0.0f;
        float acc_w3 = 0.0f;

        for (uint k = 0; k < H; k += 2) {
            float a0 = activations[k];
            float a1 = (k + 1 < H) ? activations[k + 1] : 0.0f;

            uint byte_idx = expert_w1w3_offset + (k / 2) * F + col;

            // w1 dequant
            uint8_t packed_w1 = expert_w1[byte_idx];
            float w1_lo = dequant_int4(packed_w1);
            float w1_hi = dequant_int4(packed_w1 >> 4);
            acc_w1 += a0 * w1_lo + a1 * w1_hi;

            // w3 dequant
            uint8_t packed_w3 = expert_w3[byte_idx];
            float w3_lo = dequant_int4(packed_w3);
            float w3_hi = dequant_int4(packed_w3 >> 4);
            acc_w3 += a0 * w3_lo + a1 * w3_hi;
        }

        gate_buf[col] = acc_w1;
        up_buf[col]   = acc_w3;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ════════════════════════════════════════════════════════════════════
    // STEP 3: Fused SiLU activation + element-wise multiply
    // gate_buf[i] = silu(gate_buf[i]) * up_buf[i]
    // ════════════════════════════════════════════════════════════════════
    for (uint i = tid; i < F; i += THREADS_PER_GROUP) {
        gate_buf[i] = silu(gate_buf[i]) * up_buf[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ════════════════════════════════════════════════════════════════════
    // STEP 4: Compute out = gate_buf @ w2  (ffn_dim -> hidden_dim)
    // STEP 5: Atomic accumulate gate_weight * out to output buffer
    //
    // w2 layout: [F/2, H] packed INT4 (K=ffn_dim, N=hidden_dim)
    // byte_idx = (k/2) * H + col
    // ════════════════════════════════════════════════════════════════════
    for (uint col = tid; col < H; col += THREADS_PER_GROUP) {
        float acc_w2 = 0.0f;

        for (uint k = 0; k < F; k += 2) {
            float g0 = gate_buf[k];
            float g1 = (k + 1 < F) ? gate_buf[k + 1] : 0.0f;

            uint byte_idx = expert_w2_offset + (k / 2) * H + col;
            uint8_t packed_w2 = expert_w2[byte_idx];
            float w2_lo = dequant_int4(packed_w2);
            float w2_hi = dequant_int4(packed_w2 >> 4);
            acc_w2 += g0 * w2_lo + g1 * w2_hi;
        }

        // Atomic accumulate: multiple experts write to the same output
        // Use atomic_fetch_add on the output buffer.
        // Metal 2.4+ supports atomic<float> on device buffers.
        //
        // Reinterpret as device atomic_float* for atomic accumulation.
        // This is safe because output is exclusively written via atomics
        // from this kernel (caller must zero output before dispatch).
        device atomic_float* out_atomic =
            reinterpret_cast<device atomic_float*>(&output[col]);
        atomic_fetch_add_explicit(out_atomic, gate_weight * acc_w2,
                                  memory_order_relaxed);
    }
}
