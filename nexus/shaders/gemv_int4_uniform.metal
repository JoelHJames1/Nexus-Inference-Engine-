/// NEXUS Metal Shader — Fused INT4 Uniform Dequant + GEMV
///
/// Simplified INT4 GEMV without per-group scales/zeros.
/// Maps each nibble to float via: (nibble - 8) * 0.125
/// This matches the NXF INT4 encoding from the GGUF converter.
///
/// output[col] = sum_k( activations[k] * dequant(weights_q[k, col]) )

#include <metal_stdlib>
using namespace metal;

struct UniformDequantParams {
    uint N;         // Output columns
    uint K;         // Inner dimension (input/activation dim)
};

kernel void gemv_int4_uniform(
    device const float*   activations [[buffer(0)]],   // [1, K] FP32
    device const uint8_t* weights_q   [[buffer(1)]],   // Packed INT4 data
    device float*         output      [[buffer(2)]],   // [1, N] FP32
    constant UniformDequantParams& params [[buffer(3)]],
    uint                  gid         [[thread_position_in_grid]])
{
    if (gid >= params.N) return;

    uint col = gid;
    uint K = params.K;
    uint N = params.N;
    float acc = 0.0f;

    // Process 2 elements per byte (INT4 packed)
    for (uint k = 0; k < K; k += 2) {
        // Byte index: weights are stored as [K/2, N] packed layout
        // Each byte contains two 4-bit values for the same column
        uint byte_idx = (k / 2) * N + col;
        uint8_t packed = weights_q[byte_idx];

        // Unpack and dequant: (nibble - 8) * 0.125
        float w0 = (float(packed & 0x0F) - 8.0) * 0.125;
        acc += activations[k] * w0;

        if (k + 1 < K) {
            float w1 = (float((packed >> 4) & 0x0F) - 8.0) * 0.125;
            acc += activations[k + 1] * w1;
        }
    }

    output[col] = acc;
}
