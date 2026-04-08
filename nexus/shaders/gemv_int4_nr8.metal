/// NEXUS Metal Shader — INT4 GEMV with NR0=8 (8 rows per SIMD group)
///
/// llama.cpp's "single biggest optimization" — took them from 70% to 99%
/// of theoretical throughput.
///
/// Instead of 1 thread computing 1 output element (poor SIMD utilization),
/// 1 SIMD group of 32 threads cooperatively computes 8 output elements.
///
/// Each SIMD group (32 threads):
///   - Thread t handles K-elements at indices t, t+32, t+64, ... for ALL 8 rows
///   - Partial dot products accumulated per thread
///   - simd_sum() hardware reduction gives final result (zero overhead on Apple Silicon)
///
/// This gives:
///   - 8x better work per SIMD group
///   - Coalesced memory reads (threads read consecutive K elements)
///   - Hardware SIMD reduction (no threadgroup memory needed)
///   - Better register utilization (8 accumulators per thread)

#include <metal_stdlib>
using namespace metal;

struct NR8Params {
    uint N;     // Output dimension (number of rows)
    uint K;     // Input dimension (dot product length)
};

constant uint NR0 = 8;         // Rows per SIMD group
constant uint SIMD_SIZE = 32;  // Apple Silicon SIMD width

kernel void gemv_int4_nr8(
    device const float*   activations [[buffer(0)]],   // [K] FP32 input
    device const uint8_t* weights_q   [[buffer(1)]],   // [K/2, N] packed INT4
    device float*         output      [[buffer(2)]],   // [N] FP32 output
    constant NR8Params&   params      [[buffer(3)]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]],  // Which SIMD group
    uint simd_lid  [[thread_index_in_simdgroup]],       // Lane within SIMD group
    uint tg_id     [[threadgroup_position_in_grid]])     // Which threadgroup
{
    uint N = params.N;
    uint K = params.K;

    // Each SIMD group handles NR0=8 consecutive output rows
    // Total SIMD groups across grid covers all N outputs
    uint simds_per_tg = 4;  // 4 SIMD groups × 32 threads = 128 threads/tg
    uint global_simd = tg_id * simds_per_tg + simd_gid;
    uint row_base = global_simd * NR0;

    if (row_base >= N) return;

    // 8 accumulators — one per output row
    float acc[NR0] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint rows_this_group = min(NR0, N - row_base);

    // Each of 32 SIMD lanes processes K/32 elements (strided)
    // Lane t handles elements: t*2, t*2+64, t*2+128, ... (stepping by SIMD_SIZE*2)
    for (uint k_base = simd_lid * 2; k_base < K; k_base += SIMD_SIZE * 2) {
        // Load 2 activation elements (shared across all 8 rows)
        float a0 = activations[k_base];
        float a1 = (k_base + 1 < K) ? activations[k_base + 1] : 0.0f;

        // For each of the 8 output rows, load weight and accumulate
        for (uint r = 0; r < rows_this_group; r++) {
            uint row = row_base + r;
            // Weight byte index: weights stored as [K/2, N] packed
            uint byte_idx = (k_base / 2) * N + row;
            uint8_t packed = weights_q[byte_idx];

            // Inline INT4 dequant
            float w0 = (float(packed & 0x0F) - 8.0f) * 0.125f;
            float w1 = (float(packed >> 4) - 8.0f) * 0.125f;

            acc[r] += a0 * w0 + a1 * w1;
        }
    }

    // SIMD hardware reduction — sum partial products across all 32 lanes
    // This is the key: simd_sum is a single-cycle hardware instruction on Apple Silicon
    for (uint r = 0; r < rows_this_group; r++) {
        float result = simd_sum(acc[r]);

        // Only lane 0 writes the final result
        if (simd_lid == 0) {
            output[row_base + r] = result;
        }
    }
}
