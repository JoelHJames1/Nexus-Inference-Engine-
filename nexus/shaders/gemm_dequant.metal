/// NEXUS Metal Shader — Fused Dequantization + GEMM
///
/// The key performance kernel for NEXUS. Loads INT4-quantized weights from
/// a shared MTLBuffer (UMA zero-copy), unpacks to FP16, and multiplies with
/// activations — all in one dispatch. No intermediate buffer needed.
///
/// Apple Silicon advantage: storageModeShared buffers mean the CPU can write
/// quantized weights and the GPU reads them with zero copy overhead.
///
/// Tile sizes chosen for Apple Silicon:
///   - Threadgroup memory: 16 KB limit
///   - SIMD group size: 32 threads
///   - Br=32, Bc=32 tiles (1024 threads per threadgroup)

#include <metal_stdlib>
using namespace metal;

/// INT4 dequantization parameters
struct DequantParams {
    uint rows;          // Output rows (M)
    uint cols;          // Output cols (N)
    uint K;             // Inner dimension
    uint group_size;    // Quantization group size (128)
};

/// Fused INT4 dequant + GEMV (matrix-vector multiply)
/// For single-token decode: A[1,K] x W_q[K,N] -> C[1,N]
/// W_q is stored as packed INT4 with per-group scales and zeros.
kernel void gemv_dequant_int4(
    device const float*   activations [[buffer(0)]],   // [1, K] FP32
    device const uint8_t* weights_q   [[buffer(1)]],   // [K, N/2] packed INT4
    device const float*   scales      [[buffer(2)]],   // [K/group_size, N]
    device const float*   zeros       [[buffer(3)]],   // [K/group_size, N]
    device float*         output      [[buffer(4)]],   // [1, N] FP32
    constant DequantParams& params    [[buffer(5)]],
    uint                  gid         [[thread_position_in_grid]],
    uint                  tid         [[thread_index_in_threadgroup]],
    uint                  tg_size     [[threads_per_threadgroup]])
{
    uint col = gid;
    if (col >= params.cols) return;

    float acc = 0.0f;
    uint K = params.K;
    uint gs = params.group_size;

    for (uint k = 0; k < K; k += 2) {
        // Load packed byte containing two 4-bit weights
        uint byte_idx = (k / 2) * params.cols + col;  // Row-major packed layout
        // Note: actual layout depends on quantization scheme. This is simplified.
        // Phase 2 will optimize memory access patterns for coalescing.

        uint8_t packed = weights_q[byte_idx];
        uint group_idx = (k / gs) * params.cols + col;
        float scale = scales[group_idx];
        float zero = zeros[group_idx];

        // Unpack low nibble
        float w0 = (float(packed & 0x0F) - zero) * scale;
        acc += activations[k] * w0;

        // Unpack high nibble
        if (k + 1 < K) {
            float w1 = (float((packed >> 4) & 0x0F) - zero) * scale;
            acc += activations[k + 1] * w1;
        }
    }

    output[col] = acc;
}

/// Tiled GEMM for FP32 (baseline, will be optimized in Phase 2)
/// C[M,N] = A[M,K] x B[K,N]
/// Uses 32x32 tiles in threadgroup memory.
constant uint TILE_SIZE = 32;

kernel void gemm_f32_tiled(
    device const float* A           [[buffer(0)]],
    device const float* B           [[buffer(1)]],
    device float*       C           [[buffer(2)]],
    constant uint&      M           [[buffer(3)]],
    constant uint&      N           [[buffer(4)]],
    constant uint&      K           [[buffer(5)]],
    uint2               tid         [[thread_position_in_threadgroup]],
    uint2               tg_pos      [[threadgroup_position_in_grid]])
{
    // Tile position
    uint row = tg_pos.y * TILE_SIZE + tid.y;
    uint col = tg_pos.x * TILE_SIZE + tid.x;

    if (row >= M || col >= N) return;

    // Threadgroup shared memory for tiles
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;

    for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile of A
        uint ak = t * TILE_SIZE + tid.x;
        tileA[tid.y][tid.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;

        // Load tile of B
        uint bk = t * TILE_SIZE + tid.y;
        tileB[tid.y][tid.x] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product
        for (uint i = 0; i < TILE_SIZE; i++) {
            acc += tileA[tid.y][i] * tileB[i][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    C[row * N + col] = acc;
}
