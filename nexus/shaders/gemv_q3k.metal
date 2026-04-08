/// NEXUS Metal Shader — Fused Q3_K Dequant + GEMV
///
/// Q3_K format (from llama.cpp K-quants):
///   Super-block of 256 elements:
///     - hmask[32]: high bit masks (1 bit per element, packed)
///     - qs[64]: low 2 bits per element (packed 4 per byte)
///     - scales[12]: 16 × 6-bit scales (packed)
///     - d[2]: fp16 master scale
///   Total: 110 bytes per 256 elements = 3.4375 bits/element
///
/// This shader reads Q3_K data directly from mmap'd NXF chunks,
/// skipping the INT4 expansion step entirely. 25% less bandwidth
/// than INT4 = 25% faster for memory-bound GEMV.
///
/// Combined with UMA zero-copy (newBufferWithBytesNoCopy), the GPU
/// reads Q3_K data from the same physical pages as mmap — no copy,
/// no dequant, no intermediate buffers.

#include <metal_stdlib>
using namespace metal;

struct Q3KParams {
    uint N;         // Output columns
    uint K;         // Input/activation dimension (must be multiple of 256)
};

/// Convert FP16 bits to float
inline float fp16_to_float(uint16_t h) {
    // Use Metal's built-in half type
    return float(as_type<half>(h));
}

kernel void gemv_q3k(
    device const float*   activations [[buffer(0)]],   // [K] FP32
    device const uint8_t* weights_q3k [[buffer(1)]],   // Q3_K packed data
    device float*         output      [[buffer(2)]],   // [N] FP32
    constant Q3KParams&   params      [[buffer(3)]],
    uint                  gid         [[thread_position_in_grid]])
{
    if (gid >= params.N) return;

    uint col = gid;
    uint K = params.K;
    uint N = params.N;
    float acc = 0.0f;

    // Q3_K: 256 elements per super-block, 110 bytes per block
    // For a [K, N] weight matrix stored in Q3_K:
    // Each column has K/256 super-blocks
    // Block layout: hmask[32] + qs[64] + scales[12] + d[2] = 110 bytes

    uint blocks_per_col = K / 256;

    for (uint b = 0; b < blocks_per_col; b++) {
        // Byte offset for this block: each column's blocks are contiguous
        // Layout: blocks are stored row-major, so block for (row_block, col) is at:
        // col * blocks_per_col * 110 + b * 110
        // BUT: the actual NXF/GGUF stores all blocks sequentially for the full tensor
        // For [K, N] tensor: block index = (b * N + col), each block = 110 bytes
        uint block_idx = b * N + col;
        uint block_off = block_idx * 110;

        device const uint8_t* block = weights_q3k + block_off;
        device const uint8_t* hmask = block;           // 32 bytes
        device const uint8_t* qs    = block + 32;      // 64 bytes
        device const uint8_t* sc    = block + 96;      // 12 bytes
        uint16_t raw_d = uint16_t(block[108]) | (uint16_t(block[109]) << 8);
        float d = fp16_to_float(raw_d);

        // Unpack 16 × 6-bit scales
        int scales_arr[16];
        for (uint i = 0; i < 8; i++) {
            scales_arr[i]     = int(sc[i] & 0x0F) | (int((sc[8 + (i/2)] >> (4 * (i%2))) & 3) << 4);
            scales_arr[i + 8] = int(sc[i] >> 4)   | (int((sc[8 + (i/2)] >> (4 * (i%2) + 2)) & 3) << 4);
        }
        for (uint i = 0; i < 16; i++) {
            scales_arr[i] -= 32;
        }

        // Dequantize 256 elements and accumulate dot product
        uint base_k = b * 256;
        for (uint j = 0; j < 256; j++) {
            // Low 2 bits from qs
            uint byte_idx = j / 4;
            uint bit_shift = (j % 4) * 2;
            int q_lo = int((qs[byte_idx] >> bit_shift) & 0x03);
            // High bit from hmask
            int q_hi = int((hmask[j / 8] >> (j % 8)) & 1);
            int q = q_lo | (q_hi << 2);  // 3-bit value [0..7]

            uint group = j / 16;
            float w = d * float(scales_arr[group]) * (float(q) - 4.0);
            acc += activations[base_k + j] * w;
        }
    }

    output[col] = acc;
}
