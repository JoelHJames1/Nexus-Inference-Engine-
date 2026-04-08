/// NEXUS Metal Shader — TurboQuant KV Cache Quantization
///
/// GPU implementation of TurboQuant (Google Research, arXiv:2504.19874, ICLR 2026)
/// for online KV cache compression on Apple Silicon.
///
/// Algorithm:
///   1. Random rotation: y = Π · x  (Π is a random orthogonal matrix)
///   2. Scalar quantization: each coordinate quantized to b bits using
///      precomputed Lloyd-Max codebook for Beta distribution
///   3. Store: packed bit indices + L2 norm
///
/// Dequantization:
///   1. Lookup centroids from codebook
///   2. Inverse rotation: x̃ = Π^T · ỹ
///   3. Rescale by stored norm
///
/// This enables 3.5-bit KV with quality neutrality (0.997 needle-in-haystack)
/// and 2.5-bit with marginal degradation, achieving 4.5-6.4x compression.
///
/// Phase 3 will integrate this with paged attention for tiered KV compression.

#include <metal_stdlib>
using namespace metal;

struct TurboQuantParams {
    uint dim;            // Vector dimension (head_dim, typically 128)
    uint bits;           // Bits per coordinate (2 or 4)
    uint num_centroids;  // 2^bits
    uint num_vectors;    // Number of vectors to quantize
};

/// Quantize a batch of vectors using TurboQuant MSE.
/// Input: FP32 vectors (already rotated by CPU/separate kernel)
/// Output: packed quantization indices
kernel void turbo_quant_encode(
    device const float*         input       [[buffer(0)]],  // [num_vectors, dim]
    device const float*         centroids   [[buffer(1)]],  // [num_centroids]
    device const float*         boundaries  [[buffer(2)]],  // [num_centroids - 1]
    device uint8_t*             output      [[buffer(3)]],  // Packed indices
    device float*               norms       [[buffer(4)]],  // [num_vectors] L2 norms
    constant TurboQuantParams&  params      [[buffer(5)]],
    uint2                       gid         [[thread_position_in_grid]])
{
    uint vec_idx = gid.y;
    uint coord = gid.x;

    if (vec_idx >= params.num_vectors || coord >= params.dim) return;

    float val = input[vec_idx * params.dim + coord];

    // Binary search for nearest centroid using boundaries
    uint idx = 0;
    for (uint b = 0; b < params.num_centroids - 1; b++) {
        if (val > boundaries[b]) idx = b + 1;
    }

    // Pack indices (4-bit: 2 per byte, 2-bit: 4 per byte)
    if (params.bits == 4) {
        uint byte_pos = vec_idx * (params.dim / 2) + coord / 2;
        if (coord % 2 == 0) {
            output[byte_pos] = (output[byte_pos] & 0xF0) | (idx & 0x0F);
        } else {
            output[byte_pos] = (output[byte_pos] & 0x0F) | ((idx & 0x0F) << 4);
        }
    } else if (params.bits == 2) {
        uint byte_pos = vec_idx * (params.dim / 4) + coord / 4;
        uint shift = (coord % 4) * 2;
        uint mask = ~(0x03 << shift);
        output[byte_pos] = (output[byte_pos] & mask) | ((idx & 0x03) << shift);
    }

    // Compute L2 norm (thread 0 of each vector)
    // Note: proper implementation would use parallel reduction
    if (coord == 0) {
        float norm_sq = 0.0f;
        for (uint i = 0; i < params.dim; i++) {
            float v = input[vec_idx * params.dim + i];
            norm_sq += v * v;
        }
        norms[vec_idx] = sqrt(norm_sq);
    }
}

/// Dequantize packed TurboQuant indices back to FP32.
kernel void turbo_quant_decode(
    device const uint8_t*       input       [[buffer(0)]],  // Packed indices
    device const float*         centroids   [[buffer(1)]],  // [num_centroids]
    device const float*         norms       [[buffer(2)]],  // [num_vectors]
    device float*               output      [[buffer(3)]],  // [num_vectors, dim]
    constant TurboQuantParams&  params      [[buffer(4)]],
    uint2                       gid         [[thread_position_in_grid]])
{
    uint vec_idx = gid.y;
    uint coord = gid.x;

    if (vec_idx >= params.num_vectors || coord >= params.dim) return;

    // Unpack index
    uint idx;
    if (params.bits == 4) {
        uint byte_pos = vec_idx * (params.dim / 2) + coord / 2;
        uint8_t packed = input[byte_pos];
        idx = (coord % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
    } else {
        uint byte_pos = vec_idx * (params.dim / 4) + coord / 4;
        uint shift = (coord % 4) * 2;
        idx = (input[byte_pos] >> shift) & 0x03;
    }

    // Lookup centroid and rescale
    output[vec_idx * params.dim + coord] = centroids[idx];
    // Note: rotation (Π^T) and norm rescaling are applied by a separate kernel
    // or fused into the attention computation
}

/// Random rotation kernel: y = Π · x
/// Π is stored as a dense rotation matrix [dim, dim].
/// Phase 3 will use structured Hadamard rotations for O(d log d) cost.
kernel void rotate_vectors(
    device const float*  input     [[buffer(0)]],  // [num_vectors, dim]
    device const float*  rotation  [[buffer(1)]],  // [dim, dim] orthogonal matrix
    device float*        output    [[buffer(2)]],  // [num_vectors, dim]
    constant uint&       dim       [[buffer(3)]],
    constant uint&       num_vecs  [[buffer(4)]],
    uint2                gid       [[thread_position_in_grid]])
{
    uint vec_idx = gid.y;
    uint out_coord = gid.x;

    if (vec_idx >= num_vecs || out_coord >= dim) return;

    // Matrix-vector multiply: output[vec][out_coord] = sum_j rotation[out_coord][j] * input[vec][j]
    float acc = 0.0f;
    for (uint j = 0; j < dim; j++) {
        acc += rotation[out_coord * dim + j] * input[vec_idx * dim + j];
    }
    output[vec_idx * dim + out_coord] = acc;
}
