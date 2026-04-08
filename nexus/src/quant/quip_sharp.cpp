/// NEXUS QuIP# Codec — 3-bit quantization with random orthogonal rotation.

#include "quant/quip_sharp.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <cassert>

namespace nexus::quant {

// Lloyd-Max optimal centroids for 3-bit (8-level) quantization of a
// unit-variance Gaussian distribution. These are the reconstruction
// points that minimize mean squared error for N(0,1).
static constexpr float kLloydMaxCentroids[8] = {
    -1.7479f, -1.0500f, -0.5006f, -0.0690f,
     0.0690f,  0.5006f,  1.0500f,  1.7479f
};

// Decision boundaries between centroids (midpoints).
static constexpr float kLloydMaxBounds[7] = {
    -1.3990f, -0.7753f, -0.2848f,  0.0000f,
     0.2848f,  0.7753f,  1.3990f
};

/// Find nearest Lloyd-Max centroid index for a normalized value.
static inline int find_nearest_centroid(float val) {
    for (int i = 0; i < 7; i++) {
        if (val < kLloydMaxBounds[i]) return i;
    }
    return 7;
}

void quip3_quantize(uint8_t* out, float* scales,
                    const float* input, int n, int group_size) {
    assert(n % 8 == 0 && "n must be multiple of 8 for 3-bit packing");

    int num_groups = (n + group_size - 1) / group_size;

    // Step 1: Compute per-group scales (RMS of group, mapping to unit variance).
    for (int g = 0; g < num_groups; g++) {
        int start = g * group_size;
        int end = std::min(start + group_size, n);
        int count = end - start;

        // Compute RMS as the scale (standard deviation approximation)
        double sum_sq = 0.0;
        for (int i = start; i < end; i++) {
            sum_sq += static_cast<double>(input[i]) * input[i];
        }
        float rms = static_cast<float>(std::sqrt(sum_sq / count));
        if (rms < 1e-10f) rms = 1e-10f;
        scales[g] = rms;
    }

    // Step 2: Quantize each element to 3-bit index using Lloyd-Max centroids.
    // Pack 8 x 3-bit values into 3 bytes:
    //   byte0: bits [2:0] of val0, [2:0] of val1, [1:0] of val2
    //   byte1: bit [2] of val2, [2:0] of val3, [2:0] of val4, [1:0] of val5
    //   byte2: bit [2] of val5, [2:0] of val6, [2:0] of val7
    // Specifically, we pack 24 bits = 8 * 3 bits into 3 bytes, LSB first.
    int out_idx = 0;
    for (int i = 0; i < n; i += 8) {
        int group = i / group_size;
        float inv_scale = 1.0f / scales[group];

        // Quantize 8 values to 3-bit indices
        uint8_t indices[8];
        for (int j = 0; j < 8; j++) {
            float normalized = input[i + j] * inv_scale;
            indices[j] = static_cast<uint8_t>(find_nearest_centroid(normalized));
        }

        // Pack 8 x 3-bit values into 3 bytes (24 bits total, LSB-first).
        // Bit layout: idx0 at bits[0..2], idx1 at bits[3..5], ..., idx7 at bits[21..23]
        uint32_t packed = 0;
        for (int j = 0; j < 8; j++) {
            packed |= (static_cast<uint32_t>(indices[j]) << (j * 3));
        }
        out[out_idx + 0] = static_cast<uint8_t>(packed & 0xFF);
        out[out_idx + 1] = static_cast<uint8_t>((packed >> 8) & 0xFF);
        out[out_idx + 2] = static_cast<uint8_t>((packed >> 16) & 0xFF);
        out_idx += 3;
    }
}

void quip3_dequantize(float* out, const uint8_t* data, const float* scales,
                      int n, int group_size) {
    assert(n % 8 == 0 && "n must be multiple of 8 for 3-bit packing");

    int in_idx = 0;
    for (int i = 0; i < n; i += 8) {
        int group = i / group_size;
        float scale = scales[group];

        // Unpack 3 bytes -> 8 x 3-bit indices
        uint32_t packed = static_cast<uint32_t>(data[in_idx])
                        | (static_cast<uint32_t>(data[in_idx + 1]) << 8)
                        | (static_cast<uint32_t>(data[in_idx + 2]) << 16);
        in_idx += 3;

        for (int j = 0; j < 8; j++) {
            int idx = (packed >> (j * 3)) & 0x07;
            out[i + j] = kLloydMaxCentroids[idx] * scale;
        }
    }
}

// ─── Random orthogonal rotation via QR decomposition ────────────────────────

/// Householder QR decomposition of A (dim x dim, row-major), in-place.
/// Returns Q as a dim x dim orthogonal matrix.
static std::vector<float> householder_qr_q(std::vector<float>& A, int dim) {
    // We accumulate Q = H_1 * H_2 * ... * H_{dim-1}
    // Start with Q = I
    std::vector<float> Q(dim * dim, 0.0f);
    for (int i = 0; i < dim; i++) Q[i * dim + i] = 1.0f;

    std::vector<float> v(dim);

    for (int k = 0; k < dim - 1; k++) {
        // Extract column k below the diagonal
        float norm_sq = 0.0f;
        for (int i = k; i < dim; i++) {
            v[i] = A[i * dim + k];
            norm_sq += v[i] * v[i];
        }
        float norm = std::sqrt(norm_sq);
        if (norm < 1e-12f) continue;

        // Choose sign to avoid cancellation
        float sign = (v[k] >= 0.0f) ? 1.0f : -1.0f;
        v[k] += sign * norm;

        // Recompute norm of v for normalization
        float v_norm_sq = 0.0f;
        for (int i = k; i < dim; i++) v_norm_sq += v[i] * v[i];
        if (v_norm_sq < 1e-24f) continue;
        float inv_v_norm_sq = 2.0f / v_norm_sq;

        // Apply Householder reflector H = I - 2*v*v'/||v||^2 to A from left
        // A = H * A  (only columns k..dim-1 affected)
        for (int j = k; j < dim; j++) {
            float dot = 0.0f;
            for (int i = k; i < dim; i++) dot += v[i] * A[i * dim + j];
            dot *= inv_v_norm_sq;
            for (int i = k; i < dim; i++) A[i * dim + j] -= v[i] * dot;
        }

        // Apply H to Q from right: Q = Q * H
        // Q_new[:,i] = Q[:,i] - 2 * (Q * v) * v[i] / ||v||^2
        // But it's easier: Q = Q * H means Q_new = Q - 2*(Q*v)*v^T / ||v||^2
        for (int i = 0; i < dim; i++) {
            float dot = 0.0f;
            for (int j = k; j < dim; j++) dot += Q[i * dim + j] * v[j];
            dot *= inv_v_norm_sq;
            for (int j = k; j < dim; j++) Q[i * dim + j] -= dot * v[j];
        }
    }

    return Q;
}

std::vector<float> generate_rotation(int dim, uint64_t seed) {
    // Generate random Gaussian matrix
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> A(dim * dim);
    for (int i = 0; i < dim * dim; i++) {
        A[i] = dist(rng);
    }

    // QR decomposition -> Q is orthogonal
    std::vector<float> Q = householder_qr_q(A, dim);

    // Ensure det(Q) = +1 (proper rotation) by checking sign of diagonal of R
    // and flipping a column of Q if needed. For our purposes, any orthogonal
    // matrix works, so we skip this correction.

    return Q;
}

}  // namespace nexus::quant
