/// NEXUS TurboQuant — KV cache quantization implementation.
///
/// Random rotation via QR decomposition of Gaussian matrix, followed by
/// Lloyd-Max scalar quantization with precomputed codebooks.

#include "kv/turbo_quant.h"
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <stdexcept>

namespace nexus::kv {

// ─── Lloyd-Max codebook ─────────────────────────────────────────────────────

uint8_t LloydMaxCodebook::encode(float x) const {
    // Binary search through boundaries to find the nearest centroid.
    int lo = 0;
    int hi = n_centroids - 1;
    for (int i = 0; i < static_cast<int>(boundaries.size()); ++i) {
        if (x < boundaries[i]) {
            hi = i;
            break;
        }
        lo = i + 1;
    }
    // lo == hi at this point (or lo is the last bucket).
    return static_cast<uint8_t>(lo);
}

float LloydMaxCodebook::decode(uint8_t code) const {
    return centroids[code];
}

// ─── CompressedVectors ──────────────────────────────────────────────────────

size_t CompressedVectors::byte_size() const {
    size_t code_bits = static_cast<size_t>(n_vectors) * dim * bits;
    size_t code_bytes = (code_bits + 7) / 8;
    size_t norm_bytes = static_cast<size_t>(n_vectors) * sizeof(float);
    return code_bytes + norm_bytes;
}

// ─── TurboQuantKV ───────────────────────────────────────────────────────────

TurboQuantKV::TurboQuantKV(int head_dim, int bits, uint64_t seed)
    : head_dim_(head_dim)
    , bits_(bits)
    , seed_(seed) {

    if (bits < 2 || bits > 4) {
        throw std::invalid_argument("TurboQuantKV: bits must be 2, 3, or 4");
    }
    if (head_dim <= 0) {
        throw std::invalid_argument("TurboQuantKV: head_dim must be positive");
    }

    generate_rotation_matrix();
    build_codebook();
}

TurboQuantKV::~TurboQuantKV() = default;

// ─── Rotation matrix generation via QR decomposition ────────────────────────

void TurboQuantKV::generate_rotation_matrix() {
    const int d = head_dim_;
    rotation_.resize(d * d);
    rotation_t_.resize(d * d);

    // Generate random Gaussian matrix.
    std::mt19937_64 rng(seed_);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    std::vector<float> A(d * d);
    for (int i = 0; i < d * d; ++i) {
        A[i] = normal(rng);
    }

    // QR decomposition via modified Gram-Schmidt orthogonalization.
    // Q columns are stored in rotation_ as rows (row-major = transposed column-major).
    // Working column-major: Q[:, j] is built iteratively.
    std::vector<float> Q(d * d, 0.0f);  // column-major
    std::vector<float> col(d);

    for (int j = 0; j < d; ++j) {
        // Copy column j of A.
        for (int i = 0; i < d; ++i) {
            col[i] = A[i * d + j];
        }

        // Subtract projections onto previous Q columns.
        for (int k = 0; k < j; ++k) {
            float dot = 0.0f;
            for (int i = 0; i < d; ++i) {
                dot += col[i] * Q[k * d + i];  // Q[:, k] stored at Q[k*d + ...]
            }
            for (int i = 0; i < d; ++i) {
                col[i] -= dot * Q[k * d + i];
            }
        }

        // Normalize.
        float norm = 0.0f;
        for (int i = 0; i < d; ++i) {
            norm += col[i] * col[i];
        }
        norm = std::sqrt(norm);
        if (norm < 1e-10f) {
            // Degenerate — regenerate with different seed (extremely rare).
            throw std::runtime_error("TurboQuantKV: degenerate Gram-Schmidt column");
        }
        float inv_norm = 1.0f / norm;
        for (int i = 0; i < d; ++i) {
            Q[j * d + i] = col[i] * inv_norm;
        }
    }

    // Convert Q (column-major storage where Q[:, j] = Q[j*d + ...]) to row-major Pi.
    // Pi[i][j] = Q[j*d + i] (the i-th component of the j-th column).
    // But we want rotation_[i*d + j] = Pi[i][j] so that y = Pi * x is:
    //   y[i] = sum_j Pi[i][j] * x[j]
    // So rotation_[i*d + j] = Q[j*d + i].
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            rotation_[i * d + j] = Q[j * d + i];
        }
    }

    // Transpose: rotation_t_ = Pi^T (which is Pi^{-1} for orthogonal matrices).
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            rotation_t_[i * d + j] = rotation_[j * d + i];
        }
    }
}

// ─── Lloyd-Max codebook construction ────────────────────────────────────────
//
// After random rotation, each coordinate of the rotated vector approximately
// follows N(0, 1/d). We precompute optimal Lloyd-Max centroids and boundaries
// for this distribution at the target bit width.

/// Standard normal PDF.
static float normal_pdf(float x, float sigma) {
    float z = x / sigma;
    return std::exp(-0.5f * z * z) / (sigma * std::sqrt(2.0f * static_cast<float>(M_PI)));
}

/// Standard normal CDF (Abramowitz & Stegun approximation).
static float normal_cdf(float x, float sigma) {
    float z = x / sigma;
    // Use erfc for accuracy.
    return 0.5f * std::erfc(-z * static_cast<float>(M_SQRT1_2));
}

/// Conditional mean of N(0, sigma^2) in interval [a, b]:
/// E[X | a <= X <= b] = sigma^2 * (pdf(a) - pdf(b)) / (cdf(b) - cdf(a))
static float conditional_mean(float a, float b, float sigma) {
    float pa = normal_pdf(a, sigma);
    float pb = normal_pdf(b, sigma);
    float ca = normal_cdf(a, sigma);
    float cb = normal_cdf(b, sigma);
    float denom = cb - ca;
    if (denom < 1e-12f) return 0.5f * (a + b);
    return sigma * sigma * (pa - pb) / denom;
}

void TurboQuantKV::build_codebook() {
    const int n = 1 << bits_;
    codebook_.bits = bits_;
    codebook_.n_centroids = n;
    codebook_.centroids.resize(n);
    codebook_.boundaries.resize(n - 1);

    // Standard deviation of each coordinate after rotation: sigma = 1/sqrt(d).
    // This assumes input vectors have unit variance per coordinate.
    float sigma = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    // Initialize centroids uniformly in [-3*sigma, 3*sigma].
    float lo = -3.0f * sigma;
    float hi = 3.0f * sigma;
    for (int i = 0; i < n; ++i) {
        codebook_.centroids[i] = lo + (hi - lo) * (i + 0.5f) / n;
    }

    // Lloyd-Max iteration.
    constexpr int kMaxIter = 100;
    constexpr float kTol = 1e-7f;

    for (int iter = 0; iter < kMaxIter; ++iter) {
        // Update boundaries: midpoints of adjacent centroids.
        for (int i = 0; i < n - 1; ++i) {
            codebook_.boundaries[i] = 0.5f * (codebook_.centroids[i] + codebook_.centroids[i + 1]);
        }

        // Update centroids: conditional mean in each partition.
        float max_delta = 0.0f;
        for (int i = 0; i < n; ++i) {
            float a = (i == 0) ? -10.0f * sigma : codebook_.boundaries[i - 1];
            float b = (i == n - 1) ? 10.0f * sigma : codebook_.boundaries[i];

            float new_centroid = conditional_mean(a, b, sigma);
            float delta = std::abs(new_centroid - codebook_.centroids[i]);
            if (delta > max_delta) max_delta = delta;
            codebook_.centroids[i] = new_centroid;
        }

        if (max_delta < kTol * sigma) break;
    }

    // Final boundary update.
    for (int i = 0; i < n - 1; ++i) {
        codebook_.boundaries[i] = 0.5f * (codebook_.centroids[i] + codebook_.centroids[i + 1]);
    }
}

// ─── Rotation helpers ───────────────────────────────────────────────────────

void TurboQuantKV::rotate(float* out, const float* in) const {
    const int d = head_dim_;
    for (int i = 0; i < d; ++i) {
        float sum = 0.0f;
        const float* row = rotation_.data() + i * d;
        for (int j = 0; j < d; ++j) {
            sum += row[j] * in[j];
        }
        out[i] = sum;
    }
}

void TurboQuantKV::rotate_inverse(float* out, const float* in) const {
    const int d = head_dim_;
    for (int i = 0; i < d; ++i) {
        float sum = 0.0f;
        const float* row = rotation_t_.data() + i * d;
        for (int j = 0; j < d; ++j) {
            sum += row[j] * in[j];
        }
        out[i] = sum;
    }
}

// ─── Bit packing / unpacking ────────────────────────────────────────────────

void TurboQuantKV::pack_codes(uint8_t* packed, const uint8_t* codes, int n_codes, int bits) {
    // Pack n_codes values of `bits` width each into a byte stream, LSB-first.
    size_t total_bits = static_cast<size_t>(n_codes) * bits;
    size_t total_bytes = (total_bits + 7) / 8;
    std::memset(packed, 0, total_bytes);

    uint32_t bit_pos = 0;
    for (int i = 0; i < n_codes; ++i) {
        uint8_t val = codes[i];
        uint32_t byte_idx = bit_pos / 8;
        uint32_t bit_off = bit_pos % 8;

        // Write bits — may span two bytes.
        packed[byte_idx] |= static_cast<uint8_t>(val << bit_off);
        if (bit_off + bits > 8) {
            packed[byte_idx + 1] |= static_cast<uint8_t>(val >> (8 - bit_off));
        }
        bit_pos += bits;
    }
}

void TurboQuantKV::unpack_codes(uint8_t* codes, const uint8_t* packed, int n_codes, int bits) {
    uint8_t mask = static_cast<uint8_t>((1u << bits) - 1);
    uint32_t bit_pos = 0;

    for (int i = 0; i < n_codes; ++i) {
        uint32_t byte_idx = bit_pos / 8;
        uint32_t bit_off = bit_pos % 8;

        uint16_t word = packed[byte_idx];
        if (bit_off + bits > 8) {
            word |= static_cast<uint16_t>(packed[byte_idx + 1]) << 8;
        }
        codes[i] = static_cast<uint8_t>((word >> bit_off) & mask);
        bit_pos += bits;
    }
}

// ─── Quantize ───────────────────────────────────────────────────────────────

void TurboQuantKV::quantize(uint8_t* output_codes, float* output_norms,
                            const float* input, int n_vectors) const {
    const int d = head_dim_;
    std::vector<float> rotated(d);
    std::vector<uint8_t> codes(d);

    for (int v = 0; v < n_vectors; ++v) {
        const float* vec = input + v * d;

        // Compute L2 norm.
        float norm_sq = 0.0f;
        for (int i = 0; i < d; ++i) {
            norm_sq += vec[i] * vec[i];
        }
        float norm = std::sqrt(norm_sq);
        output_norms[v] = norm;

        // Normalize and rotate.
        if (norm > 1e-12f) {
            float inv_norm = 1.0f / norm;
            std::vector<float> normalized(d);
            for (int i = 0; i < d; ++i) {
                normalized[i] = vec[i] * inv_norm;
            }
            rotate(rotated.data(), normalized.data());
        } else {
            std::memset(rotated.data(), 0, d * sizeof(float));
        }

        // Scalar quantize each coordinate.
        for (int i = 0; i < d; ++i) {
            codes[i] = codebook_.encode(rotated[i]);
        }

        // Pack codes into output.
        size_t packed_size_per_vec = (static_cast<size_t>(d) * bits_ + 7) / 8;
        pack_codes(output_codes + v * packed_size_per_vec, codes.data(), d, bits_);
    }
}

// ─── Dequantize ─────────────────────────────────────────────────────────────

void TurboQuantKV::dequantize(float* output, const uint8_t* codes,
                               const float* norms, int n_vectors) const {
    const int d = head_dim_;
    std::vector<uint8_t> unpacked(d);
    std::vector<float> rotated(d);

    for (int v = 0; v < n_vectors; ++v) {
        size_t packed_size_per_vec = (static_cast<size_t>(d) * bits_ + 7) / 8;
        unpack_codes(unpacked.data(), codes + v * packed_size_per_vec, d, bits_);

        // Decode centroids.
        for (int i = 0; i < d; ++i) {
            rotated[i] = codebook_.decode(unpacked[i]);
        }

        // Inverse rotate.
        float* out_vec = output + v * d;
        rotate_inverse(out_vec, rotated.data());

        // Rescale by L2 norm.
        float norm = norms[v];
        for (int i = 0; i < d; ++i) {
            out_vec[i] *= norm;
        }
    }
}

// ─── KV page quantize / dequantize ─────────────────────────────────────────

void TurboQuantKV::quantize_kv_page(CompressedVectors& compressed,
                                     const float* keys_or_values,
                                     int n_tokens) const {
    const int d = head_dim_;
    size_t packed_per_vec = (static_cast<size_t>(d) * bits_ + 7) / 8;

    compressed.n_vectors = n_tokens;
    compressed.dim = d;
    compressed.bits = bits_;
    compressed.codes.resize(packed_per_vec * n_tokens);
    compressed.norms.resize(n_tokens);

    quantize(compressed.codes.data(), compressed.norms.data(),
             keys_or_values, n_tokens);
}

void TurboQuantKV::dequantize_kv_page(float* keys_or_values,
                                       const CompressedVectors& compressed,
                                       int n_tokens) const {
    assert(n_tokens <= compressed.n_vectors);
    dequantize(keys_or_values, compressed.codes.data(),
               compressed.norms.data(), n_tokens);
}

size_t TurboQuantKV::estimate_size(int n_vectors) const {
    size_t code_bits = static_cast<size_t>(n_vectors) * head_dim_ * bits_;
    size_t code_bytes = (code_bits + 7) / 8;
    size_t norm_bytes = static_cast<size_t>(n_vectors) * sizeof(float);
    return code_bytes + norm_bytes;
}

}  // namespace nexus::kv
