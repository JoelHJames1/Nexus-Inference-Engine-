#pragma once
/// NEXUS TurboQuant — KV cache quantization via random rotation + Lloyd-Max coding.
///
/// Phase 3: Core differentiator from llama.cpp.
/// Algorithm:
///   1. Random rotation: y = Pi * x  (Pi is random orthogonal via QR of Gaussian matrix)
///   2. Each coordinate of rotated vector follows Beta distribution
///      -> use precomputed Lloyd-Max codebook
///   3. Quantize each coordinate to b bits using nearest centroid
///   4. Store L2 norm for rescaling on dequantization

#include "core/config.h"
#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>

namespace nexus::kv {

/// Precomputed Lloyd-Max codebook for a given bit width.
struct LloydMaxCodebook {
    int bits;                         // 2, 3, or 4
    int n_centroids;                  // 2^bits
    std::vector<float> centroids;     // [n_centroids] — reconstruction levels
    std::vector<float> boundaries;    // [n_centroids - 1] — decision boundaries

    /// Find the nearest centroid index for a scalar value.
    uint8_t encode(float x) const;

    /// Decode a centroid index back to a scalar value.
    float decode(uint8_t code) const;
};

/// Compressed representation of a batch of vectors.
/// Layout: [packed_codes (ceil(n_vectors * dim * bits / 8) bytes)] [norms (n_vectors * 4 bytes)]
struct CompressedVectors {
    std::vector<uint8_t> codes;       // Packed bit codes
    std::vector<float> norms;         // L2 norms per vector
    int n_vectors;
    int dim;
    int bits;

    /// Byte size of the compressed representation.
    size_t byte_size() const;
};

/// TurboQuant KV cache quantizer.
///
/// Generates a random orthogonal rotation matrix and precomputes
/// Lloyd-Max codebooks for fast scalar quantization of rotated KV vectors.
class TurboQuantKV {
public:
    /// @param head_dim  Dimension of each KV head vector
    /// @param bits      Quantization bit width (2, 3, or 4)
    /// @param seed      RNG seed for rotation matrix generation
    TurboQuantKV(int head_dim, int bits = 4, uint64_t seed = 42);
    ~TurboQuantKV();

    /// Quantize a batch of vectors.
    ///
    /// @param output_codes  Packed bit codes (caller must allocate, or use CompressedVectors)
    /// @param output_norms  L2 norm per vector [n_vectors]
    /// @param input         Input FP32 vectors [n_vectors * head_dim], row-major
    /// @param n_vectors     Number of vectors to quantize
    void quantize(uint8_t* output_codes, float* output_norms,
                  const float* input, int n_vectors) const;

    /// Dequantize a batch of vectors.
    ///
    /// @param output    Output FP32 vectors [n_vectors * head_dim], row-major
    /// @param codes     Packed bit codes
    /// @param norms     L2 norm per vector [n_vectors]
    /// @param n_vectors Number of vectors
    void dequantize(float* output, const uint8_t* codes,
                    const float* norms, int n_vectors) const;

    /// Quantize a page of KV vectors into a CompressedVectors struct.
    ///
    /// @param compressed     Output compressed page
    /// @param keys_or_values Input FP32 vectors [n_tokens * head_dim]
    /// @param n_tokens       Number of token vectors in this page
    void quantize_kv_page(CompressedVectors& compressed,
                          const float* keys_or_values, int n_tokens) const;

    /// Dequantize a compressed page back to FP32.
    ///
    /// @param keys_or_values Output FP32 vectors [n_tokens * head_dim]
    /// @param compressed     Compressed page
    /// @param n_tokens       Number of token vectors to decompress
    void dequantize_kv_page(float* keys_or_values,
                            const CompressedVectors& compressed,
                            int n_tokens) const;

    int head_dim() const { return head_dim_; }
    int bits() const { return bits_; }

    /// Estimate compressed byte size for n_vectors of dimension head_dim_.
    size_t estimate_size(int n_vectors) const;

private:
    int head_dim_;
    int bits_;
    uint64_t seed_;

    /// Random orthogonal rotation matrix Pi [head_dim_ x head_dim_], row-major.
    std::vector<float> rotation_;

    /// Transpose of rotation matrix (Pi^T = Pi^{-1} for orthogonal matrices).
    std::vector<float> rotation_t_;

    /// Lloyd-Max codebook for the configured bit width.
    LloydMaxCodebook codebook_;

    /// Generate the random orthogonal rotation matrix via QR decomposition.
    void generate_rotation_matrix();

    /// Build Lloyd-Max codebook for the target distribution.
    void build_codebook();

    /// Apply rotation: out = Pi * in (single vector).
    void rotate(float* out, const float* in) const;

    /// Apply inverse rotation: out = Pi^T * in (single vector).
    void rotate_inverse(float* out, const float* in) const;

    /// Pack a sequence of b-bit codes into a byte array.
    static void pack_codes(uint8_t* packed, const uint8_t* codes, int n_codes, int bits);

    /// Unpack a byte array into b-bit codes.
    static void unpack_codes(uint8_t* codes, const uint8_t* packed, int n_codes, int bits);
};

}  // namespace nexus::kv
