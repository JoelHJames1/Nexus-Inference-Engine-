#pragma once
/// NEXUS QuIP# Codec — 3-bit weight quantization with random rotation.
///
/// Simplified QuIP-Sharp: random orthogonal rotation + 3-bit Lloyd-Max
/// quantization with per-group scales. Targets ~3 bits per weight.

#include "core/config.h"
#include <cstdint>
#include <vector>

namespace nexus::quant {

/// Quantize FP32 weights to 3-bit packed format with per-group scales.
///
/// @param out        Packed 3-bit output (8 values -> 3 bytes). Size: (n / 8) * 3
/// @param scales     Per-group scale factors [n / group_size]
/// @param input      FP32 weight data [n]
/// @param n          Number of elements (must be multiple of 8)
/// @param group_size Quantization group size (default 256)
void quip3_quantize(uint8_t* out, float* scales,
                    const float* input, int n, int group_size = 256);

/// Dequantize 3-bit packed data back to FP32.
///
/// @param out        Output FP32 buffer [n]
/// @param data       Packed 3-bit data [(n / 8) * 3]
/// @param scales     Per-group scale factors [n / group_size]
/// @param n          Number of elements (must be multiple of 8)
/// @param group_size Quantization group size (default 256)
void quip3_dequantize(float* out, const uint8_t* data, const float* scales,
                      int n, int group_size = 256);

/// Generate a random orthogonal rotation matrix (dim x dim, row-major).
/// Uses QR decomposition of a random Gaussian matrix seeded by `seed`.
///
/// @param dim   Matrix dimension
/// @param seed  RNG seed for reproducibility
/// @return      Row-major dim*dim orthogonal matrix
std::vector<float> generate_rotation(int dim, uint64_t seed);

}  // namespace nexus::quant
