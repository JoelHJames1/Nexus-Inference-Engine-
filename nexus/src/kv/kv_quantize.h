#pragma once
/// NEXUS KV Quantize — Convenience wrappers for TurboQuant KV compression.
///
/// Phase 3: Utility functions for quantizing/dequantizing KV cache vectors.

#include "core/config.h"
#include <cstdint>
#include <cstddef>

namespace nexus::kv {

/// Quantize FP32 vectors to TurboQuant compressed format.
///
/// @param output    Output buffer (must be at least estimate_compressed_size() bytes)
/// @param input     Input FP32 vectors [n_vectors * dim]
/// @param dim       Vector dimension (head_dim)
/// @param n_vectors Number of vectors
/// @param bits      Quantization bit width (2, 3, or 4)
/// @param seed      RNG seed for rotation matrix (default 42)
void quantize_fp16_to_turbo(uint8_t* output, const float* input,
                            int dim, int n_vectors, int bits,
                            uint64_t seed = 42);

/// Dequantize TurboQuant compressed format back to FP32.
///
/// @param output    Output FP32 vectors [n_vectors * dim]
/// @param input     Compressed data (codes followed by norms)
/// @param dim       Vector dimension (head_dim)
/// @param n_vectors Number of vectors
/// @param bits      Quantization bit width (2, 3, or 4)
/// @param seed      RNG seed (must match quantize call)
void dequantize_turbo_to_fp32(float* output, const uint8_t* input,
                               int dim, int n_vectors, int bits,
                               uint64_t seed = 42);

/// Estimate the compressed byte size for a batch of vectors.
///
/// @param dim       Vector dimension
/// @param n_vectors Number of vectors
/// @param bits      Quantization bit width (2, 3, or 4)
/// @return          Total bytes needed (codes + norms)
size_t estimate_compressed_size(int dim, int n_vectors, int bits);

}  // namespace nexus::kv
