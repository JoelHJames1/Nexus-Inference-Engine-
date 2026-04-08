#pragma once
/// NEXUS AWQ Codec — Activation-Aware Weight Quantization (4-bit).
///
/// Per-channel scaling based on activation importance.
/// Finds scale factors that minimize quantization error weighted
/// by activation magnitudes, then quantizes to INT4 with per-group
/// scales and zero points.

#include "core/config.h"
#include <cstdint>

namespace nexus::quant {

/// Quantize weights to 4-bit with activation-aware scaling.
///
/// @param output      Packed INT4 output (2 values per byte) [rows * cols / 2]
/// @param scales      Per-group scale factors [rows * (cols / group_size)]
/// @param zeros       Per-group zero points  [rows * (cols / group_size)]
/// @param weights     FP32 weight matrix, row-major [rows x cols]
/// @param activations FP32 activation magnitudes per column [cols]
/// @param rows        Number of output channels (rows)
/// @param cols        Number of input channels (cols)
/// @param group_size  Quantization group size (default 128)
void awq_quantize(uint8_t* output, float* scales, float* zeros,
                  const float* weights, const float* activations,
                  int rows, int cols, int group_size = 128);

/// Dequantize AWQ 4-bit packed data back to FP32.
/// Same layout as GPTQ dequant but operates on AWQ-computed scales/zeros.
///
/// @param out        Output FP32 buffer [n]
/// @param data       Packed INT4 data (2 values per byte) [n / 2]
/// @param scales     Per-group scale factors [n / group_size]
/// @param zeros      Per-group zero points [n / group_size]
/// @param n          Number of elements
/// @param group_size Quantization group size (default 128)
void awq_dequantize(float* out, const uint8_t* data, const float* scales,
                    const float* zeros, int n, int group_size = 128);

}  // namespace nexus::quant
