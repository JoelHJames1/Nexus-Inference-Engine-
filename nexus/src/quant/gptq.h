#pragma once
/// NEXUS GPTQ Codec — 4-bit weight quantization with per-group scales/zeros.
///
/// Phase 1: Basic dequantization (INT4 → FP32).
/// Phase 2: Full GPTQ calibration import + QuIP# + AQLM.

#include "core/config.h"
#include <cstdint>

namespace nexus::quant {

/// Dequantize INT4 block-quantized weights to FP32.
/// Layout: packed pairs of 4-bit values in uint8, with per-group scales and zeros.
///
/// @param out       Output FP32 buffer [num_elements]
/// @param data      Packed INT4 data (2 values per byte) [num_elements / 2]
/// @param scales    Per-group scale factors [num_elements / group_size]
/// @param zeros     Per-group zero points [num_elements / group_size]
/// @param n         Number of elements
/// @param group_size Quantization group size (default 128)
void dequant_int4(float* out, const uint8_t* data, const float* scales,
                  const float* zeros, int n, int group_size = 128);

/// Dequantize INT8 quantized weights to FP32.
void dequant_int8(float* out, const int8_t* data, const float* scales,
                  int n, int group_size = 128);

/// Quantize FP32 weights to INT4 with per-group scales/zeros (for NXF export).
/// Returns packed data. Caller owns the output buffer.
void quant_int4(uint8_t* out, float* scales_out, float* zeros_out,
                const float* data, int n, int group_size = 128);

}  // namespace nexus::quant
