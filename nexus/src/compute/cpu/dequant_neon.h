#pragma once
/// NEXUS Inference Engine — NEON dequantization kernels for arm64.
///
/// Provides optimized routines for unpacking 4-bit and 8-bit quantized
/// weights to FP32, and a fused dequant + GEMV kernel for inference.

#include <cstddef>
#include <cstdint>

namespace nexus::compute {

// ─── Dequantization ─────────────────────────────────────────────────────────

/// Unpack INT4 values to FP32 with per-group scale and zero point.
///
/// Each byte in `data` holds two 4-bit values: low nibble = even index,
/// high nibble = odd index.
///
/// For element i in group g:  out[i] = (nibble - zeros[g]) * scales[g]
///
/// @param out        Output FP32 buffer, size n.
/// @param data       Packed INT4 data, size n/2 bytes.
/// @param scales     Per-group FP32 scales, size n/group_size.
/// @param zeros      Per-group FP32 zero points, size n/group_size.
/// @param n          Total number of elements (must be even).
/// @param group_size Group size for quantization (e.g. 128).
void dequant_int4_to_f32(float* out, const uint8_t* data,
                         const float* scales, const float* zeros,
                         int n, int group_size);

/// Dequantize INT8 values to FP32 with per-group scale (symmetric).
///
/// For element i in group g:  out[i] = data[i] * scales[g]
///
/// @param out        Output FP32 buffer, size n.
/// @param data       Signed INT8 data, size n.
/// @param scales     Per-group FP32 scales, size n/group_size.
/// @param n          Total number of elements.
/// @param group_size Group size for quantization (e.g. 128).
void dequant_int8_to_f32(float* out, const int8_t* data,
                         const float* scales,
                         int n, int group_size);

// ─── Fused dequant + GEMV ───────────────────────────────────────────────────

/// Fused INT4 dequantize and matrix-vector multiply.
///
/// Computes:  out[row] = sum_col( dequant(weight[row, col]) * x[col] )
///
/// This avoids materializing the full dequantized weight matrix.
///
/// @param out        Output vector, size rows.
/// @param weight     Packed INT4 weight matrix, size rows * (cols/2) bytes.
/// @param scales     Per-group scales, size rows * (cols/group_size).
/// @param zeros      Per-group zero points, same layout as scales.
/// @param x          Input vector, size cols.
/// @param rows       Number of output rows.
/// @param cols       Number of columns (input dimension, must be even).
/// @param group_size Quantization group size.
void dequant_int4_gemv(float* out, const uint8_t* weight,
                       const float* scales, const float* zeros,
                       const float* x,
                       int rows, int cols, int group_size);

}  // namespace nexus::compute
