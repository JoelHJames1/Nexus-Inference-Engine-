#pragma once
/// NEXUS ANS Entropy Codec — lossless compression for quantized weight data.
///
/// Simplified tANS (tabled Asymmetric Numeral Systems) implementation.
/// Applied after quantization to squeeze out remaining redundancy.
/// Targets 10-20% additional compression on quantized weight bytes.

#include <cstddef>
#include <cstdint>

namespace nexus::quant {

/// Compress bytes using tabled ANS.
///
/// @param output      Output buffer (must be at least input_size bytes)
/// @param output_size Available space in output buffer
/// @param input       Input data to compress
/// @param input_size  Number of input bytes
/// @return            Compressed size in bytes, or 0 on failure
size_t ans_compress(uint8_t* output, size_t output_size,
                    const uint8_t* input, size_t input_size);

/// Decompress ANS-compressed data back to original bytes.
///
/// @param output      Output buffer [output_size]
/// @param output_size Expected decompressed size
/// @param input       Compressed data
/// @param input_size  Compressed data size
/// @return            Decompressed size, or 0 on failure
size_t ans_decompress(uint8_t* output, size_t output_size,
                      const uint8_t* input, size_t input_size);

/// Estimate compressed size without actually compressing.
/// Uses Shannon entropy of input byte distribution.
///
/// @param input       Input data
/// @param input_size  Number of input bytes
/// @return            Estimated compressed size in bytes
size_t ans_compressed_size(const uint8_t* input, size_t input_size);

}  // namespace nexus::quant
