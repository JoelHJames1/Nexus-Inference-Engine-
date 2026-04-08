/// NEXUS KV Quantize — Convenience wrapper implementation.

#include "kv/kv_quantize.h"
#include "kv/turbo_quant.h"
#include <cstring>

namespace nexus::kv {

void quantize_fp16_to_turbo(uint8_t* output, const float* input,
                            int dim, int n_vectors, int bits,
                            uint64_t seed) {
    TurboQuantKV quantizer(dim, bits, seed);

    size_t code_bits = static_cast<size_t>(n_vectors) * dim * bits;
    size_t code_bytes = (code_bits + 7) / 8;

    // Output layout: [packed_codes (code_bytes)] [norms (n_vectors * 4 bytes)]
    uint8_t* codes_ptr = output;
    float* norms_ptr = reinterpret_cast<float*>(output + code_bytes);

    quantizer.quantize(codes_ptr, norms_ptr, input, n_vectors);
}

void dequantize_turbo_to_fp32(float* output, const uint8_t* input,
                               int dim, int n_vectors, int bits,
                               uint64_t seed) {
    TurboQuantKV quantizer(dim, bits, seed);

    size_t code_bits = static_cast<size_t>(n_vectors) * dim * bits;
    size_t code_bytes = (code_bits + 7) / 8;

    const uint8_t* codes_ptr = input;
    const float* norms_ptr = reinterpret_cast<const float*>(input + code_bytes);

    quantizer.dequantize(output, codes_ptr, norms_ptr, n_vectors);
}

size_t estimate_compressed_size(int dim, int n_vectors, int bits) {
    size_t code_bits = static_cast<size_t>(n_vectors) * dim * bits;
    size_t code_bytes = (code_bits + 7) / 8;
    size_t norm_bytes = static_cast<size_t>(n_vectors) * sizeof(float);
    return code_bytes + norm_bytes;
}

}  // namespace nexus::kv
