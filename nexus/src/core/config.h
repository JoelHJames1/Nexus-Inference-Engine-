#pragma once
/// NEXUS Inference Engine — Global configuration and constants.

#include <cstddef>
#include <cstdint>

namespace nexus {

// ─── Platform constants (Apple Silicon) ─────────────────────────────────────
constexpr size_t kPageSize         = 16 * 1024;        // 16 KB (arm64 macOS VM page)
constexpr size_t kChunkAlignment   = kPageSize;         // NXF chunks aligned to VM page
constexpr size_t kWeightSlabSize   = 1 * 1024 * 1024;  // 1 MB slabs for weight chunks
constexpr size_t kKVPageSize       = 64 * 1024;         // 64 KB slabs for KV pages
constexpr size_t kDefaultRAMLimit  = 48ULL * 1024 * 1024 * 1024;  // 48 GB default cap

// ─── NXF format constants ───────────────────────────────────────────────────
constexpr uint32_t kNXFMagic       = 0x3146584E;        // "NXF1" in little-endian
constexpr uint16_t kNXFVersion     = 1;

// ─── Codec identifiers ─────────────────────────────────────────────────────
enum class Codec : uint8_t {
    FP32     = 0,
    FP16     = 1,
    BF16     = 2,
    INT8     = 3,
    INT4     = 4,     // Block quantized, group_size=128
    GPTQ     = 5,     // GPTQ per-group 4-bit with scales/zeros
    AWQ      = 6,     // Activation-aware weight quantization
    QUIP3    = 7,     // QuIP# 3-bit E8 lattice
    AQLM2    = 8,     // AQLM 2-bit additive codebooks
    ANS      = 9,     // Entropy coded (post-quant lossless)
    TURBO_Q  = 10,    // TurboQuant (KV cache)
    Q3K      = 11,    // Q3_K native passthrough (3.4 bits, no conversion)
    Q4K      = 12,    // Q4_K native passthrough (4.5 bits)
    Q2K      = 13,    // Q2_K native passthrough (2.6 bits)
};

// ─── Data types ─────────────────────────────────────────────────────────────
enum class DType : uint8_t {
    F32  = 0,
    F16  = 1,
    BF16 = 2,
    I8   = 3,
    I4   = 4,
    I2   = 5,
};

/// Returns the byte size per element for a given dtype (0 for sub-byte types).
constexpr size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::F32:  return 4;
        case DType::F16:  return 2;
        case DType::BF16: return 2;
        case DType::I8:   return 1;
        default:          return 0;  // Sub-byte: handled by codec
    }
}

// ─── Sampling parameters ────────────────────────────────────────────────────
struct SamplingParams {
    float temperature  = 0.7f;
    float top_p        = 0.9f;
    int   top_k        = 40;
    int   max_tokens   = 2048;
    uint64_t seed      = 0;        // 0 = random
};

// ─── Memory budget ──────────────────────────────────────────────────────────
struct MemoryConfig {
    size_t ram_limit         = kDefaultRAMLimit;
    size_t weight_buffer_mb  = 8192;   // 8 GB for double-buffered weight streaming
    size_t kv_hot_mb         = 2048;   // 2 GB for FP16 KV
    size_t kv_warm_mb        = 6144;   // 6 GB for TurboQuant 3.5-bit KV
    size_t kv_cool_mb        = 4096;   // 4 GB for TurboQuant 2.5-bit KV
    size_t scratch_mb        = 4096;   // 4 GB for activations/intermediates
};

}  // namespace nexus
