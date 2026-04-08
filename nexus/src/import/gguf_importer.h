#pragma once
/// NEXUS GGUF Importer — Converts llama.cpp GGUF model files to NXF format.
///
/// Parses the GGUF header, metadata key-value pairs, and tensor index,
/// then streams tensor data into an NXF file with optional re-quantization.
///
/// Supported workflows:
///   - F16/F32 source -> INT4 target (quantizes via quant_int4)
///   - Q4_0/Q4_1 source -> INT4 passthrough (repack block quant)
///   - Q8_0 source -> INT8 passthrough
///   - Any source -> FP16/FP32 passthrough (copy raw)

#include "core/config.h"
#include "format/nxf.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nexus::import {

// ─── GGUF on-disk constants ────────────────────────────────────────────────

constexpr uint32_t kGGUFMagic   = 0x46554747;  // "GGUF" as uint32 on little-endian (bytes: 47 47 55 46)
constexpr uint32_t kGGUFVersion = 3;

/// GGUF tensor type IDs (subset we handle).
enum class GGUFType : uint32_t {
    F32   = 0,
    F16   = 1,
    Q4_0  = 2,
    Q4_1  = 3,
    Q5_0  = 6,
    Q5_1  = 7,
    Q8_0  = 8,
    Q8_1  = 9,
    Q2_K  = 10,
    Q3_K  = 11,
    Q4_K  = 12,
    Q5_K  = 13,
    Q6_K  = 14,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8    = 24,
    I16   = 25,
    I32   = 26,
    I64   = 27,
    F64   = 28,
    IQ1_M = 29,
    BF16  = 30,
};

/// GGUF metadata value type tags.
enum class GGUFValueType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

// ─── Parsed structures ─────────────────────────────────────────────────────

/// One metadata key-value pair parsed from the GGUF header.
struct GGUFMetaKV {
    std::string key;
    GGUFValueType type;
    // Stored value — only one field is active based on `type`.
    uint64_t    val_uint  = 0;
    int64_t     val_int   = 0;
    double      val_float = 0.0;
    bool        val_bool  = false;
    std::string val_str;
    // Arrays are not fully parsed; we skip them during import.
};

/// One tensor descriptor from the GGUF tensor index.
struct GGUFTensorDesc {
    std::string            name;
    uint32_t               n_dims;
    std::vector<uint64_t>  dims;
    GGUFType               type;
    uint64_t               offset;  // Relative to data section start
};

/// Fully parsed GGUF file (metadata + tensor index, no tensor data loaded).
struct GGUFFile {
    uint32_t version      = 0;
    uint64_t tensor_count = 0;
    uint64_t meta_count   = 0;

    std::vector<GGUFMetaKV>    metadata;
    std::vector<GGUFTensorDesc> tensors;

    /// Absolute byte offset where tensor data begins in the file.
    uint64_t data_start   = 0;

    // Convenience accessors for metadata ─────────────────────────────────────

    /// Find a string metadata value by key. Returns empty string if absent.
    std::string meta_string(const std::string& key) const;

    /// Find a uint metadata value by key. Returns fallback if absent.
    uint64_t meta_uint(const std::string& key, uint64_t fallback = 0) const;

    /// Find a float metadata value by key. Returns fallback if absent.
    double meta_float(const std::string& key, double fallback = 0.0) const;
};

// ─── Importer ──────────────────────────────────────────────────────────────

class GGUFImporter {
public:
    /// Convert a GGUF file to NXF format.
    ///
    /// @param gguf_path   Path to the input .gguf file.
    /// @param nxf_path    Path for the output .nxf file.
    /// @param target_codec Desired weight codec (default INT4).
    /// @return true on success, false on error (details logged to stderr).
    static bool convert(const std::string& gguf_path,
                        const std::string& nxf_path,
                        Codec target_codec = Codec::INT4);

private:
    /// Parse the GGUF header, metadata, and tensor index from an open fd.
    static bool parse_gguf(int fd, GGUFFile& out);

    /// Build the NXF ModelManifest from parsed GGUF metadata.
    static format::ModelManifest build_manifest(const GGUFFile& gguf, Codec target_codec);

    /// Map a GGUF tensor name to the NEXUS naming convention.
    /// Returns empty string if the tensor should be skipped.
    static std::string map_tensor_name(const std::string& gguf_name);

    /// Return the byte size of one element (or one quant block) for a GGUF type.
    static size_t gguf_type_block_size(GGUFType type);

    /// Return the number of elements per quant block for a GGUF type.
    static size_t gguf_type_block_elements(GGUFType type);

    /// Choose the NXF codec + DType for a given GGUF type and target codec.
    static void choose_codec(GGUFType src_type, Codec target,
                             Codec& out_codec, DType& out_dtype);

    /// Compute total byte size for a tensor given its dims and GGUF type.
    static size_t tensor_data_size(const GGUFTensorDesc& desc);
};

}  // namespace nexus::import
