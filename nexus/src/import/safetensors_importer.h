#pragma once
/// NEXUS Safetensors Importer — Converts HuggingFace safetensors models to NXF format.
///
/// Parses the safetensors binary header (8-byte size + JSON metadata), extracts
/// tensor descriptors (name, dtype, shape, data_offsets), and streams tensor data
/// into an NXF file with optional re-quantization.
///
/// Supports both single-file and multi-shard models:
///   - Single file:  input_path = "model.safetensors"
///   - Multi-shard:  input_path = directory containing model-00001-of-00003.safetensors, etc.
///
/// When input_path is a directory, also reads config.json for model architecture metadata.
///
/// Supported dtype conversions:
///   - F16/BF16/F32 source -> INT4 target (quantizes via quant_int4)
///   - F16/BF16/F32 source -> INT8 target (per-group symmetric)
///   - F16/BF16/F32 source -> FP16/FP32 passthrough

#include "core/config.h"
#include "format/nxf.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nexus::import {

// ─── Safetensors dtype identifiers ────────────────────────────────────────

enum class SafetensorsDType {
    F16,
    F32,
    BF16,
    I8,
    I32,
    I64,
    U8,
    BOOL,
    UNKNOWN,
};

// ─── Parsed structures ─────────────────────────────────────────────────────

/// One tensor descriptor parsed from the safetensors JSON header.
struct SafetensorsTensorDesc {
    std::string              name;
    SafetensorsDType         dtype;
    std::vector<int64_t>     shape;
    uint64_t                 data_start;  // Byte offset relative to end of JSON header
    uint64_t                 data_end;    // Byte offset relative to end of JSON header
};

/// Parsed model configuration from config.json.
struct HFModelConfig {
    std::string architecture;     // e.g., "llama"
    std::string model_type;       // e.g., "llama"
    uint32_t hidden_size          = 0;
    uint32_t num_hidden_layers    = 0;
    uint32_t num_attention_heads  = 0;
    uint32_t num_key_value_heads  = 0;
    uint32_t vocab_size           = 0;
    uint32_t max_position_embeddings = 4096;
    float    rope_theta           = 10000.0f;
    float    rms_norm_eps         = 1e-5f;
    bool     valid                = false;
};

/// A single safetensors shard with its parsed tensor index.
struct SafetensorsShard {
    std::string                       path;
    uint64_t                          header_size;     // JSON header size (from 8-byte prefix)
    uint64_t                          data_base;       // Absolute byte offset where tensor data starts
    std::vector<SafetensorsTensorDesc> tensors;
};

// ─── Importer ──────────────────────────────────────────────────────────────

class SafetensorsImporter {
public:
    /// Convert safetensors file(s) to NXF format.
    ///
    /// @param input_path  Path to a .safetensors file, or a directory with shards + config.json.
    /// @param nxf_path    Path for the output .nxf file.
    /// @param target_codec Desired weight codec (default INT4).
    /// @return true on success, false on error (details logged to stderr).
    static bool convert(const std::string& input_path,
                        const std::string& nxf_path,
                        Codec target_codec = Codec::INT4);

private:
    /// Parse the JSON header of a single safetensors file.
    static bool parse_shard(const std::string& path, SafetensorsShard& out);

    /// Parse config.json from a directory to extract model architecture info.
    static HFModelConfig parse_config_json(const std::string& dir_path);

    /// Build the NXF ModelManifest from a parsed HFModelConfig.
    static format::ModelManifest build_manifest(const HFModelConfig& config, Codec target_codec);

    /// Build a default ModelManifest by inspecting tensor shapes (fallback).
    static format::ModelManifest infer_manifest(const std::vector<SafetensorsShard>& shards,
                                                Codec target_codec);

    /// Map a HuggingFace tensor name to the NEXUS naming convention.
    /// Returns empty string if the tensor should be skipped.
    static std::string map_tensor_name(const std::string& hf_name);

    /// Convert a safetensors dtype string to SafetensorsDType enum.
    static SafetensorsDType parse_dtype(const std::string& dtype_str);

    /// Return the byte size per element for a safetensors dtype.
    static size_t dtype_element_size(SafetensorsDType dtype);

    /// Choose the NXF codec + DType for a given source dtype and target codec.
    static void choose_codec(SafetensorsDType src_dtype, Codec target,
                             Codec& out_codec, DType& out_dtype);

    /// Discover all .safetensors shard files in a directory, sorted by name.
    static std::vector<std::string> discover_shards(const std::string& dir_path);
};

}  // namespace nexus::import
