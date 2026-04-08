#pragma once
/// NXF (Nexus Format) — Streaming-native tensor container for Apple Silicon.
///
/// Unlike GGUF (load-and-go) or safetensors (flat memory-map), NXF is designed
/// for chunked streaming: each tensor is split into page-aligned chunks with
/// independent codecs, enabling layer-by-layer weight streaming from SSD.
///
/// Key design choices for Apple Silicon:
///   - 16 KB chunk alignment (arm64 macOS VM page size, not x86 4KB)
///   - Per-chunk codec (mixed precision within a single tensor)
///   - Entropy coding support (ANS lossless post-quant compression)
///   - MoE routing metadata in manifest
///   - KV sidecar section for persistent prefix caches

#include "core/config.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace nexus::format {

// ─── On-disk structures ─────────────────────────────────────────────────────

/// NXF file header (fixed 64 bytes at offset 0).
struct NXFHeader {
    uint32_t magic;              // kNXFMagic = "NXF1"
    uint16_t version;            // kNXFVersion = 1
    uint16_t flags;              // Reserved
    uint64_t manifest_offset;    // Byte offset to JSON/FlatBuffers manifest
    uint64_t manifest_size;      // Size of manifest blob
    uint64_t tensor_index_offset;// Byte offset to tensor index
    uint64_t tensor_index_size;  // Size of tensor index blob
    uint64_t data_offset;        // Byte offset to first chunk data
    uint64_t total_file_size;    // Total file size for integrity check
    uint8_t  reserved[8];        // Padding to 64 bytes
};
static_assert(sizeof(NXFHeader) == 64, "NXFHeader must be exactly 64 bytes");

/// Describes one chunk of a tensor stored on disk.
struct ChunkDesc {
    uint64_t file_offset;        // Byte offset in NXF file
    uint32_t compressed_size;    // Size on disk (after codec)
    uint32_t decompressed_size;  // Size after decompression
    uint32_t checksum;           // xxHash32 of compressed data
    Codec    codec;              // Compression/quantization codec
    uint8_t  group_size;         // Quantization group size (e.g., 128 for INT4)
    uint8_t  reserved[2];        // Padding
};
static_assert(sizeof(ChunkDesc) == 24, "ChunkDesc must be 24 bytes");

/// Metadata for a single tensor in the model.
struct TensorInfo {
    std::string name;                // e.g., "layers.0.attention.wq.weight"
    std::vector<int64_t> shape;      // e.g., {4096, 4096}
    DType dtype;                     // Original data type
    std::vector<ChunkDesc> chunks;   // Ordered list of chunks

    /// Total compressed size on disk.
    uint64_t compressed_bytes() const {
        uint64_t total = 0;
        for (const auto& c : chunks) total += c.compressed_size;
        return total;
    }

    /// Total decompressed size in memory.
    uint64_t decompressed_bytes() const {
        uint64_t total = 0;
        for (const auto& c : chunks) total += c.decompressed_size;
        return total;
    }
};

/// Model architecture metadata from the manifest.
struct ModelManifest {
    std::string architecture;     // e.g., "llama", "deepseek_v3"
    std::string name;             // e.g., "LLaMA-3.1-405B"
    uint32_t num_layers;
    uint32_t hidden_dim;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t vocab_size;
    uint32_t max_seq_len;
    float    rope_theta;
    float    rms_norm_eps;

    // MoE fields (0 if dense model)
    uint32_t num_experts;
    uint32_t num_active_experts;

    // Quantization metadata
    Codec    default_codec;
    uint8_t  default_group_size;
};

// ─── NXF Reader ─────────────────────────────────────────────────────────────

class NXFReader {
public:
    ~NXFReader();

    /// Open an NXF file. Reads header, manifest, and tensor index.
    /// Does NOT load tensor data — that's done lazily via map_chunk().
    static std::unique_ptr<NXFReader> open(const std::string& path);

    /// Get model manifest.
    const ModelManifest& manifest() const { return manifest_; }

    /// Get tensor info by name. Returns nullptr if not found.
    const TensorInfo* get_tensor(const std::string& name) const;

    /// Get all tensor names.
    std::vector<std::string> tensor_names() const;

    /// Memory-map a chunk region. Returns pointer to compressed data.
    /// The pointer is valid until unmap_chunk() or destructor.
    const void* map_chunk(const ChunkDesc& desc);

    /// Unmap a previously mapped chunk.
    void unmap_chunk(const void* ptr);

    /// Unmap all chunks and close the file.
    void close();

    /// File descriptor (for advanced I/O like fcntl F_RDADVISE).
    int fd() const { return fd_; }

    /// Total file size.
    uint64_t file_size() const { return header_.total_file_size; }

private:
    NXFReader() = default;

    int fd_ = -1;
    NXFHeader header_{};
    ModelManifest manifest_{};
    std::unordered_map<std::string, TensorInfo> tensors_;

    // Memory-mapped regions tracking
    struct MappedRegion {
        void*  base;
        size_t length;
    };
    std::vector<MappedRegion> mapped_regions_;

    bool read_header();
    bool read_manifest();
    bool read_tensor_index();
};

// ─── NXF Writer ─────────────────────────────────────────────────────────────

class NXFWriter {
public:
    ~NXFWriter();

    /// Create a new NXF file for writing.
    static std::unique_ptr<NXFWriter> create(const std::string& path);

    /// Set model manifest.
    void set_manifest(const ModelManifest& manifest);

    /// Begin writing a tensor. Call add_chunk() for each chunk, then end_tensor().
    void begin_tensor(const std::string& name, const std::vector<int64_t>& shape, DType dtype);

    /// Write a chunk of tensor data with the specified codec.
    /// Data is written aligned to kChunkAlignment (16 KB).
    void add_chunk(const void* data, size_t size, Codec codec, uint8_t group_size = 128);

    /// Finish the current tensor.
    void end_tensor();

    /// Finalize the file: write tensor index and header, then close.
    void finalize();

private:
    NXFWriter() = default;

    int fd_ = -1;
    uint64_t write_pos_ = 0;
    ModelManifest manifest_{};
    std::vector<TensorInfo> tensors_;
    TensorInfo current_tensor_{};
    bool in_tensor_ = false;

    void write_aligned(const void* data, size_t size);
    void pad_to_alignment();
};

}  // namespace nexus::format
