/// NXF Writer — Creates NXF (Nexus Format) files with chunk-aligned tensor data.
///
/// Write flow:
///   1. create() — open file, reserve space for header
///   2. set_manifest() — store manifest (written at finalize)
///   3. For each tensor: begin_tensor() -> add_chunk() x N -> end_tensor()
///   4. finalize() — write tensor index, manifest, then header, close

#include "format/nxf.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace nexus::format {

// ─── xxHash32 (same as reader) ──────────────────────────────────────────────

static constexpr uint32_t kXXH32_P1 = 0x9E3779B1u;
static constexpr uint32_t kXXH32_P2 = 0x85EBCA77u;
static constexpr uint32_t kXXH32_P3 = 0xC2B2AE3Du;
static constexpr uint32_t kXXH32_P4 = 0x27D4EB2Fu;
static constexpr uint32_t kXXH32_P5 = 0x165667B1u;

static inline uint32_t xxh32_rotl(uint32_t x, int r) {
    return (x << r) | (x >> (32 - r));
}

static uint32_t xxhash32(const void* data, size_t len, uint32_t seed = 0) {
    const auto* p = static_cast<const uint8_t*>(data);
    const uint8_t* end = p + len;
    uint32_t h;

    if (len >= 16) {
        const uint8_t* limit = end - 16;
        uint32_t v1 = seed + kXXH32_P1 + kXXH32_P2;
        uint32_t v2 = seed + kXXH32_P2;
        uint32_t v3 = seed;
        uint32_t v4 = seed - kXXH32_P1;
        do {
            uint32_t k;
            memcpy(&k, p, 4); v1 = xxh32_rotl(v1 + k * kXXH32_P2, 13) * kXXH32_P1; p += 4;
            memcpy(&k, p, 4); v2 = xxh32_rotl(v2 + k * kXXH32_P2, 13) * kXXH32_P1; p += 4;
            memcpy(&k, p, 4); v3 = xxh32_rotl(v3 + k * kXXH32_P2, 13) * kXXH32_P1; p += 4;
            memcpy(&k, p, 4); v4 = xxh32_rotl(v4 + k * kXXH32_P2, 13) * kXXH32_P1; p += 4;
        } while (p <= limit);
        h = xxh32_rotl(v1, 1) + xxh32_rotl(v2, 7) + xxh32_rotl(v3, 12) + xxh32_rotl(v4, 18);
    } else {
        h = seed + kXXH32_P5;
    }

    h += static_cast<uint32_t>(len);

    while (p + 4 <= end) {
        uint32_t k;
        memcpy(&k, p, 4);
        h = xxh32_rotl(h + k * kXXH32_P3, 17) * kXXH32_P4;
        p += 4;
    }
    while (p < end) {
        h = xxh32_rotl(h + (*p++) * kXXH32_P5, 11) * kXXH32_P1;
    }

    h ^= h >> 15; h *= kXXH32_P2;
    h ^= h >> 13; h *= kXXH32_P3;
    h ^= h >> 16;
    return h;
}

// ─── JSON string escaping ───────────────────────────────────────────────────

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

// ─── Manifest serialization to JSON ─────────────────────────────────────────

static std::string serialize_manifest(const ModelManifest& m) {
    char buf[4096];
    int n = snprintf(buf, sizeof(buf),
        "{\n"
        "  \"architecture\": \"%s\",\n"
        "  \"name\": \"%s\",\n"
        "  \"num_layers\": %u,\n"
        "  \"hidden_dim\": %u,\n"
        "  \"num_heads\": %u,\n"
        "  \"num_kv_heads\": %u,\n"
        "  \"head_dim\": %u,\n"
        "  \"vocab_size\": %u,\n"
        "  \"max_seq_len\": %u,\n"
        "  \"rope_theta\": %.1f,\n"
        "  \"rms_norm_eps\": %.10g,\n"
        "  \"num_experts\": %u,\n"
        "  \"num_active_experts\": %u,\n"
        "  \"default_codec\": %u,\n"
        "  \"default_group_size\": %u\n"
        "}",
        json_escape(m.architecture).c_str(),
        json_escape(m.name).c_str(),
        m.num_layers,
        m.hidden_dim,
        m.num_heads,
        m.num_kv_heads,
        m.head_dim,
        m.vocab_size,
        m.max_seq_len,
        static_cast<double>(m.rope_theta),
        static_cast<double>(m.rms_norm_eps),
        m.num_experts,
        m.num_active_experts,
        static_cast<unsigned>(m.default_codec),
        static_cast<unsigned>(m.default_group_size));
    return std::string(buf, static_cast<size_t>(n));
}

// ─── Tensor index serialization to binary ───────────────────────────────────

static std::vector<uint8_t> serialize_tensor_index(const std::vector<TensorInfo>& tensors) {
    std::vector<uint8_t> buf;
    // Pre-estimate capacity
    size_t est = 0;
    for (const auto& t : tensors) {
        est += 4 + t.name.size() + 4 + t.shape.size() * 8 + 1 + 4 + t.chunks.size() * sizeof(ChunkDesc);
    }
    buf.reserve(est);

    auto push_u32 = [&](uint32_t v) {
        size_t pos = buf.size();
        buf.resize(pos + 4);
        memcpy(buf.data() + pos, &v, 4);
    };
    auto push_u8 = [&](uint8_t v) {
        buf.push_back(v);
    };
    auto push_i64 = [&](int64_t v) {
        size_t pos = buf.size();
        buf.resize(pos + 8);
        memcpy(buf.data() + pos, &v, 8);
    };
    auto push_bytes = [&](const void* data, size_t len) {
        size_t pos = buf.size();
        buf.resize(pos + len);
        memcpy(buf.data() + pos, data, len);
    };

    for (const auto& t : tensors) {
        // [name_length:u32][name:chars]
        push_u32(static_cast<uint32_t>(t.name.size()));
        push_bytes(t.name.data(), t.name.size());

        // [ndim:u32][shape:i64*ndim]
        push_u32(static_cast<uint32_t>(t.shape.size()));
        for (int64_t dim : t.shape) {
            push_i64(dim);
        }

        // [dtype:u8]
        push_u8(static_cast<uint8_t>(t.dtype));

        // [num_chunks:u32][chunks:ChunkDesc*num_chunks]
        push_u32(static_cast<uint32_t>(t.chunks.size()));
        for (const auto& chunk : t.chunks) {
            push_bytes(&chunk, sizeof(ChunkDesc));
        }
    }

    return buf;
}

// ─── NXFWriter implementation ───────────────────────────────────────────────

NXFWriter::~NXFWriter() {
    if (fd_ >= 0) {
        // If finalize() was not called, leave the partial file on disk
        // so the user can inspect what was written. Don't truncate.
        fprintf(stderr, "[nxf] WARNING: writer destroyed without finalize() — partial file preserved\n");
        ::close(fd_);
        fd_ = -1;
    }
}

std::unique_ptr<NXFWriter> NXFWriter::create(const std::string& path) {
    std::unique_ptr<NXFWriter> writer(new NXFWriter());

    writer->fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (writer->fd_ < 0) {
        fprintf(stderr, "nxf: failed to create '%s': %s\n", path.c_str(), strerror(errno));
        return nullptr;
    }

    // Reserve space for the header (64 bytes). We'll write it at finalize().
    // Pad to chunk alignment so the first data chunk starts aligned.
    static_assert(sizeof(NXFHeader) <= kChunkAlignment,
                  "Header must fit in one alignment unit");

    // Write zeros for the header reservation (one full alignment unit)
    std::vector<uint8_t> zeros(kChunkAlignment, 0);
    ssize_t n = ::write(writer->fd_, zeros.data(), kChunkAlignment);
    if (n != static_cast<ssize_t>(kChunkAlignment)) {
        fprintf(stderr, "nxf: failed to reserve header space: %s\n", strerror(errno));
        ::close(writer->fd_);
        writer->fd_ = -1;
        return nullptr;
    }
    writer->write_pos_ = kChunkAlignment;

    return writer;
}

void NXFWriter::set_manifest(const ModelManifest& manifest) {
    manifest_ = manifest;
}

void NXFWriter::begin_tensor(const std::string& name, const std::vector<int64_t>& shape, DType dtype) {
    if (in_tensor_) {
        fprintf(stderr, "nxf: begin_tensor called while already in tensor '%s'\n",
                current_tensor_.name.c_str());
        return;
    }

    current_tensor_ = TensorInfo{};
    current_tensor_.name = name;
    current_tensor_.shape = shape;
    current_tensor_.dtype = dtype;
    current_tensor_.chunks.clear();
    in_tensor_ = true;
}

void NXFWriter::add_chunk(const void* data, size_t size, Codec codec, uint8_t group_size) {
    if (!in_tensor_) {
        fprintf(stderr, "nxf: add_chunk called outside of begin_tensor/end_tensor\n");
        return;
    }
    if (fd_ < 0 || !data || size == 0) return;

    // Ensure we're at chunk alignment before writing
    pad_to_alignment();

    // Compute checksum of the data
    uint32_t checksum = xxhash32(data, size);

    // Build chunk descriptor
    ChunkDesc desc{};
    desc.file_offset = write_pos_;
    desc.compressed_size = static_cast<uint32_t>(size);
    desc.decompressed_size = static_cast<uint32_t>(size); // No compression at write time
    desc.checksum = checksum;
    desc.codec = codec;
    desc.group_size = group_size;
    desc.reserved[0] = 0;
    desc.reserved[1] = 0;

    // Write data
    write_aligned(data, size);

    current_tensor_.chunks.push_back(desc);
}

void NXFWriter::end_tensor() {
    if (!in_tensor_) {
        fprintf(stderr, "nxf: end_tensor called without begin_tensor\n");
        return;
    }

    tensors_.push_back(std::move(current_tensor_));
    current_tensor_ = TensorInfo{};
    in_tensor_ = false;
}

void NXFWriter::finalize() {
    if (fd_ < 0) return;

    if (in_tensor_) {
        fprintf(stderr, "nxf: finalize called with unclosed tensor '%s', closing it\n",
                current_tensor_.name.c_str());
        end_tensor();
    }

    // Pad to alignment before writing metadata sections
    pad_to_alignment();

    // ─── Write tensor index ─────────────────────────────────────────────
    std::vector<uint8_t> index_blob = serialize_tensor_index(tensors_);
    uint64_t tensor_index_offset = write_pos_;
    write_aligned(index_blob.data(), index_blob.size());
    uint64_t tensor_index_size = index_blob.size();

    // Pad to alignment before manifest
    pad_to_alignment();

    // ─── Write manifest (JSON) ──────────────────────────────────────────
    std::string manifest_json = serialize_manifest(manifest_);
    uint64_t manifest_offset = write_pos_;
    write_aligned(manifest_json.data(), manifest_json.size());
    uint64_t manifest_size = manifest_json.size();

    // Pad to final alignment
    pad_to_alignment();

    uint64_t total_file_size = write_pos_;

    // ─── Write header at offset 0 ───────────────────────────────────────
    NXFHeader header{};
    header.magic = kNXFMagic;
    header.version = kNXFVersion;
    header.flags = 0;
    header.manifest_offset = manifest_offset;
    header.manifest_size = manifest_size;
    header.tensor_index_offset = tensor_index_offset;
    header.tensor_index_size = tensor_index_size;
    header.data_offset = kChunkAlignment; // First chunk starts right after header reservation
    header.total_file_size = total_file_size;
    memset(header.reserved, 0, sizeof(header.reserved));

    ssize_t n = ::pwrite(fd_, &header, sizeof(NXFHeader), 0);
    if (n != sizeof(NXFHeader)) {
        fprintf(stderr, "nxf: failed to write header: %s\n", strerror(errno));
    }

    // Close the file
    ::close(fd_);
    fd_ = -1;
}

// ─── Private helpers ────────────────────────────────────────────────────────

void NXFWriter::write_aligned(const void* data, size_t size) {
    if (fd_ < 0 || size == 0) return;

    const auto* p = static_cast<const uint8_t*>(data);
    size_t remaining = size;

    while (remaining > 0) {
        ssize_t n = ::write(fd_, p, remaining);
        if (n < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "nxf: write failed: %s\n", strerror(errno));
            return;
        }
        p += n;
        remaining -= static_cast<size_t>(n);
        write_pos_ += static_cast<uint64_t>(n);
    }
}

void NXFWriter::pad_to_alignment() {
    if (fd_ < 0) return;

    uint64_t remainder = write_pos_ % kChunkAlignment;
    if (remainder == 0) return;

    uint64_t padding = kChunkAlignment - remainder;

    // Write zeros for padding. Use a stack buffer for small padding,
    // which covers the common case since kChunkAlignment is 16 KB.
    uint8_t zeros[4096];
    memset(zeros, 0, sizeof(zeros));

    while (padding > 0) {
        size_t to_write = (padding > sizeof(zeros)) ? sizeof(zeros) : static_cast<size_t>(padding);
        ssize_t n = ::write(fd_, zeros, to_write);
        if (n < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "nxf: padding write failed: %s\n", strerror(errno));
            return;
        }
        padding -= static_cast<uint64_t>(n);
        write_pos_ += static_cast<uint64_t>(n);
    }
}

} // namespace nexus::format
