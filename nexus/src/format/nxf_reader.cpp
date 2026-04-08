/// NXF Reader — Reads NXF (Nexus Format) files with lazy chunk mapping.
///
/// Uses POSIX I/O and mmap for zero-copy chunk access on Apple Silicon.

#include "format/nxf.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace nexus::format {

// ─── Simple xxHash32-like checksum ──────────────────────────────────────────
// A fast, non-cryptographic 32-bit hash. This is a simplified version of
// xxHash32 using the same prime constants and mixing strategy.

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

// ─── Minimal JSON parser for ModelManifest ──────────────────────────────────
// Parses a flat JSON object with string and numeric values. No nested objects,
// no arrays, no escapes beyond \" — sufficient for NXF manifests.

namespace json {

/// Skip whitespace, return pointer to next non-whitespace char.
static const char* skip_ws(const char* p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    return p;
}

/// Parse a JSON string (assumes p points at opening '"'). Returns pointer past closing '"'.
/// Writes the string content into `out`.
static const char* parse_string(const char* p, const char* end, std::string& out) {
    out.clear();
    if (p >= end || *p != '"') return nullptr;
    ++p; // skip opening quote
    while (p < end && *p != '"') {
        if (*p == '\\' && p + 1 < end) {
            ++p;
            switch (*p) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'n':  out += '\n'; break;
                case 't':  out += '\t'; break;
                default:   out += *p;   break;
            }
        } else {
            out += *p;
        }
        ++p;
    }
    if (p >= end) return nullptr;
    return p + 1; // skip closing quote
}

/// Parse a JSON number (integer or float). Returns pointer past the number.
static const char* parse_number(const char* p, const char* end, double& out) {
    char buf[64];
    size_t i = 0;
    while (p < end && i < sizeof(buf) - 1 &&
           (*p == '-' || *p == '+' || *p == '.' || (*p >= '0' && *p <= '9') || *p == 'e' || *p == 'E')) {
        buf[i++] = *p++;
    }
    buf[i] = '\0';
    out = strtod(buf, nullptr);
    return p;
}

/// Parse manifest JSON into a ModelManifest struct.
static bool parse_manifest(const char* data, size_t len, ModelManifest& m) {
    const char* p = data;
    const char* end = data + len;

    p = skip_ws(p, end);
    if (p >= end || *p != '{') return false;
    ++p;

    while (true) {
        p = skip_ws(p, end);
        if (p >= end) return false;
        if (*p == '}') break;

        // Expect comma between entries (skip it)
        if (*p == ',') { ++p; p = skip_ws(p, end); }
        if (p >= end) return false;
        if (*p == '}') break;

        // Parse key
        std::string key;
        p = parse_string(p, end, key);
        if (!p) return false;

        // Expect colon
        p = skip_ws(p, end);
        if (p >= end || *p != ':') return false;
        ++p;
        p = skip_ws(p, end);
        if (p >= end) return false;

        // Parse value — string or number
        if (*p == '"') {
            std::string val;
            p = parse_string(p, end, val);
            if (!p) return false;
            if      (key == "architecture") m.architecture = val;
            else if (key == "name")         m.name = val;
            else if (key == "default_codec") m.default_codec = static_cast<Codec>(atoi(val.c_str()));
        } else {
            double val;
            p = parse_number(p, end, val);
            if      (key == "num_layers")          m.num_layers = static_cast<uint32_t>(val);
            else if (key == "hidden_dim")          m.hidden_dim = static_cast<uint32_t>(val);
            else if (key == "num_heads")           m.num_heads = static_cast<uint32_t>(val);
            else if (key == "num_kv_heads")        m.num_kv_heads = static_cast<uint32_t>(val);
            else if (key == "head_dim")            m.head_dim = static_cast<uint32_t>(val);
            else if (key == "vocab_size")          m.vocab_size = static_cast<uint32_t>(val);
            else if (key == "max_seq_len")         m.max_seq_len = static_cast<uint32_t>(val);
            else if (key == "rope_theta")          m.rope_theta = static_cast<float>(val);
            else if (key == "rms_norm_eps")        m.rms_norm_eps = static_cast<float>(val);
            else if (key == "num_experts")         m.num_experts = static_cast<uint32_t>(val);
            else if (key == "num_active_experts")  m.num_active_experts = static_cast<uint32_t>(val);
            else if (key == "default_codec")       m.default_codec = static_cast<Codec>(static_cast<uint8_t>(val));
            else if (key == "default_group_size")  m.default_group_size = static_cast<uint8_t>(val);
        }
    }
    return true;
}

} // namespace json

// ─── NXFReader implementation ───────────────────────────────────────────────

NXFReader::~NXFReader() {
    close();
}

std::unique_ptr<NXFReader> NXFReader::open(const std::string& path) {
    std::unique_ptr<NXFReader> reader(new NXFReader());

    reader->fd_ = ::open(path.c_str(), O_RDONLY);
    if (reader->fd_ < 0) {
        fprintf(stderr, "nxf: failed to open '%s': %s\n", path.c_str(), strerror(errno));
        return nullptr;
    }

    if (!reader->read_header()) {
        fprintf(stderr, "nxf: invalid header in '%s'\n", path.c_str());
        reader->close();
        return nullptr;
    }

    if (!reader->read_manifest()) {
        fprintf(stderr, "nxf: failed to read manifest in '%s'\n", path.c_str());
        reader->close();
        return nullptr;
    }

    if (!reader->read_tensor_index()) {
        fprintf(stderr, "nxf: failed to read tensor index in '%s'\n", path.c_str());
        reader->close();
        return nullptr;
    }

    return reader;
}

const TensorInfo* NXFReader::get_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return nullptr;
    return &it->second;
}

std::vector<std::string> NXFReader::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& kv : tensors_) {
        names.push_back(kv.first);
    }
    return names;
}

const void* NXFReader::map_chunk(const ChunkDesc& desc) {
    if (fd_ < 0) return nullptr;

    // mmap requires page-aligned offset. Compute the aligned base.
    size_t page_size = static_cast<size_t>(kChunkAlignment);
    uint64_t aligned_offset = (desc.file_offset / page_size) * page_size;
    size_t offset_within_page = static_cast<size_t>(desc.file_offset - aligned_offset);
    size_t map_length = offset_within_page + desc.compressed_size;

    void* base = ::mmap(nullptr, map_length, PROT_READ, MAP_PRIVATE, fd_,
                        static_cast<off_t>(aligned_offset));
    if (base == MAP_FAILED) {
        fprintf(stderr, "nxf: mmap failed at offset %llu size %zu: %s\n",
                (unsigned long long)aligned_offset, map_length, strerror(errno));
        return nullptr;
    }

    mapped_regions_.push_back({base, map_length});

    // Return pointer to the actual data within the mapped region
    return static_cast<const uint8_t*>(base) + offset_within_page;
}

void NXFReader::unmap_chunk(const void* ptr) {
    if (!ptr) return;

    for (auto it = mapped_regions_.begin(); it != mapped_regions_.end(); ++it) {
        // Check if ptr falls within this mapped region
        const auto* base_byte = static_cast<const uint8_t*>(it->base);
        const auto* ptr_byte = static_cast<const uint8_t*>(ptr);
        if (ptr_byte >= base_byte && ptr_byte < base_byte + it->length) {
            ::munmap(it->base, it->length);
            mapped_regions_.erase(it);
            return;
        }
    }

    fprintf(stderr, "nxf: unmap_chunk called with unknown pointer %p\n", ptr);
}

void NXFReader::close() {
    // Unmap all regions
    for (auto& region : mapped_regions_) {
        ::munmap(region.base, region.length);
    }
    mapped_regions_.clear();

    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }

    tensors_.clear();
    header_ = {};
    manifest_ = {};
}

// ─── Private helpers ────────────────────────────────────────────────────────

bool NXFReader::read_header() {
    // Read the 64-byte header
    ssize_t n = ::pread(fd_, &header_, sizeof(NXFHeader), 0);
    if (n != sizeof(NXFHeader)) {
        fprintf(stderr, "nxf: short read on header (%zd bytes)\n", n);
        return false;
    }

    // Validate magic
    if (header_.magic != kNXFMagic) {
        fprintf(stderr, "nxf: bad magic 0x%08X (expected 0x%08X)\n",
                header_.magic, kNXFMagic);
        return false;
    }

    // Validate version
    if (header_.version != kNXFVersion) {
        fprintf(stderr, "nxf: unsupported version %u (expected %u)\n",
                header_.version, kNXFVersion);
        return false;
    }

    // Validate file size against actual file size
    struct stat st;
    if (fstat(fd_, &st) == 0) {
        if (static_cast<uint64_t>(st.st_size) != header_.total_file_size) {
            fprintf(stderr, "nxf: file size mismatch: header says %llu, actual %llu\n",
                    (unsigned long long)header_.total_file_size,
                    (unsigned long long)st.st_size);
            return false;
        }
    }

    return true;
}

bool NXFReader::read_manifest() {
    if (header_.manifest_size == 0) return true; // No manifest is OK (empty model)

    std::vector<char> buf(header_.manifest_size);
    ssize_t n = ::pread(fd_, buf.data(), header_.manifest_size,
                        static_cast<off_t>(header_.manifest_offset));
    if (n < 0 || static_cast<uint64_t>(n) != header_.manifest_size) {
        fprintf(stderr, "nxf: short read on manifest (%zd / %llu)\n",
                n, (unsigned long long)header_.manifest_size);
        return false;
    }

    return json::parse_manifest(buf.data(), buf.size(), manifest_);
}

bool NXFReader::read_tensor_index() {
    if (header_.tensor_index_size == 0) return true;

    std::vector<uint8_t> buf(header_.tensor_index_size);
    ssize_t n = ::pread(fd_, buf.data(), header_.tensor_index_size,
                        static_cast<off_t>(header_.tensor_index_offset));
    if (n < 0 || static_cast<uint64_t>(n) != header_.tensor_index_size) {
        fprintf(stderr, "nxf: short read on tensor index (%zd / %llu)\n",
                n, (unsigned long long)header_.tensor_index_size);
        return false;
    }

    // Binary tensor index format:
    // For each tensor:
    //   [name_length:u32][name:chars][ndim:u32][shape:i64*ndim]
    //   [dtype:u8][num_chunks:u32][chunks:ChunkDesc*num_chunks]

    const uint8_t* p = buf.data();
    const uint8_t* end = p + buf.size();

    auto read_u32 = [&](uint32_t& val) -> bool {
        if (p + 4 > end) return false;
        memcpy(&val, p, 4);
        p += 4;
        return true;
    };
    auto read_u8 = [&](uint8_t& val) -> bool {
        if (p + 1 > end) return false;
        val = *p++;
        return true;
    };
    auto read_i64 = [&](int64_t& val) -> bool {
        if (p + 8 > end) return false;
        memcpy(&val, p, 8);
        p += 8;
        return true;
    };

    while (p < end) {
        TensorInfo ti;

        // Name
        uint32_t name_len;
        if (!read_u32(name_len)) break;
        if (p + name_len > end) {
            fprintf(stderr, "nxf: tensor index truncated at name\n");
            return false;
        }
        ti.name.assign(reinterpret_cast<const char*>(p), name_len);
        p += name_len;

        // Shape
        uint32_t ndim;
        if (!read_u32(ndim)) return false;
        ti.shape.resize(ndim);
        for (uint32_t d = 0; d < ndim; ++d) {
            if (!read_i64(ti.shape[d])) return false;
        }

        // DType
        uint8_t dtype_val;
        if (!read_u8(dtype_val)) return false;
        ti.dtype = static_cast<DType>(dtype_val);

        // Chunks
        uint32_t num_chunks;
        if (!read_u32(num_chunks)) return false;
        ti.chunks.resize(num_chunks);
        for (uint32_t c = 0; c < num_chunks; ++c) {
            if (p + sizeof(ChunkDesc) > end) {
                fprintf(stderr, "nxf: tensor index truncated at chunk %u of '%s'\n",
                        c, ti.name.c_str());
                return false;
            }
            memcpy(&ti.chunks[c], p, sizeof(ChunkDesc));
            p += sizeof(ChunkDesc);
        }

        tensors_.emplace(ti.name, std::move(ti));
    }

    return true;
}

} // namespace nexus::format
