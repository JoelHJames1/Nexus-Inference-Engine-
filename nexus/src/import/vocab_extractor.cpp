/// NEXUS Vocab Extractor — Implementation.
///
/// Re-reads the GGUF file header and metadata to locate and parse the
/// tokenizer string/uint32 arrays that the normal GGUF importer skips.

#include "import/vocab_extractor.h"

#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <unistd.h>

namespace nexus::import {

namespace {

/// Read exactly `len` bytes from fd at absolute offset. Returns false on short read.
bool pread_exact(int fd, void* buf, size_t len, off_t offset) {
    auto* dst = static_cast<uint8_t*>(buf);
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = ::pread(fd, dst, remaining, offset);
        if (n <= 0) return false;
        dst       += n;
        offset    += n;
        remaining -= static_cast<size_t>(n);
    }
    return true;
}

/// Read a GGUF-encoded string: [u64 length][chars...].  Advances `offset`.
bool read_gguf_string(int fd, off_t& offset, std::string& out) {
    uint64_t len = 0;
    if (!pread_exact(fd, &len, 8, offset)) return false;
    offset += 8;
    if (len > 16 * 1024 * 1024) return false;  // Sanity cap: 16 MB string
    out.resize(static_cast<size_t>(len));
    if (len > 0) {
        if (!pread_exact(fd, out.data(), static_cast<size_t>(len), offset)) return false;
        offset += static_cast<off_t>(len);
    }
    return true;
}

/// Return the byte size of a fixed-size GGUF value type, or 0 for variable types.
size_t gguf_value_fixed_size(GGUFValueType type) {
    switch (type) {
        case GGUFValueType::UINT8:
        case GGUFValueType::INT8:
        case GGUFValueType::BOOL:
            return 1;
        case GGUFValueType::UINT16:
        case GGUFValueType::INT16:
            return 2;
        case GGUFValueType::UINT32:
        case GGUFValueType::INT32:
        case GGUFValueType::FLOAT32:
            return 4;
        case GGUFValueType::UINT64:
        case GGUFValueType::INT64:
        case GGUFValueType::FLOAT64:
            return 8;
        default:
            return 0;  // STRING and ARRAY are variable-size
    }
}

/// Skip over a GGUF metadata value.  Advances `offset`.
bool skip_gguf_value(int fd, off_t& offset, GGUFValueType type) {
    size_t fixed = gguf_value_fixed_size(type);
    if (fixed > 0) {
        offset += static_cast<off_t>(fixed);
        return true;
    }
    if (type == GGUFValueType::STRING) {
        std::string tmp;
        return read_gguf_string(fd, offset, tmp);
    }
    if (type == GGUFValueType::ARRAY) {
        uint32_t elem_type = 0;
        uint64_t count = 0;
        if (!pread_exact(fd, &elem_type, 4, offset)) return false;
        offset += 4;
        if (!pread_exact(fd, &count, 8, offset)) return false;
        offset += 8;
        for (uint64_t i = 0; i < count; ++i) {
            if (!skip_gguf_value(fd, offset, static_cast<GGUFValueType>(elem_type)))
                return false;
        }
        return true;
    }
    return false;
}

}  // anonymous namespace

// ─── Read string array ────────────────────────────────────────────────────

bool VocabExtractor::read_string_array(int fd, off_t offset,
                                        std::vector<std::string>& out) {
    // At `offset` we expect the array header: [elem_type:u32][count:u64]
    uint32_t elem_type = 0;
    uint64_t count = 0;
    if (!pread_exact(fd, &elem_type, 4, offset)) return false;
    offset += 4;
    if (!pread_exact(fd, &count, 8, offset)) return false;
    offset += 8;

    if (static_cast<GGUFValueType>(elem_type) != GGUFValueType::STRING) {
        fprintf(stderr, "[vocab] Expected string array, got element type %u\n", elem_type);
        return false;
    }

    if (count > 1000000) {
        fprintf(stderr, "[vocab] Suspiciously large token array: %llu entries\n",
                (unsigned long long)count);
        return false;
    }

    out.reserve(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
        std::string s;
        if (!read_gguf_string(fd, offset, s)) {
            fprintf(stderr, "[vocab] Failed to read string %llu/%llu\n",
                    (unsigned long long)i, (unsigned long long)count);
            return false;
        }
        out.push_back(std::move(s));
    }
    return true;
}

// ─── Read uint32 array ────────────────────────────────────────────────────

bool VocabExtractor::read_uint32_array(int fd, off_t offset,
                                        std::vector<uint32_t>& out) {
    uint32_t elem_type = 0;
    uint64_t count = 0;
    if (!pread_exact(fd, &elem_type, 4, offset)) return false;
    offset += 4;
    if (!pread_exact(fd, &count, 8, offset)) return false;
    offset += 8;

    // Accept INT32 or UINT32 element types
    GGUFValueType vtype = static_cast<GGUFValueType>(elem_type);
    if (vtype != GGUFValueType::UINT32 && vtype != GGUFValueType::INT32) {
        fprintf(stderr, "[vocab] Expected uint32 array, got element type %u\n", elem_type);
        return false;
    }

    if (count > 1000000) return false;

    out.resize(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
        uint32_t v = 0;
        if (!pread_exact(fd, &v, 4, offset)) return false;
        offset += 4;
        out[static_cast<size_t>(i)] = v;
    }
    return true;
}

// ─── Extract from GGUF ───────────────────────────────────────────────────

bool VocabExtractor::extract_from_gguf(int gguf_fd,
                                        const GGUFFile& gguf,
                                        VocabData& data) {
    // We need to re-scan the metadata section to find the byte offsets of the
    // tokenizer arrays.  The parsed GGUFFile skips arrays, so we walk through
    // the file ourselves, matching keys to find the right offsets.

    off_t pos = 24;  // Skip the 24-byte GGUF header (magic + version + tensor_count + meta_count)

    data.tokens.clear();
    data.merges.clear();
    data.types.clear();

    bool found_tokens = false;

    for (uint64_t i = 0; i < gguf.meta_count; ++i) {
        // Read key string
        std::string key;
        if (!read_gguf_string(gguf_fd, pos, key)) {
            fprintf(stderr, "[vocab] Failed to read metadata key at index %llu\n",
                    (unsigned long long)i);
            return false;
        }

        // Read value type tag
        uint32_t raw_type = 0;
        if (!pread_exact(gguf_fd, &raw_type, 4, pos)) return false;
        pos += 4;
        GGUFValueType vtype = static_cast<GGUFValueType>(raw_type);

        // Check if this is one of the tokenizer keys we want.
        if (vtype == GGUFValueType::ARRAY) {
            if (key == "tokenizer.ggml.tokens") {
                // Read the string array in place (pos points to array header).
                off_t arr_start = pos;
                if (!read_string_array(gguf_fd, arr_start, data.tokens)) {
                    fprintf(stderr, "[vocab] Failed to read tokenizer.ggml.tokens array\n");
                    // Non-fatal: skip and continue
                } else {
                    found_tokens = true;
                    fprintf(stderr, "[vocab] Extracted %zu tokens from GGUF\n", data.tokens.size());
                }
                // Still need to skip past this array in the sequential scan.
                if (!skip_gguf_value(gguf_fd, pos, GGUFValueType::ARRAY)) return false;
                continue;
            }
            if (key == "tokenizer.ggml.merges") {
                off_t arr_start = pos;
                if (!read_string_array(gguf_fd, arr_start, data.merges)) {
                    fprintf(stderr, "[vocab] Failed to read tokenizer.ggml.merges array\n");
                } else {
                    fprintf(stderr, "[vocab] Extracted %zu merge rules from GGUF\n", data.merges.size());
                }
                if (!skip_gguf_value(gguf_fd, pos, GGUFValueType::ARRAY)) return false;
                continue;
            }
            if (key == "tokenizer.ggml.token_type") {
                off_t arr_start = pos;
                if (!read_uint32_array(gguf_fd, arr_start, data.types)) {
                    fprintf(stderr, "[vocab] Failed to read tokenizer.ggml.token_type array\n");
                } else {
                    fprintf(stderr, "[vocab] Extracted %zu token types from GGUF\n", data.types.size());
                }
                if (!skip_gguf_value(gguf_fd, pos, GGUFValueType::ARRAY)) return false;
                continue;
            }
            // Other array: skip it.
            if (!skip_gguf_value(gguf_fd, pos, GGUFValueType::ARRAY)) return false;
        } else {
            // Scalar or string: skip it.
            size_t fixed = gguf_value_fixed_size(vtype);
            if (fixed > 0) {
                pos += static_cast<off_t>(fixed);
            } else if (vtype == GGUFValueType::STRING) {
                std::string tmp;
                if (!read_gguf_string(gguf_fd, pos, tmp)) return false;
            } else {
                fprintf(stderr, "[vocab] Unknown value type %u for key '%s'\n",
                        raw_type, key.c_str());
                return false;
            }
        }
    }

    if (!found_tokens) {
        fprintf(stderr, "[vocab] WARNING: tokenizer.ggml.tokens not found in GGUF metadata\n");
    }

    return found_tokens;
}

// ─── Save/load vocab file ─────────────────────────────────────────────────

bool VocabExtractor::save_vocab_file(const std::vector<std::string>& tokens,
                                      const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        fprintf(stderr, "[vocab] Failed to create vocab file: %s\n", path.c_str());
        return false;
    }

    for (const auto& token : tokens) {
        // Write token as-is.  Tokens may contain arbitrary bytes; we write
        // them verbatim.  The file is treated as binary, one token per line.
        // Since tokens themselves can contain newlines (rare but possible in
        // BPE vocabs), we encode such bytes as <0xHH>.
        bool has_newline = false;
        for (char c : token) {
            if (c == '\n' || c == '\r') { has_newline = true; break; }
        }

        if (has_newline) {
            // Encode the entire token using hex escapes for problematic bytes.
            for (unsigned char c : token) {
                if (c == '\n' || c == '\r') {
                    char hex[8];
                    snprintf(hex, sizeof(hex), "<0x%02X>", c);
                    ofs.write(hex, static_cast<std::streamsize>(strlen(hex)));
                } else {
                    ofs.put(static_cast<char>(c));
                }
            }
        } else {
            ofs.write(token.data(), static_cast<std::streamsize>(token.size()));
        }
        ofs.put('\n');
    }

    fprintf(stderr, "[vocab] Saved %zu tokens to: %s\n", tokens.size(), path.c_str());
    return true;
}

bool VocabExtractor::save_merges_file(const std::vector<std::string>& merges,
                                       const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        fprintf(stderr, "[vocab] Failed to create merges file: %s\n", path.c_str());
        return false;
    }

    for (const auto& merge : merges) {
        ofs.write(merge.data(), static_cast<std::streamsize>(merge.size()));
        ofs.put('\n');
    }

    fprintf(stderr, "[vocab] Saved %zu merge rules to: %s\n", merges.size(), path.c_str());
    return true;
}

std::vector<std::string> VocabExtractor::load_vocab_file(const std::string& path) {
    std::vector<std::string> tokens;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        fprintf(stderr, "[vocab] Cannot open vocab file: %s\n", path.c_str());
        return tokens;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        // Remove trailing \r
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        tokens.push_back(std::move(line));
    }

    fprintf(stderr, "[vocab] Loaded %zu tokens from: %s\n", tokens.size(), path.c_str());
    return tokens;
}

std::vector<std::string> VocabExtractor::load_merges_file(const std::string& path) {
    std::vector<std::string> merges;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        return merges;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            merges.push_back(std::move(line));
        }
    }

    fprintf(stderr, "[vocab] Loaded %zu merge rules from: %s\n", merges.size(), path.c_str());
    return merges;
}

}  // namespace nexus::import
