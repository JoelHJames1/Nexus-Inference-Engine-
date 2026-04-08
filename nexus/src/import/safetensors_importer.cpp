/// NEXUS Safetensors Importer — Implementation.
///
/// Reads HuggingFace safetensors files using POSIX I/O (open/pread), parses
/// the JSON header, then writes each tensor into an NXF file via NXFWriter.
/// For multi-shard models, discovers all .safetensors files in a directory
/// and reads config.json for architecture metadata.

#include "import/safetensors_importer.h"
#include "format/nxf.h"
#include "quant/gptq.h"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <regex>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace nexus::import {

// ─── Helpers ───────────────────────────────────────────────────────────────

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

/// Convert FP16 (IEEE 754 half) to FP32.
float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t expo = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;

    if (expo == 0) {
        if (mant == 0) {
            float result;
            uint32_t bits = sign;
            std::memcpy(&result, &bits, 4);
            return result;
        }
        expo = 1;
        while ((mant & 0x0400) == 0) { mant <<= 1; expo--; }
        mant &= 0x03FF;
        uint32_t bits = sign | (static_cast<uint32_t>(expo + 127 - 15) << 23)
                       | (static_cast<uint32_t>(mant) << 13);
        float result;
        std::memcpy(&result, &bits, 4);
        return result;
    }
    if (expo == 31) {
        uint32_t bits = sign | 0x7F800000u | (static_cast<uint32_t>(mant) << 13);
        float result;
        std::memcpy(&result, &bits, 4);
        return result;
    }
    uint32_t bits = sign | (static_cast<uint32_t>(expo + 127 - 15) << 23)
                   | (static_cast<uint32_t>(mant) << 13);
    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}

/// Convert BF16 to FP32.
float bf16_to_fp32(uint16_t h) {
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}

/// Dequantize FP16 array to FP32.
void dequant_fp16(float* out, const uint16_t* src, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = fp16_to_fp32(src[i]);
}

/// Dequantize BF16 array to FP32.
void dequant_bf16(float* out, const uint16_t* src, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = bf16_to_fp32(src[i]);
}

// ─── Minimal JSON parser ──────────────────────────────────────────────────
//
// Handles the subset of JSON needed for safetensors headers and config.json:
//   - Objects (nested), arrays, strings, numbers (int/float), booleans, null.
// No dynamic allocation beyond std::string and std::vector.

enum class JsonType {
    NONE, OBJECT, ARRAY, STRING, NUMBER, BOOL, JNULL
};

struct JsonValue {
    JsonType type = JsonType::NONE;
    std::string str_val;
    double num_val = 0.0;
    bool bool_val = false;
    // For objects: parallel vectors of keys and values.
    std::vector<std::string> obj_keys;
    std::vector<JsonValue>   obj_vals;
    // For arrays: list of values.
    std::vector<JsonValue>   arr_vals;

    /// Lookup a key in an object. Returns nullptr if not found or not an object.
    const JsonValue* get(const std::string& key) const {
        if (type != JsonType::OBJECT) return nullptr;
        for (size_t i = 0; i < obj_keys.size(); ++i) {
            if (obj_keys[i] == key) return &obj_vals[i];
        }
        return nullptr;
    }

    /// Get a string value, or empty string if not a string.
    std::string as_string() const {
        return (type == JsonType::STRING) ? str_val : std::string{};
    }

    /// Get a numeric value, or fallback if not a number.
    double as_number(double fallback = 0.0) const {
        return (type == JsonType::NUMBER) ? num_val : fallback;
    }

    /// Get a uint32 value from a number.
    uint32_t as_uint32(uint32_t fallback = 0) const {
        return (type == JsonType::NUMBER) ? static_cast<uint32_t>(num_val) : fallback;
    }

    /// Get a float value from a number.
    float as_float(float fallback = 0.0f) const {
        return (type == JsonType::NUMBER) ? static_cast<float>(num_val) : fallback;
    }
};

/// Skip whitespace in a JSON string. Advances pos.
void json_skip_ws(const char* json, size_t len, size_t& pos) {
    while (pos < len && (json[pos] == ' ' || json[pos] == '\t' ||
                         json[pos] == '\n' || json[pos] == '\r')) {
        ++pos;
    }
}

/// Parse a JSON string (starting after the opening quote). Advances pos past closing quote.
bool json_parse_string(const char* json, size_t len, size_t& pos, std::string& out) {
    if (pos >= len || json[pos] != '"') return false;
    ++pos;  // skip opening quote
    out.clear();
    while (pos < len) {
        char c = json[pos++];
        if (c == '"') return true;
        if (c == '\\') {
            if (pos >= len) return false;
            char esc = json[pos++];
            switch (esc) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'b':  out += '\b'; break;
                case 'f':  out += '\f'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                case 'u': {
                    // Parse 4 hex digits; store as UTF-8 (simplified: ASCII range only)
                    if (pos + 4 > len) return false;
                    uint32_t cp = 0;
                    for (int i = 0; i < 4; ++i) {
                        char h = json[pos++];
                        cp <<= 4;
                        if (h >= '0' && h <= '9') cp |= (h - '0');
                        else if (h >= 'a' && h <= 'f') cp |= (h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') cp |= (h - 'A' + 10);
                        else return false;
                    }
                    if (cp < 0x80) {
                        out += static_cast<char>(cp);
                    } else if (cp < 0x800) {
                        out += static_cast<char>(0xC0 | (cp >> 6));
                        out += static_cast<char>(0x80 | (cp & 0x3F));
                    } else {
                        out += static_cast<char>(0xE0 | (cp >> 12));
                        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                        out += static_cast<char>(0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default: out += esc; break;
            }
        } else {
            out += c;
        }
    }
    return false;  // Unterminated string
}

// Forward declaration
bool json_parse_value(const char* json, size_t len, size_t& pos, JsonValue& out);

/// Parse a JSON object.
bool json_parse_object(const char* json, size_t len, size_t& pos, JsonValue& out) {
    if (pos >= len || json[pos] != '{') return false;
    ++pos;
    out.type = JsonType::OBJECT;
    out.obj_keys.clear();
    out.obj_vals.clear();

    json_skip_ws(json, len, pos);
    if (pos < len && json[pos] == '}') { ++pos; return true; }

    while (pos < len) {
        json_skip_ws(json, len, pos);
        std::string key;
        if (!json_parse_string(json, len, pos, key)) return false;

        json_skip_ws(json, len, pos);
        if (pos >= len || json[pos] != ':') return false;
        ++pos;

        json_skip_ws(json, len, pos);
        JsonValue val;
        if (!json_parse_value(json, len, pos, val)) return false;

        out.obj_keys.push_back(std::move(key));
        out.obj_vals.push_back(std::move(val));

        json_skip_ws(json, len, pos);
        if (pos < len && json[pos] == ',') { ++pos; continue; }
        if (pos < len && json[pos] == '}') { ++pos; return true; }
        return false;
    }
    return false;
}

/// Parse a JSON array.
bool json_parse_array(const char* json, size_t len, size_t& pos, JsonValue& out) {
    if (pos >= len || json[pos] != '[') return false;
    ++pos;
    out.type = JsonType::ARRAY;
    out.arr_vals.clear();

    json_skip_ws(json, len, pos);
    if (pos < len && json[pos] == ']') { ++pos; return true; }

    while (pos < len) {
        json_skip_ws(json, len, pos);
        JsonValue val;
        if (!json_parse_value(json, len, pos, val)) return false;
        out.arr_vals.push_back(std::move(val));

        json_skip_ws(json, len, pos);
        if (pos < len && json[pos] == ',') { ++pos; continue; }
        if (pos < len && json[pos] == ']') { ++pos; return true; }
        return false;
    }
    return false;
}

/// Parse a JSON number (int or float).
bool json_parse_number(const char* json, size_t len, size_t& pos, JsonValue& out) {
    size_t start = pos;
    if (pos < len && json[pos] == '-') ++pos;
    while (pos < len && json[pos] >= '0' && json[pos] <= '9') ++pos;
    if (pos < len && json[pos] == '.') {
        ++pos;
        while (pos < len && json[pos] >= '0' && json[pos] <= '9') ++pos;
    }
    if (pos < len && (json[pos] == 'e' || json[pos] == 'E')) {
        ++pos;
        if (pos < len && (json[pos] == '+' || json[pos] == '-')) ++pos;
        while (pos < len && json[pos] >= '0' && json[pos] <= '9') ++pos;
    }
    if (pos == start) return false;

    std::string num_str(json + start, pos - start);
    out.type = JsonType::NUMBER;
    out.num_val = std::strtod(num_str.c_str(), nullptr);
    return true;
}

/// Parse any JSON value.
bool json_parse_value(const char* json, size_t len, size_t& pos, JsonValue& out) {
    json_skip_ws(json, len, pos);
    if (pos >= len) return false;

    char c = json[pos];
    if (c == '{') return json_parse_object(json, len, pos, out);
    if (c == '[') return json_parse_array(json, len, pos, out);
    if (c == '"') {
        out.type = JsonType::STRING;
        return json_parse_string(json, len, pos, out.str_val);
    }
    if (c == 't') {
        if (pos + 4 <= len && std::strncmp(json + pos, "true", 4) == 0) {
            pos += 4; out.type = JsonType::BOOL; out.bool_val = true; return true;
        }
        return false;
    }
    if (c == 'f') {
        if (pos + 5 <= len && std::strncmp(json + pos, "false", 5) == 0) {
            pos += 5; out.type = JsonType::BOOL; out.bool_val = false; return true;
        }
        return false;
    }
    if (c == 'n') {
        if (pos + 4 <= len && std::strncmp(json + pos, "null", 4) == 0) {
            pos += 4; out.type = JsonType::JNULL; return true;
        }
        return false;
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
        return json_parse_number(json, len, pos, out);
    }
    return false;
}

/// Parse a complete JSON document from a buffer.
bool json_parse(const char* json, size_t len, JsonValue& out) {
    size_t pos = 0;
    if (!json_parse_value(json, len, pos, out)) return false;
    json_skip_ws(json, len, pos);
    return true;  // Allow trailing content (safetensors header may have __metadata__)
}

/// Read an entire file into a string.
bool read_file_to_string(const std::string& path, std::string& out) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st{};
    if (::fstat(fd, &st) < 0) { ::close(fd); return false; }
    size_t file_size = static_cast<size_t>(st.st_size);
    out.resize(file_size);
    bool ok = pread_exact(fd, out.data(), file_size, 0);
    ::close(fd);
    return ok;
}

/// Check if a path is a directory.
bool is_directory(const std::string& path) {
    struct stat st{};
    if (::stat(path.c_str(), &st) < 0) return false;
    return S_ISDIR(st.st_mode);
}

/// Check if a path is a regular file.
bool is_regular_file(const std::string& path) {
    struct stat st{};
    if (::stat(path.c_str(), &st) < 0) return false;
    return S_ISREG(st.st_mode);
}

}  // anonymous namespace

// ─── SafetensorsDType helpers ─────────────────────────────────────────────

SafetensorsDType SafetensorsImporter::parse_dtype(const std::string& dtype_str) {
    if (dtype_str == "F16")  return SafetensorsDType::F16;
    if (dtype_str == "F32")  return SafetensorsDType::F32;
    if (dtype_str == "BF16") return SafetensorsDType::BF16;
    if (dtype_str == "I8")   return SafetensorsDType::I8;
    if (dtype_str == "I32")  return SafetensorsDType::I32;
    if (dtype_str == "I64")  return SafetensorsDType::I64;
    if (dtype_str == "U8")   return SafetensorsDType::U8;
    if (dtype_str == "BOOL") return SafetensorsDType::BOOL;
    return SafetensorsDType::UNKNOWN;
}

size_t SafetensorsImporter::dtype_element_size(SafetensorsDType dtype) {
    switch (dtype) {
        case SafetensorsDType::F16:  return 2;
        case SafetensorsDType::F32:  return 4;
        case SafetensorsDType::BF16: return 2;
        case SafetensorsDType::I8:   return 1;
        case SafetensorsDType::I32:  return 4;
        case SafetensorsDType::I64:  return 8;
        case SafetensorsDType::U8:   return 1;
        case SafetensorsDType::BOOL: return 1;
        default:                     return 0;
    }
}

// ─── Codec selection ──────────────────────────────────────────────────────

void SafetensorsImporter::choose_codec(SafetensorsDType src_dtype, Codec target,
                                       Codec& out_codec, DType& out_dtype) {
    switch (src_dtype) {
        case SafetensorsDType::F32:
        case SafetensorsDType::F16:
        case SafetensorsDType::BF16:
            // Floating-point source: apply target codec.
            out_codec = target;
            if (target == Codec::INT4) {
                out_dtype = DType::I4;
            } else if (target == Codec::INT8) {
                out_dtype = DType::I8;
            } else if (target == Codec::FP16) {
                out_dtype = DType::F16;
            } else {
                out_dtype = DType::F32;
                out_codec = Codec::FP32;
            }
            return;

        // Integer/boolean types are typically non-weight tensors; store as FP32.
        case SafetensorsDType::I8:
        case SafetensorsDType::I32:
        case SafetensorsDType::I64:
        case SafetensorsDType::U8:
        case SafetensorsDType::BOOL:
            out_codec = Codec::FP32;
            out_dtype = DType::F32;
            return;

        default:
            out_codec = Codec::FP32;
            out_dtype = DType::F32;
            return;
    }
}

// ─── Tensor name mapping ──────────────────────────────────────────────────

std::string SafetensorsImporter::map_tensor_name(const std::string& hf_name) {
    // HuggingFace LLaMA-style naming -> NEXUS naming conventions.
    //
    // Token embeddings:
    //   "model.embed_tokens.weight" -> "tok_embeddings.weight"
    //
    // Output norm + output proj:
    //   "model.norm.weight"  -> "norm.weight"
    //   "lm_head.weight"     -> "output.weight"
    //
    // Per-layer transformer blocks (model.layers.N.*):
    //   "model.layers.N.self_attn.q_proj.weight"                -> "layers.N.attention.wq.weight"
    //   "model.layers.N.self_attn.k_proj.weight"                -> "layers.N.attention.wk.weight"
    //   "model.layers.N.self_attn.v_proj.weight"                -> "layers.N.attention.wv.weight"
    //   "model.layers.N.self_attn.o_proj.weight"                -> "layers.N.attention.wo.weight"
    //   "model.layers.N.mlp.gate_proj.weight"                   -> "layers.N.feed_forward.w1.weight"
    //   "model.layers.N.mlp.down_proj.weight"                   -> "layers.N.feed_forward.w2.weight"
    //   "model.layers.N.mlp.up_proj.weight"                     -> "layers.N.feed_forward.w3.weight"
    //   "model.layers.N.input_layernorm.weight"                 -> "layers.N.attention_norm.weight"
    //   "model.layers.N.post_attention_layernorm.weight"        -> "layers.N.ffn_norm.weight"

    // Token embeddings
    if (hf_name == "model.embed_tokens.weight") return "tok_embeddings.weight";

    // Output
    if (hf_name == "model.norm.weight")  return "norm.weight";
    if (hf_name == "lm_head.weight")     return "output.weight";

    // Per-layer tensors: match "model.layers.<N>.<suffix>"
    static const std::regex layer_re(R"(^model\.layers\.(\d+)\.(.+)$)");
    std::smatch m;
    if (std::regex_match(hf_name, m, layer_re)) {
        std::string layer_idx = m[1].str();
        std::string suffix    = m[2].str();
        std::string prefix    = "layers." + layer_idx;

        // Attention weights
        if (suffix == "self_attn.q_proj.weight")  return prefix + ".attention.wq.weight";
        if (suffix == "self_attn.k_proj.weight")  return prefix + ".attention.wk.weight";
        if (suffix == "self_attn.v_proj.weight")  return prefix + ".attention.wv.weight";
        if (suffix == "self_attn.o_proj.weight")  return prefix + ".attention.wo.weight";

        // Attention bias (some models have these)
        if (suffix == "self_attn.q_proj.bias")  return prefix + ".attention.wq.bias";
        if (suffix == "self_attn.k_proj.bias")  return prefix + ".attention.wk.bias";
        if (suffix == "self_attn.v_proj.bias")  return prefix + ".attention.wv.bias";
        if (suffix == "self_attn.o_proj.bias")  return prefix + ".attention.wo.bias";

        // Layer norms
        if (suffix == "input_layernorm.weight")            return prefix + ".attention_norm.weight";
        if (suffix == "post_attention_layernorm.weight")   return prefix + ".ffn_norm.weight";

        // MLP (SwiGLU: gate=w1, down=w2, up=w3)
        if (suffix == "mlp.gate_proj.weight")  return prefix + ".feed_forward.w1.weight";
        if (suffix == "mlp.down_proj.weight")  return prefix + ".feed_forward.w2.weight";
        if (suffix == "mlp.up_proj.weight")    return prefix + ".feed_forward.w3.weight";

        // MoE gate
        if (suffix == "mlp.gate.weight")       return prefix + ".feed_forward.gate.weight";

        // MoE expert sub-tensors: model.layers.N.mlp.experts.E.{gate_proj,down_proj,up_proj}.weight
        static const std::regex expert_re(R"(^mlp\.experts\.(\d+)\.(gate_proj|down_proj|up_proj)\.weight$)");
        std::smatch em;
        if (std::regex_match(suffix, em, expert_re)) {
            std::string eidx = em[1].str();
            std::string proj = em[2].str();
            std::string wname;
            if (proj == "gate_proj") wname = "w1";
            else if (proj == "down_proj") wname = "w2";
            else wname = "w3";
            return prefix + ".feed_forward.experts." + eidx + "." + wname + ".weight";
        }

        // Fallback: keep with layers.N prefix
        return prefix + "." + suffix;
    }

    // Fallback: keep original name
    return hf_name;
}

// ─── Shard parser ─────────────────────────────────────────────────────────

bool SafetensorsImporter::parse_shard(const std::string& path, SafetensorsShard& out) {
    out.path = path;

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "safetensors: failed to open '%s': %s\n", path.c_str(), strerror(errno));
        return false;
    }

    struct stat st{};
    if (::fstat(fd, &st) < 0) {
        fprintf(stderr, "safetensors: fstat failed: %s\n", strerror(errno));
        ::close(fd);
        return false;
    }
    uint64_t file_size = static_cast<uint64_t>(st.st_size);

    // Read first 8 bytes: little-endian uint64_t header_size
    if (file_size < 8) {
        fprintf(stderr, "safetensors: file too small: %llu bytes\n", (unsigned long long)file_size);
        ::close(fd);
        return false;
    }

    uint64_t header_size = 0;
    if (!pread_exact(fd, &header_size, 8, 0)) {
        fprintf(stderr, "safetensors: failed to read header size\n");
        ::close(fd);
        return false;
    }
    out.header_size = header_size;
    out.data_base = 8 + header_size;  // Tensor data starts after 8-byte prefix + JSON header

    // Sanity check
    if (8 + header_size > file_size) {
        fprintf(stderr, "safetensors: header_size (%llu) exceeds file size (%llu)\n",
                (unsigned long long)header_size, (unsigned long long)file_size);
        ::close(fd);
        return false;
    }
    if (header_size > 100 * 1024 * 1024) {
        fprintf(stderr, "safetensors: header_size too large (%llu), likely corrupt\n",
                (unsigned long long)header_size);
        ::close(fd);
        return false;
    }

    // Read JSON header
    std::vector<char> json_buf(static_cast<size_t>(header_size));
    if (!pread_exact(fd, json_buf.data(), static_cast<size_t>(header_size), 8)) {
        fprintf(stderr, "safetensors: failed to read JSON header\n");
        ::close(fd);
        return false;
    }
    ::close(fd);

    // Parse JSON
    JsonValue root;
    if (!json_parse(json_buf.data(), json_buf.size(), root)) {
        fprintf(stderr, "safetensors: failed to parse JSON header\n");
        return false;
    }
    if (root.type != JsonType::OBJECT) {
        fprintf(stderr, "safetensors: JSON header is not an object\n");
        return false;
    }

    // Extract tensor descriptors from the JSON object.
    // Each key is a tensor name (except "__metadata__"), each value is an object:
    //   { "dtype": "F16", "shape": [4096, 4096], "data_offsets": [0, 33554432] }
    out.tensors.clear();
    for (size_t i = 0; i < root.obj_keys.size(); ++i) {
        const std::string& key = root.obj_keys[i];
        const JsonValue& val = root.obj_vals[i];

        // Skip metadata entry
        if (key == "__metadata__") continue;

        if (val.type != JsonType::OBJECT) {
            fprintf(stderr, "safetensors: tensor '%s' is not an object, skipping\n", key.c_str());
            continue;
        }

        SafetensorsTensorDesc td{};
        td.name = key;

        // dtype
        const JsonValue* dtype_val = val.get("dtype");
        if (!dtype_val || dtype_val->type != JsonType::STRING) {
            fprintf(stderr, "safetensors: tensor '%s' missing dtype, skipping\n", key.c_str());
            continue;
        }
        td.dtype = parse_dtype(dtype_val->str_val);
        if (td.dtype == SafetensorsDType::UNKNOWN) {
            fprintf(stderr, "safetensors: tensor '%s' has unknown dtype '%s', skipping\n",
                    key.c_str(), dtype_val->str_val.c_str());
            continue;
        }

        // shape
        const JsonValue* shape_val = val.get("shape");
        if (!shape_val || shape_val->type != JsonType::ARRAY) {
            fprintf(stderr, "safetensors: tensor '%s' missing shape, skipping\n", key.c_str());
            continue;
        }
        td.shape.reserve(shape_val->arr_vals.size());
        for (const auto& dim : shape_val->arr_vals) {
            td.shape.push_back(static_cast<int64_t>(dim.as_number()));
        }

        // data_offsets
        const JsonValue* offsets_val = val.get("data_offsets");
        if (!offsets_val || offsets_val->type != JsonType::ARRAY || offsets_val->arr_vals.size() != 2) {
            fprintf(stderr, "safetensors: tensor '%s' missing/invalid data_offsets, skipping\n",
                    key.c_str());
            continue;
        }
        td.data_start = static_cast<uint64_t>(offsets_val->arr_vals[0].as_number());
        td.data_end   = static_cast<uint64_t>(offsets_val->arr_vals[1].as_number());

        out.tensors.push_back(std::move(td));
    }

    fprintf(stderr, "safetensors: '%s' — header=%llu bytes, %zu tensors\n",
            path.c_str(), (unsigned long long)header_size, out.tensors.size());

    return true;
}

// ─── config.json parser ───────────────────────────────────────────────────

HFModelConfig SafetensorsImporter::parse_config_json(const std::string& dir_path) {
    HFModelConfig config{};

    std::string config_path = dir_path + "/config.json";
    std::string json_str;
    if (!read_file_to_string(config_path, json_str)) {
        fprintf(stderr, "safetensors: no config.json found at '%s'\n", config_path.c_str());
        return config;
    }

    JsonValue root;
    if (!json_parse(json_str.c_str(), json_str.size(), root) || root.type != JsonType::OBJECT) {
        fprintf(stderr, "safetensors: failed to parse config.json\n");
        return config;
    }

    // Extract architecture fields
    const JsonValue* v;

    v = root.get("model_type");
    if (v) config.model_type = v->as_string();
    config.architecture = config.model_type.empty() ? "llama" : config.model_type;

    v = root.get("hidden_size");
    if (v) config.hidden_size = v->as_uint32();

    v = root.get("num_hidden_layers");
    if (v) config.num_hidden_layers = v->as_uint32();

    v = root.get("num_attention_heads");
    if (v) config.num_attention_heads = v->as_uint32();

    v = root.get("num_key_value_heads");
    if (v) config.num_key_value_heads = v->as_uint32();
    else   config.num_key_value_heads = config.num_attention_heads;  // default: MHA

    v = root.get("vocab_size");
    if (v) config.vocab_size = v->as_uint32();

    v = root.get("max_position_embeddings");
    if (v) config.max_position_embeddings = v->as_uint32();

    v = root.get("rope_theta");
    if (v) config.rope_theta = v->as_float(10000.0f);

    v = root.get("rms_norm_eps");
    if (v) config.rms_norm_eps = v->as_float(1e-5f);

    config.valid = (config.hidden_size > 0 && config.num_hidden_layers > 0);

    if (config.valid) {
        fprintf(stderr, "safetensors: config.json — arch=%s  hidden=%u  layers=%u  "
                "heads=%u  kv_heads=%u  vocab=%u  max_pos=%u  rope_theta=%.1f  rms_eps=%.1e\n",
                config.architecture.c_str(), config.hidden_size, config.num_hidden_layers,
                config.num_attention_heads, config.num_key_value_heads, config.vocab_size,
                config.max_position_embeddings, config.rope_theta, config.rms_norm_eps);
    }

    return config;
}

// ─── Build NXF manifest ──────────────────────────────────────────────────

format::ModelManifest SafetensorsImporter::build_manifest(const HFModelConfig& config,
                                                          Codec target_codec) {
    format::ModelManifest m{};
    m.architecture    = config.architecture;
    m.name            = config.model_type;
    m.num_layers      = config.num_hidden_layers;
    m.hidden_dim      = config.hidden_size;
    m.num_heads       = config.num_attention_heads;
    m.num_kv_heads    = config.num_key_value_heads;
    m.vocab_size      = config.vocab_size;
    m.max_seq_len     = config.max_position_embeddings;
    m.rope_theta      = config.rope_theta;
    m.rms_norm_eps    = config.rms_norm_eps;

    if (m.num_heads > 0 && m.hidden_dim > 0) {
        m.head_dim = m.hidden_dim / m.num_heads;
    }

    m.num_experts        = 0;
    m.num_active_experts = 0;
    m.default_codec      = target_codec;
    m.default_group_size = 128;

    fprintf(stderr, "nxf: manifest — arch=%s  layers=%u  hidden=%u  heads=%u  kv_heads=%u  vocab=%u\n",
            m.architecture.c_str(), m.num_layers, m.hidden_dim,
            m.num_heads, m.num_kv_heads, m.vocab_size);

    return m;
}

format::ModelManifest SafetensorsImporter::infer_manifest(
        const std::vector<SafetensorsShard>& shards, Codec target_codec) {
    // Fallback: try to infer architecture from tensor shapes.
    format::ModelManifest m{};
    m.architecture = "llama";
    m.default_codec = target_codec;
    m.default_group_size = 128;

    // Count layers by looking for layers.N patterns after name mapping.
    uint32_t max_layer = 0;
    for (const auto& shard : shards) {
        for (const auto& td : shard.tensors) {
            std::string nxf_name = map_tensor_name(td.name);
            static const std::regex layer_num_re(R"(^layers\.(\d+)\.)");
            std::smatch lm;
            if (std::regex_search(nxf_name, lm, layer_num_re)) {
                uint32_t idx = static_cast<uint32_t>(std::stoul(lm[1].str()));
                max_layer = std::max(max_layer, idx + 1);
            }
            // Infer hidden_dim from tok_embeddings shape
            if (nxf_name == "tok_embeddings.weight" && td.shape.size() == 2) {
                m.vocab_size = static_cast<uint32_t>(td.shape[0]);
                m.hidden_dim = static_cast<uint32_t>(td.shape[1]);
            }
        }
    }
    m.num_layers = max_layer;

    // Guess heads from hidden_dim (common: head_dim=128 for modern LLaMA)
    if (m.hidden_dim > 0) {
        m.head_dim = 128;
        m.num_heads = m.hidden_dim / m.head_dim;
        m.num_kv_heads = m.num_heads;  // Assume MHA unless config.json says otherwise
    }

    m.max_seq_len = 4096;
    m.rope_theta = 10000.0f;
    m.rms_norm_eps = 1e-5f;

    fprintf(stderr, "nxf: inferred manifest — layers=%u  hidden=%u  vocab=%u\n",
            m.num_layers, m.hidden_dim, m.vocab_size);

    return m;
}

// ─── Shard discovery ──────────────────────────────────────────────────────

std::vector<std::string> SafetensorsImporter::discover_shards(const std::string& dir_path) {
    std::vector<std::string> shards;

    DIR* dir = ::opendir(dir_path.c_str());
    if (!dir) {
        fprintf(stderr, "safetensors: failed to open directory '%s': %s\n",
                dir_path.c_str(), strerror(errno));
        return shards;
    }

    struct dirent* entry;
    while ((entry = ::readdir(dir)) != nullptr) {
        std::string name(entry->d_name);
        // Match files ending in .safetensors
        if (name.size() > 12 && name.substr(name.size() - 12) == ".safetensors") {
            shards.push_back(dir_path + "/" + name);
        }
    }
    ::closedir(dir);

    // Sort alphabetically so shards are processed in order
    // (model-00001-of-00003.safetensors < model-00002-of-00003.safetensors < ...)
    std::sort(shards.begin(), shards.end());

    fprintf(stderr, "safetensors: discovered %zu shard(s) in '%s'\n",
            shards.size(), dir_path.c_str());

    return shards;
}

// ─── Main conversion entry point ──────────────────────────────────────────

bool SafetensorsImporter::convert(const std::string& input_path,
                                  const std::string& nxf_path,
                                  Codec target_codec) {
    fprintf(stderr, "nexus: converting safetensors '%s' -> '%s'\n",
            input_path.c_str(), nxf_path.c_str());

    // Determine whether input is a single file or a directory of shards.
    std::vector<std::string> shard_paths;
    std::string config_dir;

    if (is_directory(input_path)) {
        shard_paths = discover_shards(input_path);
        if (shard_paths.empty()) {
            fprintf(stderr, "safetensors: no .safetensors files found in '%s'\n",
                    input_path.c_str());
            return false;
        }
        config_dir = input_path;
    } else if (is_regular_file(input_path)) {
        shard_paths.push_back(input_path);
        // Derive config_dir from parent directory
        size_t slash = input_path.rfind('/');
        config_dir = (slash != std::string::npos) ? input_path.substr(0, slash) : ".";
    } else {
        fprintf(stderr, "safetensors: '%s' is not a file or directory\n", input_path.c_str());
        return false;
    }

    // Parse all shards
    std::vector<SafetensorsShard> shards;
    shards.reserve(shard_paths.size());
    for (const auto& sp : shard_paths) {
        SafetensorsShard shard{};
        if (!parse_shard(sp, shard)) return false;
        shards.push_back(std::move(shard));
    }

    // Build manifest (prefer config.json, fall back to inference)
    HFModelConfig hf_config = parse_config_json(config_dir);
    format::ModelManifest manifest;
    if (hf_config.valid) {
        manifest = build_manifest(hf_config, target_codec);
    } else {
        manifest = infer_manifest(shards, target_codec);
    }

    // Create NXF writer
    auto writer = format::NXFWriter::create(nxf_path);
    if (!writer) {
        fprintf(stderr, "nxf: failed to create '%s'\n", nxf_path.c_str());
        return false;
    }
    writer->set_manifest(manifest);

    // Process each tensor across all shards
    size_t tensors_written   = 0;
    size_t tensors_skipped   = 0;
    uint64_t total_bytes_in  = 0;
    uint64_t total_bytes_out = 0;
    size_t total_tensors     = 0;
    for (const auto& shard : shards) total_tensors += shard.tensors.size();

    size_t global_idx = 0;
    for (const auto& shard : shards) {
        // Open the shard file for reading tensor data
        int fd = ::open(shard.path.c_str(), O_RDONLY);
        if (fd < 0) {
            fprintf(stderr, "safetensors: failed to reopen '%s': %s\n",
                    shard.path.c_str(), strerror(errno));
            return false;
        }

        struct stat st{};
        ::fstat(fd, &st);
        uint64_t file_size = static_cast<uint64_t>(st.st_size);

        for (const auto& td : shard.tensors) {
            ++global_idx;

            // Map tensor name
            std::string nxf_name = map_tensor_name(td.name);
            if (nxf_name.empty()) {
                fprintf(stderr, "  [%zu/%zu] skip: %s\n", global_idx, total_tensors, td.name.c_str());
                ++tensors_skipped;
                continue;
            }

            // Compute element count
            uint64_t num_elements = 1;
            for (auto d : td.shape) {
                if (d > 0) num_elements *= static_cast<uint64_t>(d);
            }

            // Compute data size
            uint64_t data_size = td.data_end - td.data_start;
            if (data_size == 0 || num_elements == 0) {
                fprintf(stderr, "  [%zu/%zu] skip (zero-size): %s\n",
                        global_idx, total_tensors, td.name.c_str());
                ++tensors_skipped;
                continue;
            }

            // Choose output codec
            Codec out_codec;
            DType out_dtype;
            choose_codec(td.dtype, target_codec, out_codec, out_dtype);

            fprintf(stderr, "  [%zu/%zu] %s -> %s  (%llu elems, %llu bytes)\n",
                    global_idx, total_tensors,
                    td.name.c_str(), nxf_name.c_str(),
                    (unsigned long long)num_elements, (unsigned long long)data_size);

            // Compute absolute file offset for this tensor's data
            uint64_t abs_offset = shard.data_base + td.data_start;
            if (abs_offset + data_size > file_size) {
                fprintf(stderr, "  ERROR: tensor data extends past end of file "
                        "(offset=0x%llX, size=%llu, file_size=%llu)\n",
                        (unsigned long long)abs_offset, (unsigned long long)data_size,
                        (unsigned long long)file_size);
                ::close(fd);
                return false;
            }

            // Read tensor data from shard
            std::vector<uint8_t> src_buf(static_cast<size_t>(data_size));
            if (!pread_exact(fd, src_buf.data(), static_cast<size_t>(data_size),
                             static_cast<off_t>(abs_offset))) {
                fprintf(stderr, "  ERROR: failed to read tensor data for '%s'\n", td.name.c_str());
                ::close(fd);
                return false;
            }
            total_bytes_in += data_size;

            // Determine whether we need dequant-to-FP32 path
            bool is_float_src = (td.dtype == SafetensorsDType::F32 ||
                                 td.dtype == SafetensorsDType::F16 ||
                                 td.dtype == SafetensorsDType::BF16);
            bool need_quant = (out_codec == Codec::INT4 || out_codec == Codec::INT8);

            // Build shape vector for NXF
            std::vector<int64_t> shape = td.shape;

            if (is_float_src && need_quant) {
                // --- Path A: dequant source to FP32, then quantize to INT4/INT8 ---
                size_t ne = static_cast<size_t>(num_elements);
                std::vector<float> fp32_buf(ne);

                // Convert source to FP32
                switch (td.dtype) {
                    case SafetensorsDType::F32:
                        std::memcpy(fp32_buf.data(), src_buf.data(), ne * sizeof(float));
                        break;
                    case SafetensorsDType::F16:
                        dequant_fp16(fp32_buf.data(),
                                     reinterpret_cast<const uint16_t*>(src_buf.data()), ne);
                        break;
                    case SafetensorsDType::BF16:
                        dequant_bf16(fp32_buf.data(),
                                     reinterpret_cast<const uint16_t*>(src_buf.data()), ne);
                        break;
                    default:
                        break;
                }
                src_buf.clear();  // Free raw data

                if (out_codec == Codec::INT4) {
                    // Quantize FP32 -> INT4
                    size_t group_size   = 128;
                    size_t num_groups   = (ne + group_size - 1) / group_size;
                    size_t packed_bytes = (ne + 1) / 2;  // 2 values per byte

                    std::vector<uint8_t> packed(packed_bytes);
                    std::vector<float>   scales(num_groups);
                    std::vector<float>   zeros(num_groups);

                    quant::quant_int4(packed.data(), scales.data(), zeros.data(),
                                      fp32_buf.data(), static_cast<int>(ne),
                                      static_cast<int>(group_size));

                    // Build NXF chunk: [packed_data][scales][zeros]
                    size_t chunk_size = packed_bytes + num_groups * sizeof(float) * 2;
                    std::vector<uint8_t> chunk_buf(chunk_size);
                    std::memcpy(chunk_buf.data(), packed.data(), packed_bytes);
                    std::memcpy(chunk_buf.data() + packed_bytes,
                                scales.data(), num_groups * sizeof(float));
                    std::memcpy(chunk_buf.data() + packed_bytes + num_groups * sizeof(float),
                                zeros.data(), num_groups * sizeof(float));

                    writer->begin_tensor(nxf_name, shape, out_dtype);
                    writer->add_chunk(chunk_buf.data(), chunk_buf.size(),
                                      Codec::INT4, static_cast<uint8_t>(group_size));
                    writer->end_tensor();

                    total_bytes_out += chunk_buf.size();
                } else {
                    // INT8: symmetric per-group quantization
                    size_t group_size = 128;
                    size_t num_groups = (ne + group_size - 1) / group_size;
                    std::vector<int8_t> qdata(ne);
                    std::vector<float>  scales(num_groups);

                    for (size_t g = 0; g < num_groups; ++g) {
                        size_t start = g * group_size;
                        size_t end   = std::min(start + group_size, ne);
                        float amax = 0.0f;
                        for (size_t j = start; j < end; ++j)
                            amax = std::max(amax, std::fabs(fp32_buf[j]));
                        float s = (amax > 0.0f) ? (127.0f / amax) : 1.0f;
                        scales[g] = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
                        for (size_t j = start; j < end; ++j) {
                            int v = static_cast<int>(std::round(fp32_buf[j] * s));
                            qdata[j] = static_cast<int8_t>(std::max(-128, std::min(127, v)));
                        }
                    }

                    size_t chunk_size = ne + num_groups * sizeof(float);
                    std::vector<uint8_t> chunk_buf(chunk_size);
                    std::memcpy(chunk_buf.data(), qdata.data(), ne);
                    std::memcpy(chunk_buf.data() + ne, scales.data(), num_groups * sizeof(float));

                    writer->begin_tensor(nxf_name, shape, out_dtype);
                    writer->add_chunk(chunk_buf.data(), chunk_buf.size(),
                                      Codec::INT8, static_cast<uint8_t>(group_size));
                    writer->end_tensor();

                    total_bytes_out += chunk_buf.size();
                }
            } else if (is_float_src && out_codec == Codec::FP16) {
                // --- Path B: convert to FP16 passthrough ---
                if (td.dtype == SafetensorsDType::F16) {
                    // Already FP16, direct passthrough
                    writer->begin_tensor(nxf_name, shape, DType::F16);
                    writer->add_chunk(src_buf.data(), src_buf.size(), Codec::FP16, 0);
                    writer->end_tensor();
                    total_bytes_out += src_buf.size();
                } else {
                    // Convert F32/BF16 -> FP32 first, then we store as FP16
                    // (FP16 target from FP32/BF16 source — dequant to FP32, then truncate)
                    size_t ne = static_cast<size_t>(num_elements);
                    std::vector<float> fp32_buf(ne);
                    if (td.dtype == SafetensorsDType::F32) {
                        std::memcpy(fp32_buf.data(), src_buf.data(), ne * sizeof(float));
                    } else {
                        dequant_bf16(fp32_buf.data(),
                                     reinterpret_cast<const uint16_t*>(src_buf.data()), ne);
                    }
                    src_buf.clear();

                    // Convert FP32 -> FP16
                    std::vector<uint16_t> fp16_buf(ne);
                    for (size_t i = 0; i < ne; ++i) {
                        // FP32 -> FP16 conversion
                        float val = fp32_buf[i];
                        uint32_t fbits;
                        std::memcpy(&fbits, &val, 4);
                        uint32_t s = (fbits >> 16) & 0x8000u;
                        int32_t  e = ((fbits >> 23) & 0xFF) - 127 + 15;
                        uint32_t m = fbits & 0x007FFFFFu;
                        if (e <= 0) {
                            fp16_buf[i] = static_cast<uint16_t>(s);  // Flush to zero
                        } else if (e >= 31) {
                            fp16_buf[i] = static_cast<uint16_t>(s | 0x7C00u);  // Inf
                        } else {
                            fp16_buf[i] = static_cast<uint16_t>(s | (e << 10) | (m >> 13));
                        }
                    }

                    size_t out_size = ne * 2;
                    writer->begin_tensor(nxf_name, shape, DType::F16);
                    writer->add_chunk(fp16_buf.data(), out_size, Codec::FP16, 0);
                    writer->end_tensor();
                    total_bytes_out += out_size;
                }
            } else if (is_float_src) {
                // --- Path C: FP32 passthrough ---
                if (td.dtype == SafetensorsDType::F32) {
                    writer->begin_tensor(nxf_name, shape, DType::F32);
                    writer->add_chunk(src_buf.data(), src_buf.size(), Codec::FP32, 0);
                    writer->end_tensor();
                    total_bytes_out += src_buf.size();
                } else {
                    // F16/BF16 -> FP32
                    size_t ne = static_cast<size_t>(num_elements);
                    std::vector<float> fp32_buf(ne);
                    if (td.dtype == SafetensorsDType::F16) {
                        dequant_fp16(fp32_buf.data(),
                                     reinterpret_cast<const uint16_t*>(src_buf.data()), ne);
                    } else {
                        dequant_bf16(fp32_buf.data(),
                                     reinterpret_cast<const uint16_t*>(src_buf.data()), ne);
                    }
                    size_t out_size = ne * sizeof(float);
                    writer->begin_tensor(nxf_name, shape, DType::F32);
                    writer->add_chunk(fp32_buf.data(), out_size, Codec::FP32, 0);
                    writer->end_tensor();
                    total_bytes_out += out_size;
                }
            } else {
                // --- Path D: Non-float types (I8, I32, etc.) — convert to FP32 ---
                size_t ne = static_cast<size_t>(num_elements);
                std::vector<float> fp32_buf(ne, 0.0f);

                switch (td.dtype) {
                    case SafetensorsDType::I8: {
                        const auto* p = reinterpret_cast<const int8_t*>(src_buf.data());
                        for (size_t i = 0; i < ne; ++i) fp32_buf[i] = static_cast<float>(p[i]);
                        break;
                    }
                    case SafetensorsDType::U8: {
                        const auto* p = src_buf.data();
                        for (size_t i = 0; i < ne; ++i) fp32_buf[i] = static_cast<float>(p[i]);
                        break;
                    }
                    case SafetensorsDType::I32: {
                        const auto* p = reinterpret_cast<const int32_t*>(src_buf.data());
                        for (size_t i = 0; i < ne; ++i) fp32_buf[i] = static_cast<float>(p[i]);
                        break;
                    }
                    case SafetensorsDType::I64: {
                        const auto* p = reinterpret_cast<const int64_t*>(src_buf.data());
                        for (size_t i = 0; i < ne; ++i) fp32_buf[i] = static_cast<float>(p[i]);
                        break;
                    }
                    case SafetensorsDType::BOOL: {
                        const auto* p = src_buf.data();
                        for (size_t i = 0; i < ne; ++i) fp32_buf[i] = (p[i] != 0) ? 1.0f : 0.0f;
                        break;
                    }
                    default:
                        break;
                }

                size_t out_size = ne * sizeof(float);
                writer->begin_tensor(nxf_name, shape, DType::F32);
                writer->add_chunk(fp32_buf.data(), out_size, Codec::FP32, 0);
                writer->end_tensor();
                total_bytes_out += out_size;
            }

            ++tensors_written;
        }

        ::close(fd);
    }

    // Finalize NXF file
    writer->finalize();

    fprintf(stderr, "\nnexus: safetensors conversion complete\n");
    fprintf(stderr, "  shards processed: %zu\n", shards.size());
    fprintf(stderr, "  tensors written : %zu\n", tensors_written);
    fprintf(stderr, "  tensors skipped : %zu\n", tensors_skipped);
    fprintf(stderr, "  input size      : %.2f GB\n",
            static_cast<double>(total_bytes_in) / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "  output size     : %.2f GB\n",
            static_cast<double>(total_bytes_out) / (1024.0 * 1024.0 * 1024.0));

    return true;
}

}  // namespace nexus::import
