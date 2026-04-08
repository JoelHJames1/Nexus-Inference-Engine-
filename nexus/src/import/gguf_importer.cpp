/// NEXUS GGUF Importer — Implementation.
///
/// Reads a GGUF file using POSIX I/O (open/pread), parses metadata and tensor
/// index, then writes each tensor into an NXF file via NXFWriter.

#include "import/gguf_importer.h"
#include "import/vocab_extractor.h"
#include "format/nxf.h"
#include "quant/gptq.h"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

/// Read a GGUF-encoded string: [u64 length][chars...].  Advances `offset`.
bool read_gguf_string(int fd, off_t& offset, std::string& out) {
    uint64_t len = 0;
    if (!pread_exact(fd, &len, 8, offset)) return false;
    offset += 8;
    if (len > 1024 * 1024) return false;  // Sanity cap: 1 MB string
    out.resize(static_cast<size_t>(len));
    if (len > 0) {
        if (!pread_exact(fd, out.data(), static_cast<size_t>(len), offset)) return false;
        offset += static_cast<off_t>(len);
    }
    return true;
}

/// Skip over a GGUF metadata value of the given type.  Advances `offset`.
bool skip_gguf_value(int fd, off_t& offset, GGUFValueType type);

/// Read a GGUF metadata value into a GGUFMetaKV.  Advances `offset`.
bool read_gguf_value(int fd, off_t& offset, GGUFMetaKV& kv) {
    kv.type = static_cast<GGUFValueType>(0);
    uint32_t raw_type = 0;
    if (!pread_exact(fd, &raw_type, 4, offset)) return false;
    offset += 4;
    kv.type = static_cast<GGUFValueType>(raw_type);

    switch (kv.type) {
        case GGUFValueType::UINT8: {
            uint8_t v = 0;
            if (!pread_exact(fd, &v, 1, offset)) return false;
            offset += 1; kv.val_uint = v;
            return true;
        }
        case GGUFValueType::INT8: {
            int8_t v = 0;
            if (!pread_exact(fd, &v, 1, offset)) return false;
            offset += 1; kv.val_int = v;
            return true;
        }
        case GGUFValueType::UINT16: {
            uint16_t v = 0;
            if (!pread_exact(fd, &v, 2, offset)) return false;
            offset += 2; kv.val_uint = v;
            return true;
        }
        case GGUFValueType::INT16: {
            int16_t v = 0;
            if (!pread_exact(fd, &v, 2, offset)) return false;
            offset += 2; kv.val_int = v;
            return true;
        }
        case GGUFValueType::UINT32: {
            uint32_t v = 0;
            if (!pread_exact(fd, &v, 4, offset)) return false;
            offset += 4; kv.val_uint = v;
            return true;
        }
        case GGUFValueType::INT32: {
            int32_t v = 0;
            if (!pread_exact(fd, &v, 4, offset)) return false;
            offset += 4; kv.val_int = v;
            return true;
        }
        case GGUFValueType::FLOAT32: {
            float v = 0;
            if (!pread_exact(fd, &v, 4, offset)) return false;
            offset += 4; kv.val_float = static_cast<double>(v);
            return true;
        }
        case GGUFValueType::BOOL: {
            uint8_t v = 0;
            if (!pread_exact(fd, &v, 1, offset)) return false;
            offset += 1; kv.val_bool = (v != 0);
            return true;
        }
        case GGUFValueType::STRING: {
            return read_gguf_string(fd, offset, kv.val_str);
        }
        case GGUFValueType::UINT64: {
            uint64_t v = 0;
            if (!pread_exact(fd, &v, 8, offset)) return false;
            offset += 8; kv.val_uint = v;
            return true;
        }
        case GGUFValueType::INT64: {
            int64_t v = 0;
            if (!pread_exact(fd, &v, 8, offset)) return false;
            offset += 8; kv.val_int = v;
            return true;
        }
        case GGUFValueType::FLOAT64: {
            double v = 0;
            if (!pread_exact(fd, &v, 8, offset)) return false;
            offset += 8; kv.val_float = v;
            return true;
        }
        case GGUFValueType::ARRAY: {
            // Read array header: [element_type:u32][count:u64]
            uint32_t elem_type = 0;
            uint64_t count = 0;
            if (!pread_exact(fd, &elem_type, 4, offset)) return false;
            offset += 4;
            if (!pread_exact(fd, &count, 8, offset)) return false;
            offset += 8;
            // Skip all elements
            for (uint64_t i = 0; i < count; ++i) {
                if (!skip_gguf_value(fd, offset, static_cast<GGUFValueType>(elem_type)))
                    return false;
            }
            return true;
        }
    }
    return false;
}

bool skip_gguf_value(int fd, off_t& offset, GGUFValueType type) {
    switch (type) {
        case GGUFValueType::UINT8:
        case GGUFValueType::INT8:
        case GGUFValueType::BOOL:
            offset += 1; return true;
        case GGUFValueType::UINT16:
        case GGUFValueType::INT16:
            offset += 2; return true;
        case GGUFValueType::UINT32:
        case GGUFValueType::INT32:
        case GGUFValueType::FLOAT32:
            offset += 4; return true;
        case GGUFValueType::UINT64:
        case GGUFValueType::INT64:
        case GGUFValueType::FLOAT64:
            offset += 8; return true;
        case GGUFValueType::STRING: {
            std::string tmp;
            return read_gguf_string(fd, offset, tmp);
        }
        case GGUFValueType::ARRAY: {
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
    }
    return false;
}

/// Align a value up to `alignment`.
uint64_t align_up(uint64_t v, uint64_t alignment) {
    return (v + alignment - 1) & ~(alignment - 1);
}

/// Convert FP16 (IEEE 754 half) to FP32.
float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t expo = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;

    if (expo == 0) {
        if (mant == 0) {
            // Signed zero
            float result;
            uint32_t bits = sign;
            std::memcpy(&result, &bits, 4);
            return result;
        }
        // Denormalized: convert to normalized FP32
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
        // Inf / NaN
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

/// Convert BF16 to FP32 (just shift the 16-bit value left by 16).
float bf16_to_fp32(uint16_t h) {
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}

}  // anonymous namespace

// ─── GGUFFile metadata accessors ───────────────────────────────────────────

std::string GGUFFile::meta_string(const std::string& key) const {
    for (const auto& kv : metadata) {
        if (kv.key == key && kv.type == GGUFValueType::STRING) return kv.val_str;
    }
    return {};
}

uint64_t GGUFFile::meta_uint(const std::string& key, uint64_t fallback) const {
    for (const auto& kv : metadata) {
        if (kv.key == key) {
            switch (kv.type) {
                case GGUFValueType::UINT8:
                case GGUFValueType::UINT16:
                case GGUFValueType::UINT32:
                case GGUFValueType::UINT64:
                    return kv.val_uint;
                case GGUFValueType::INT8:
                case GGUFValueType::INT16:
                case GGUFValueType::INT32:
                case GGUFValueType::INT64:
                    return static_cast<uint64_t>(kv.val_int);
                default: break;
            }
        }
    }
    return fallback;
}

double GGUFFile::meta_float(const std::string& key, double fallback) const {
    for (const auto& kv : metadata) {
        if (kv.key == key) {
            if (kv.type == GGUFValueType::FLOAT32 || kv.type == GGUFValueType::FLOAT64)
                return kv.val_float;
        }
    }
    return fallback;
}

// ─── GGUF type size tables ─────────────────────────────────────────────────

size_t GGUFImporter::gguf_type_block_size(GGUFType type) {
    // Byte size of one quantization block (or one element for unquantized).
    switch (type) {
        case GGUFType::F32:   return 4;
        case GGUFType::F16:   return 2;
        case GGUFType::BF16:  return 2;
        case GGUFType::Q4_0:  return 18;   // 32 values: 16 bytes data + 2 bytes scale
        case GGUFType::Q4_1:  return 20;   // 32 values: 16 bytes data + 2 scale + 2 min
        case GGUFType::Q5_0:  return 22;   // 32 values
        case GGUFType::Q5_1:  return 24;   // 32 values
        case GGUFType::Q8_0:  return 34;   // 32 values: 32 bytes data + 2 bytes scale
        case GGUFType::Q8_1:  return 36;   // 32 values: 32 bytes + 2 scale + 2 min
        case GGUFType::Q2_K:  return 256;  // Block size varies; approximate
        case GGUFType::Q3_K:  return 256;
        case GGUFType::Q4_K:  return 144;  // 256 values per super-block
        case GGUFType::Q5_K:  return 176;
        case GGUFType::Q6_K:  return 210;
        case GGUFType::I8:    return 1;
        case GGUFType::I16:   return 2;
        case GGUFType::I32:   return 4;
        case GGUFType::I64:   return 8;
        case GGUFType::F64:   return 8;
        default:              return 0;  // Unknown / unsupported
    }
}

size_t GGUFImporter::gguf_type_block_elements(GGUFType type) {
    // Number of elements represented by one block.
    switch (type) {
        case GGUFType::F32:
        case GGUFType::F16:
        case GGUFType::BF16:
        case GGUFType::I8:
        case GGUFType::I16:
        case GGUFType::I32:
        case GGUFType::I64:
        case GGUFType::F64:
            return 1;  // One element per "block"
        case GGUFType::Q4_0:
        case GGUFType::Q4_1:
        case GGUFType::Q5_0:
        case GGUFType::Q5_1:
        case GGUFType::Q8_0:
        case GGUFType::Q8_1:
            return 32;
        case GGUFType::Q2_K:
        case GGUFType::Q3_K:
        case GGUFType::Q4_K:
        case GGUFType::Q5_K:
        case GGUFType::Q6_K:
            return 256;
        default:
            return 1;
    }
}

size_t GGUFImporter::tensor_data_size(const GGUFTensorDesc& desc) {
    uint64_t num_elements = 1;
    for (auto d : desc.dims) num_elements *= d;
    size_t block_elems = gguf_type_block_elements(desc.type);
    size_t block_bytes = gguf_type_block_size(desc.type);
    if (block_elems == 0 || block_bytes == 0) return 0;
    uint64_t num_blocks = (num_elements + block_elems - 1) / block_elems;
    return static_cast<size_t>(num_blocks * block_bytes);
}

// ─── GGUF parser ───────────────────────────────────────────────────────────

bool GGUFImporter::parse_gguf(int fd, GGUFFile& out) {
    off_t pos = 0;

    // --- Header: magic, version, tensor_count, metadata_count ---
    struct {
        uint32_t magic;
        uint32_t version;
        uint64_t tensor_count;
        uint64_t meta_count;
    } hdr{};
    static_assert(sizeof(hdr) == 24);

    if (!pread_exact(fd, &hdr, sizeof(hdr), pos)) {
        fprintf(stderr, "gguf: failed to read header\n");
        return false;
    }
    pos += sizeof(hdr);

    if (hdr.magic != kGGUFMagic) {
        fprintf(stderr, "gguf: bad magic 0x%08X (expected 0x%08X)\n", hdr.magic, kGGUFMagic);
        return false;
    }
    if (hdr.version < 2 || hdr.version > 3) {
        fprintf(stderr, "gguf: unsupported version %u (expected 2 or 3)\n", hdr.version);
        return false;
    }
    out.version      = hdr.version;
    out.tensor_count = hdr.tensor_count;
    out.meta_count   = hdr.meta_count;

    fprintf(stderr, "gguf: version=%u  tensors=%llu  metadata=%llu\n",
            out.version, (unsigned long long)out.tensor_count,
            (unsigned long long)out.meta_count);

    // --- Parse metadata KV pairs ---
    out.metadata.reserve(static_cast<size_t>(out.meta_count));
    for (uint64_t i = 0; i < out.meta_count; ++i) {
        GGUFMetaKV kv{};
        if (!read_gguf_string(fd, pos, kv.key)) {
            fprintf(stderr, "gguf: failed to read metadata key %llu\n", (unsigned long long)i);
            return false;
        }
        if (!read_gguf_value(fd, pos, kv)) {
            fprintf(stderr, "gguf: failed to read metadata value for key '%s'\n", kv.key.c_str());
            return false;
        }
        out.metadata.push_back(std::move(kv));
    }

    // --- Parse tensor index ---
    out.tensors.reserve(static_cast<size_t>(out.tensor_count));
    for (uint64_t i = 0; i < out.tensor_count; ++i) {
        GGUFTensorDesc td{};
        if (!read_gguf_string(fd, pos, td.name)) {
            fprintf(stderr, "gguf: failed to read tensor name %llu\n", (unsigned long long)i);
            return false;
        }
        // n_dims
        if (!pread_exact(fd, &td.n_dims, 4, pos)) return false;
        pos += 4;
        // dims
        td.dims.resize(td.n_dims);
        for (uint32_t d = 0; d < td.n_dims; ++d) {
            uint64_t dim_val = 0;
            if (!pread_exact(fd, &dim_val, 8, pos)) return false;
            pos += 8;
            td.dims[d] = dim_val;
        }
        // type
        uint32_t raw_type = 0;
        if (!pread_exact(fd, &raw_type, 4, pos)) return false;
        pos += 4;
        td.type = static_cast<GGUFType>(raw_type);
        // offset (relative to data section start)
        if (!pread_exact(fd, &td.offset, 8, pos)) return false;
        pos += 8;

        out.tensors.push_back(std::move(td));
    }

    // --- Compute data section start (aligned to GGUF alignment, default 32) ---
    uint64_t alignment = 32;
    // Check for custom alignment in metadata
    for (const auto& kv : out.metadata) {
        if (kv.key == "general.alignment") {
            alignment = kv.val_uint;
            if (alignment == 0) alignment = 32;
            break;
        }
    }
    out.data_start = align_up(static_cast<uint64_t>(pos), alignment);

    fprintf(stderr, "gguf: data section starts at offset 0x%llX (alignment=%llu)\n",
            (unsigned long long)out.data_start, (unsigned long long)alignment);

    return true;
}

// ─── Tensor name mapping ───────────────────────────────────────────────────

std::string GGUFImporter::map_tensor_name(const std::string& gguf_name) {
    // GGUF (llama.cpp) naming conventions -> NEXUS naming conventions.
    //
    // Token embeddings:
    //   "token_embd.weight" -> "tok_embeddings.weight"
    //
    // Output norm + output proj:
    //   "output_norm.weight" -> "norm.weight"
    //   "output.weight"      -> "output.weight"
    //
    // Per-layer transformer blocks (blk.N.*):
    //   "blk.N.attn_q.weight"      -> "layers.N.attention.wq.weight"
    //   "blk.N.attn_k.weight"      -> "layers.N.attention.wk.weight"
    //   "blk.N.attn_v.weight"      -> "layers.N.attention.wv.weight"
    //   "blk.N.attn_output.weight" -> "layers.N.attention.wo.weight"
    //   "blk.N.attn_norm.weight"   -> "layers.N.attention_norm.weight"
    //   "blk.N.ffn_gate.weight"    -> "layers.N.feed_forward.w1.weight"
    //   "blk.N.ffn_down.weight"    -> "layers.N.feed_forward.w2.weight"
    //   "blk.N.ffn_up.weight"      -> "layers.N.feed_forward.w3.weight"
    //   "blk.N.ffn_norm.weight"    -> "layers.N.ffn_norm.weight"
    //
    // Rope freqs (sometimes present):
    //   "rope_freqs.weight"        -> (skip)

    // Skip rope_freqs — we recompute these.
    if (gguf_name == "rope_freqs.weight") return {};

    // Token embeddings
    if (gguf_name == "token_embd.weight") return "tok_embeddings.weight";

    // Output
    if (gguf_name == "output_norm.weight") return "norm.weight";
    if (gguf_name == "output.weight")      return "output.weight";

    // Block-level tensors: match "blk.<N>.<suffix>"
    static const std::regex blk_re(R"(^blk\.(\d+)\.(.+)$)");
    std::smatch m;
    if (std::regex_match(gguf_name, m, blk_re)) {
        std::string layer_idx = m[1].str();
        std::string suffix    = m[2].str();
        std::string prefix    = "layers." + layer_idx;

        // Attention weights
        if (suffix == "attn_q.weight")       return prefix + ".attention.wq.weight";
        if (suffix == "attn_k.weight")       return prefix + ".attention.wk.weight";
        if (suffix == "attn_v.weight")       return prefix + ".attention.wv.weight";
        if (suffix == "attn_output.weight")  return prefix + ".attention.wo.weight";

        // Attention norm
        if (suffix == "attn_norm.weight")    return prefix + ".attention_norm.weight";

        // FFN (SwiGLU: gate=w1, down=w2, up=w3)
        if (suffix == "ffn_gate.weight")     return prefix + ".feed_forward.w1.weight";
        if (suffix == "ffn_down.weight")     return prefix + ".feed_forward.w2.weight";
        if (suffix == "ffn_up.weight")       return prefix + ".feed_forward.w3.weight";

        // FFN norm
        if (suffix == "ffn_norm.weight")     return prefix + ".ffn_norm.weight";

        // MoE gate
        if (suffix == "ffn_gate_inp.weight") return prefix + ".feed_forward.gate.weight";

        // MoE expert sub-tensors: ffn_gate_exps, ffn_down_exps, ffn_up_exps
        if (suffix == "ffn_gate_exps.weight") return prefix + ".feed_forward.experts.w1.weight";
        if (suffix == "ffn_down_exps.weight") return prefix + ".feed_forward.experts.w2.weight";
        if (suffix == "ffn_up_exps.weight")   return prefix + ".feed_forward.experts.w3.weight";

        // Per-expert tensors: ffn_gate.N.weight etc.
        static const std::regex expert_re(R"(^ffn_(gate|down|up)\.(\d+)\.weight$)");
        std::smatch em;
        if (std::regex_match(suffix, em, expert_re)) {
            std::string role = em[1].str();
            std::string eidx = em[2].str();
            std::string wname;
            if (role == "gate") wname = "w1";
            else if (role == "down") wname = "w2";
            else wname = "w3";
            return prefix + ".feed_forward.experts." + eidx + "." + wname + ".weight";
        }

        // Fallback: pass through with layers.N prefix
        return prefix + "." + suffix;
    }

    // Fallback: keep original name
    return gguf_name;
}

// ─── Build NXF manifest from GGUF metadata ─────────────────────────────────

format::ModelManifest GGUFImporter::build_manifest(const GGUFFile& gguf, Codec target_codec) {
    format::ModelManifest m{};

    std::string arch = gguf.meta_string("general.architecture");
    if (arch.empty()) arch = "llama";
    m.architecture = arch;
    m.name = gguf.meta_string("general.name");

    // Architecture-prefixed keys (e.g., "llama.block_count")
    std::string p = arch + ".";

    m.num_layers      = static_cast<uint32_t>(gguf.meta_uint(p + "block_count", 0));
    m.hidden_dim      = static_cast<uint32_t>(gguf.meta_uint(p + "embedding_length", 0));
    m.num_heads       = static_cast<uint32_t>(gguf.meta_uint(p + "attention.head_count", 0));
    m.num_kv_heads    = static_cast<uint32_t>(gguf.meta_uint(p + "attention.head_count_kv", m.num_heads));
    m.vocab_size      = static_cast<uint32_t>(gguf.meta_uint(p + "vocab_size", 0));
    m.max_seq_len     = static_cast<uint32_t>(gguf.meta_uint(p + "context_length", 4096));
    m.rope_theta      = static_cast<float>(gguf.meta_float(p + "rope.freq_base", 10000.0));
    m.rms_norm_eps    = static_cast<float>(gguf.meta_float(p + "attention.layer_norm_rms_epsilon", 1e-5));

    // Compute head_dim
    if (m.num_heads > 0 && m.hidden_dim > 0) {
        m.head_dim = m.hidden_dim / m.num_heads;
    }

    // MoE fields
    m.num_experts       = static_cast<uint32_t>(gguf.meta_uint(p + "expert_count", 0));
    m.num_active_experts = static_cast<uint32_t>(gguf.meta_uint(p + "expert_used_count", 0));

    m.default_codec     = target_codec;
    m.default_group_size = 128;

    fprintf(stderr, "nxf: arch=%s  name=%s  layers=%u  hidden=%u  heads=%u  kv_heads=%u  vocab=%u\n",
            m.architecture.c_str(), m.name.c_str(),
            m.num_layers, m.hidden_dim, m.num_heads, m.num_kv_heads, m.vocab_size);

    return m;
}

// ─── Codec selection ───────────────────────────────────────────────────────

void GGUFImporter::choose_codec(GGUFType src_type, Codec target,
                                Codec& out_codec, DType& out_dtype) {
    switch (src_type) {
        // Unquantized sources: apply target codec via re-quantization.
        case GGUFType::F32:
        case GGUFType::F16:
        case GGUFType::BF16:
        case GGUFType::F64:
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

        // Already 4-bit quantized: passthrough as INT4.
        case GGUFType::Q4_0:
        case GGUFType::Q4_1:
        case GGUFType::Q4_K:
        case GGUFType::IQ4_NL:
        case GGUFType::IQ4_XS:
            out_codec = Codec::INT4;
            out_dtype = DType::I4;
            return;

        // 8-bit quantized: passthrough as INT8.
        case GGUFType::Q8_0:
        case GGUFType::Q8_1:
            out_codec = Codec::INT8;
            out_dtype = DType::I8;
            return;

        // 5-bit and higher K-quants: if target is INT4, dequant+requant;
        // otherwise passthrough (approximated as INT8 for now).
        case GGUFType::Q5_0:
        case GGUFType::Q5_1:
        case GGUFType::Q5_K:
        case GGUFType::Q6_K:
            if (target == Codec::INT4 || target == Codec::INT8) {
                out_codec = target;
                out_dtype = (target == Codec::INT4) ? DType::I4 : DType::I8;
            } else {
                out_codec = Codec::INT8;
                out_dtype = DType::I8;
            }
            return;

        // Low-bit types: dequant to FP32 first, then apply target.
        case GGUFType::Q2_K:
        case GGUFType::Q3_K:
        case GGUFType::IQ2_XXS:
        case GGUFType::IQ2_XS:
        case GGUFType::IQ2_S:
        case GGUFType::IQ3_XXS:
        case GGUFType::IQ3_S:
        case GGUFType::IQ1_S:
        case GGUFType::IQ1_M:
            out_codec = target;
            out_dtype = (target == Codec::INT4) ? DType::I4 : DType::F16;
            return;

        // Integer types (not weights): store as-is.
        case GGUFType::I8:
        case GGUFType::I16:
        case GGUFType::I32:
        case GGUFType::I64:
            out_codec = Codec::FP32;
            out_dtype = DType::F32;
            return;

        default:
            out_codec = Codec::FP32;
            out_dtype = DType::F32;
            return;
    }
}

// ─── Dequantize GGUF block-quantized data to FP32 ─────────────────────────

namespace {

/// Dequantize Q4_0 block data to FP32.
/// Each block: [fp16 scale][16 bytes of packed 4-bit data] = 18 bytes for 32 elements.
void dequant_q4_0(float* out, const uint8_t* src, size_t num_elements) {
    size_t num_blocks = num_elements / 32;
    for (size_t b = 0; b < num_blocks; ++b) {
        const uint8_t* block = src + b * 18;
        uint16_t raw_scale;
        std::memcpy(&raw_scale, block, 2);
        float scale = fp16_to_fp32(raw_scale);
        const uint8_t* quants = block + 2;
        for (int j = 0; j < 16; ++j) {
            uint8_t byte = quants[j];
            int lo = (byte & 0x0F) - 8;
            int hi = (byte >> 4)   - 8;
            out[b * 32 + j]      = static_cast<float>(lo) * scale;
            out[b * 32 + j + 16] = static_cast<float>(hi) * scale;
        }
    }
}

/// Dequantize Q4_1 block data to FP32.
/// Each block: [fp16 scale][fp16 min][16 bytes of packed 4-bit data] = 20 bytes for 32 elements.
void dequant_q4_1(float* out, const uint8_t* src, size_t num_elements) {
    size_t num_blocks = num_elements / 32;
    for (size_t b = 0; b < num_blocks; ++b) {
        const uint8_t* block = src + b * 20;
        uint16_t raw_scale, raw_min;
        std::memcpy(&raw_scale, block, 2);
        std::memcpy(&raw_min, block + 2, 2);
        float scale = fp16_to_fp32(raw_scale);
        float minv  = fp16_to_fp32(raw_min);
        const uint8_t* quants = block + 4;
        for (int j = 0; j < 16; ++j) {
            uint8_t byte = quants[j];
            int lo = byte & 0x0F;
            int hi = byte >> 4;
            out[b * 32 + j]      = static_cast<float>(lo) * scale + minv;
            out[b * 32 + j + 16] = static_cast<float>(hi) * scale + minv;
        }
    }
}

/// Dequantize Q8_0 block data to FP32.
/// Each block: [fp16 scale][32 int8 values] = 34 bytes for 32 elements.
void dequant_q8_0(float* out, const uint8_t* src, size_t num_elements) {
    size_t num_blocks = num_elements / 32;
    for (size_t b = 0; b < num_blocks; ++b) {
        const uint8_t* block = src + b * 34;
        uint16_t raw_scale;
        std::memcpy(&raw_scale, block, 2);
        float scale = fp16_to_fp32(raw_scale);
        const int8_t* quants = reinterpret_cast<const int8_t*>(block + 2);
        for (int j = 0; j < 32; ++j) {
            out[b * 32 + j] = static_cast<float>(quants[j]) * scale;
        }
    }
}

/// Dequantize FP16 array to FP32.
void dequant_fp16(float* out, const uint16_t* src, size_t num_elements) {
    for (size_t i = 0; i < num_elements; ++i) {
        out[i] = fp16_to_fp32(src[i]);
    }
}

/// Dequantize BF16 array to FP32.
void dequant_bf16(float* out, const uint16_t* src, size_t num_elements) {
    for (size_t i = 0; i < num_elements; ++i) {
        out[i] = bf16_to_fp32(src[i]);
    }
}

// ─── K-quant dequantizers ────────────────────────────────────────────────────
// K-quant formats use "super-blocks" of 256 elements with multi-level quantization.
// Each super-block has scales/mins stored at reduced precision, plus packed quant values.

/// Dequantize Q2_K: 2-bit quantization with 4-bit scales/mins.
/// Super-block: 256 elements.
/// Layout: [16 bytes scales (4-bit packed)][16 bytes mins (4-bit packed)][64 bytes quants (2-bit packed)][fp16 dmin][fp16 d] = 84 bytes
void dequant_q2_k(float* out, const uint8_t* src, size_t num_elements) {
    const size_t QK = 256;
    const size_t block_size = 84;
    size_t num_blocks = num_elements / QK;

    for (size_t b = 0; b < num_blocks; ++b) {
        const uint8_t* block = src + b * block_size;
        const uint8_t* scales_buf = block;       // 16 bytes: 16 x 4-bit scales packed as pairs
        const uint8_t* quants = block + 16;      // 64 bytes: 256 x 2-bit values
        uint16_t raw_dmin, raw_d;
        std::memcpy(&raw_dmin, block + 80, 2);
        std::memcpy(&raw_d, block + 82, 2);
        float dmin = fp16_to_fp32(raw_dmin);
        float d    = fp16_to_fp32(raw_d);

        for (int j = 0; j < QK / 16; ++j) {
            // Each group of 16 elements shares a 4-bit scale and 4-bit min
            uint8_t sc_byte = scales_buf[j];
            float scale = d * (sc_byte & 0x0F);
            float min   = dmin * ((sc_byte >> 4) & 0x0F);

            for (int k = 0; k < 16; ++k) {
                int idx = j * 16 + k;
                int byte_idx = idx / 4;
                int bit_shift = (idx % 4) * 2;
                int q = (quants[byte_idx] >> bit_shift) & 0x03;
                out[b * QK + idx] = scale * static_cast<float>(q) - min;
            }
        }
    }
}

/// Dequantize Q3_K: 3-bit quantization with 6-bit super-block scales.
/// Super-block: 256 elements, divided into 16 groups of 16.
/// Layout: [32 bytes hmasks][96 bytes quants (low 2 bits, packed)][12 bytes scales (6-bit packed)][fp16 d] = 110 bytes
void dequant_q3_k(float* out, const uint8_t* src, size_t num_elements) {
    const size_t QK = 256;
    // block_q3_K: 96 bytes qs + 32 bytes hmask + 12 bytes scales + 2 bytes d = 110
    // But llama.cpp layout is: hmask[32], qs[64], scales[12], d[2] for 256 elements
    // Actually: struct block_q3_K { uint8_t hmask[QK/8]; uint8_t qs[QK/4]; uint8_t scales[12]; ggml_half d; }
    // QK/8 = 32, QK/4 = 64, so total = 32 + 64 + 12 + 2 = 110 bytes
    const size_t block_size = 110;
    size_t num_blocks = num_elements / QK;

    for (size_t b = 0; b < num_blocks; ++b) {
        const uint8_t* block = src + b * block_size;
        const uint8_t* hmask = block;            // 32 bytes: high bit mask
        const uint8_t* qs    = block + 32;       // 64 bytes: low 2 bits packed (4 per byte)
        const uint8_t* sc    = block + 96;       // 12 bytes: scales packed
        uint16_t raw_d;
        std::memcpy(&raw_d, block + 108, 2);
        float d = fp16_to_fp32(raw_d);

        // Unpack scales (16 x 6-bit values packed into 12 bytes)
        int32_t scales[16];
        for (int i = 0; i < 8; ++i) {
            scales[i]     = (sc[i] & 0x0F) | (((sc[8 + (i / 2)] >> (4 * (i % 2))) & 3) << 4);
            scales[i + 8] = (sc[i] >> 4)   | (((sc[8 + (i / 2)] >> (4 * (i % 2) + 2)) & 3) << 4);
        }
        // Center around 32: scale = scale - 32
        for (int i = 0; i < 16; ++i) {
            scales[i] -= 32;
        }

        for (int j = 0; j < QK; ++j) {
            // Low 2 bits from qs
            int byte_idx = j / 4;
            int bit_shift = (j % 4) * 2;
            int q_lo = (qs[byte_idx] >> bit_shift) & 0x03;
            // High bit from hmask
            int q_hi = (hmask[j / 8] >> (j % 8)) & 1;
            int q = q_lo | (q_hi << 2);  // 3-bit value [0..7]
            // Center: q - 4
            int group = j / 16;
            out[b * QK + j] = d * scales[group] * (static_cast<float>(q) - 4.0f);
        }
    }
}

/// Dequantize Q4_K: 4-bit quantization with 6-bit scales/mins.
/// Super-block: 256 elements.
/// Layout: fp16 d, fp16 dmin, 12 bytes scales, 128 bytes quants = 144 bytes
void dequant_q4_k(float* out, const uint8_t* src, size_t num_elements) {
    const size_t QK = 256;
    // struct block_q4_K { ggml_half d; ggml_half dmin; uint8_t scales[12]; uint8_t qs[QK/2]; }
    const size_t block_size = 2 + 2 + 12 + QK / 2;  // 144 bytes
    size_t num_blocks = num_elements / QK;

    for (size_t b = 0; b < num_blocks; ++b) {
        const uint8_t* block = src + b * block_size;
        uint16_t raw_d, raw_dmin;
        std::memcpy(&raw_d, block, 2);
        std::memcpy(&raw_dmin, block + 2, 2);
        float d    = fp16_to_fp32(raw_d);
        float dmin = fp16_to_fp32(raw_dmin);
        const uint8_t* sc = block + 4;        // 12 bytes scales
        const uint8_t* qs = block + 16;       // 128 bytes quants

        // Unpack 8 x (6-bit scale, 6-bit min) from 12 bytes
        uint8_t scales[8], mins[8];
        for (int i = 0; i < 4; ++i) {
            scales[i]     = sc[i] & 0x3F;
            scales[i + 4] = sc[i + 4] & 0x3F;
            mins[i]       = sc[i] >> 6 | ((sc[i + 8] & 0x0F) << 2);
            mins[i + 4]   = sc[i + 4] >> 6 | ((sc[i + 8] >> 4) << 2);
        }

        for (int j = 0; j < QK / 2; ++j) {
            uint8_t byte = qs[j];
            int group_lo = (j * 2) / 32;
            int group_hi = (j * 2 + 1) / 32;
            int lo = byte & 0x0F;
            int hi = byte >> 4;
            out[b * QK + j * 2]     = d * scales[group_lo] * lo - dmin * mins[group_lo];
            out[b * QK + j * 2 + 1] = d * scales[group_hi] * hi - dmin * mins[group_hi];
        }
    }
}

/// Dequantize Q5_K: 5-bit quantization.
/// Super-block: 256 elements.
/// Layout: fp16 d, fp16 dmin, 12 bytes scales, 128 bytes qs, 32 bytes qh = 176 bytes
void dequant_q5_k(float* out, const uint8_t* src, size_t num_elements) {
    const size_t QK = 256;
    // struct block_q5_K { ggml_half d; ggml_half dmin; uint8_t scales[12]; uint8_t qh[QK/8]; uint8_t qs[QK/2]; }
    const size_t block_size = 2 + 2 + 12 + QK / 8 + QK / 2;  // 176 bytes
    size_t num_blocks = num_elements / QK;

    for (size_t b = 0; b < num_blocks; ++b) {
        const uint8_t* block = src + b * block_size;
        uint16_t raw_d, raw_dmin;
        std::memcpy(&raw_d, block, 2);
        std::memcpy(&raw_dmin, block + 2, 2);
        float d    = fp16_to_fp32(raw_d);
        float dmin = fp16_to_fp32(raw_dmin);
        const uint8_t* sc = block + 4;
        const uint8_t* qh = block + 16;       // 32 bytes: high bits
        const uint8_t* qs = block + 48;       // 128 bytes: low 4 bits

        // Unpack scales/mins same as Q4_K
        uint8_t scales[8], mins[8];
        for (int i = 0; i < 4; ++i) {
            scales[i]     = sc[i] & 0x3F;
            scales[i + 4] = sc[i + 4] & 0x3F;
            mins[i]       = sc[i] >> 6 | ((sc[i + 8] & 0x0F) << 2);
            mins[i + 4]   = sc[i + 4] >> 6 | ((sc[i + 8] >> 4) << 2);
        }

        for (int j = 0; j < QK / 2; ++j) {
            uint8_t byte = qs[j];
            int lo = byte & 0x0F;
            int hi = byte >> 4;
            // High bit from qh
            int h_lo = (qh[(j * 2) / 8] >> ((j * 2) % 8)) & 1;
            int h_hi = (qh[(j * 2 + 1) / 8] >> ((j * 2 + 1) % 8)) & 1;
            int q_lo = lo | (h_lo << 4);  // 5-bit
            int q_hi = hi | (h_hi << 4);  // 5-bit
            int group_lo = (j * 2) / 32;
            int group_hi = (j * 2 + 1) / 32;
            out[b * QK + j * 2]     = d * scales[group_lo] * q_lo - dmin * mins[group_lo];
            out[b * QK + j * 2 + 1] = d * scales[group_hi] * q_hi - dmin * mins[group_hi];
        }
    }
}

/// Dequantize Q6_K: 6-bit quantization.
/// Super-block: 256 elements.
/// Layout: 128 bytes ql, 64 bytes qh, 16 bytes scales (int8), fp16 d = 210 bytes
void dequant_q6_k(float* out, const uint8_t* src, size_t num_elements) {
    const size_t QK = 256;
    // struct block_q6_K { uint8_t ql[QK/2]; uint8_t qh[QK/4]; int8_t scales[QK/16]; ggml_half d; }
    const size_t block_size = QK / 2 + QK / 4 + QK / 16 + 2;  // 210 bytes
    size_t num_blocks = num_elements / QK;

    for (size_t b = 0; b < num_blocks; ++b) {
        const uint8_t* block = src + b * block_size;
        const uint8_t* ql = block;              // 128 bytes: low 4 bits
        const uint8_t* qh = block + 128;        // 64 bytes: high 2 bits
        const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);  // 16 bytes
        uint16_t raw_d;
        std::memcpy(&raw_d, block + 208, 2);
        float d = fp16_to_fp32(raw_d);

        for (int j = 0; j < QK; ++j) {
            // Low 4 bits from ql (packed 2 per byte)
            int ql_byte = j / 2;
            int q_lo = (j % 2 == 0) ? (ql[ql_byte] & 0x0F) : (ql[ql_byte] >> 4);
            // High 2 bits from qh (packed 4 per byte)
            int qh_byte = j / 4;
            int qh_shift = (j % 4) * 2;
            int q_hi = (qh[qh_byte] >> qh_shift) & 0x03;
            // 6-bit value
            int q = q_lo | (q_hi << 4);
            // Center around 32: q - 32
            int group = j / 16;
            out[b * QK + j] = d * scales[group] * (static_cast<float>(q) - 32.0f);
        }
    }
}

/// Dequantize a GGUF tensor's raw data to FP32.
/// Returns true if dequantization was performed; false if the type is not supported.
bool dequant_gguf_to_fp32(float* out, const uint8_t* src, GGUFType type, size_t num_elements) {
    switch (type) {
        case GGUFType::F32:
            std::memcpy(out, src, num_elements * 4);
            return true;
        case GGUFType::F16:
            dequant_fp16(out, reinterpret_cast<const uint16_t*>(src), num_elements);
            return true;
        case GGUFType::BF16:
            dequant_bf16(out, reinterpret_cast<const uint16_t*>(src), num_elements);
            return true;
        case GGUFType::Q4_0:
            dequant_q4_0(out, src, num_elements);
            return true;
        case GGUFType::Q4_1:
            dequant_q4_1(out, src, num_elements);
            return true;
        case GGUFType::Q8_0:
            dequant_q8_0(out, src, num_elements);
            return true;
        case GGUFType::Q2_K:
            dequant_q2_k(out, src, num_elements);
            return true;
        case GGUFType::Q3_K:
            dequant_q3_k(out, src, num_elements);
            return true;
        case GGUFType::Q4_K:
            dequant_q4_k(out, src, num_elements);
            return true;
        case GGUFType::Q5_K:
            dequant_q5_k(out, src, num_elements);
            return true;
        case GGUFType::Q6_K:
            dequant_q6_k(out, src, num_elements);
            return true;
        default:
            // For unsupported quant types, zero-fill and warn.
            fprintf(stderr, "  warning: unsupported GGUF type %u — zero-filling\n",
                    static_cast<unsigned>(type));
            std::memset(out, 0, num_elements * 4);
            return true;
    }
}

}  // anonymous namespace

// ─── Main conversion entry point ───────────────────────────────────────────

bool GGUFImporter::convert(const std::string& gguf_path,
                           const std::string& nxf_path,
                           Codec target_codec) {
    fprintf(stderr, "nexus: converting '%s' -> '%s'\n", gguf_path.c_str(), nxf_path.c_str());

    // Open GGUF file
    int gguf_fd = ::open(gguf_path.c_str(), O_RDONLY);
    if (gguf_fd < 0) {
        fprintf(stderr, "gguf: failed to open '%s': %s\n", gguf_path.c_str(), strerror(errno));
        return false;
    }

    // Get file size for validation
    struct stat st{};
    if (::fstat(gguf_fd, &st) < 0) {
        fprintf(stderr, "gguf: fstat failed: %s\n", strerror(errno));
        ::close(gguf_fd);
        return false;
    }
    uint64_t gguf_file_size = static_cast<uint64_t>(st.st_size);
    fprintf(stderr, "gguf: file size = %llu bytes (%.2f GB)\n",
            (unsigned long long)gguf_file_size,
            static_cast<double>(gguf_file_size) / (1024.0 * 1024.0 * 1024.0));

    // Parse GGUF structure
    GGUFFile gguf{};
    if (!parse_gguf(gguf_fd, gguf)) {
        ::close(gguf_fd);
        return false;
    }

    // Create NXF writer
    auto writer = format::NXFWriter::create(nxf_path);
    if (!writer) {
        fprintf(stderr, "nxf: failed to create '%s'\n", nxf_path.c_str());
        ::close(gguf_fd);
        return false;
    }

    // Build and write manifest
    format::ModelManifest manifest = build_manifest(gguf, target_codec);
    writer->set_manifest(manifest);

    // Process each tensor
    size_t tensors_written   = 0;
    size_t tensors_skipped   = 0;
    uint64_t total_bytes_in  = 0;
    uint64_t total_bytes_out = 0;

    for (size_t i = 0; i < gguf.tensors.size(); ++i) {
        const GGUFTensorDesc& td = gguf.tensors[i];

        // Map name
        std::string nxf_name = map_tensor_name(td.name);
        if (nxf_name.empty()) {
            fprintf(stderr, "  [%zu/%llu] skip: %s\n",
                    i + 1, (unsigned long long)gguf.tensor_count, td.name.c_str());
            ++tensors_skipped;
            continue;
        }

        // Compute element count and data sizes
        uint64_t num_elements = 1;
        for (auto d : td.dims) num_elements *= d;
        size_t src_data_size = tensor_data_size(td);

        if (src_data_size == 0 || num_elements == 0) {
            fprintf(stderr, "  [%zu/%llu] skip (zero-size): %s\n",
                    i + 1, (unsigned long long)gguf.tensor_count, td.name.c_str());
            ++tensors_skipped;
            continue;
        }

        // Choose output codec
        Codec  out_codec;
        DType  out_dtype;
        choose_codec(td.type, target_codec, out_codec, out_dtype);

        // Build shape vector (convert u64 -> i64)
        std::vector<int64_t> shape;
        shape.reserve(td.dims.size());
        for (auto d : td.dims) shape.push_back(static_cast<int64_t>(d));

        fprintf(stderr, "  [%zu/%llu] %s -> %s  (%llu elems, %zu bytes)\n",
                i + 1, (unsigned long long)gguf.tensor_count,
                td.name.c_str(), nxf_name.c_str(),
                (unsigned long long)num_elements, src_data_size);

        // Read source tensor data from GGUF
        uint64_t abs_offset = gguf.data_start + td.offset;
        if (abs_offset + src_data_size > gguf_file_size) {
            fprintf(stderr, "  WARNING: tensor data extends past end of file "
                    "(offset=0x%llX, size=%zu, file_size=%llu) — skipping\n",
                    (unsigned long long)abs_offset, src_data_size,
                    (unsigned long long)gguf_file_size);
            continue;  // Skip this tensor but keep converting the rest
        }

        std::vector<uint8_t> src_buf(src_data_size);
        if (!pread_exact(gguf_fd, src_buf.data(), src_data_size,
                         static_cast<off_t>(abs_offset))) {
            fprintf(stderr, "  ERROR: failed to read tensor data\n");
            ::close(gguf_fd);
            return false;
        }
        total_bytes_in += src_data_size;

        // Determine whether we need dequant+requant or can passthrough.
        bool need_dequant = false;
        switch (td.type) {
            case GGUFType::F32:
            case GGUFType::F16:
            case GGUFType::BF16:
            case GGUFType::F64:
                // Unquantized: dequant to FP32 if target requires quantization.
                need_dequant = (out_codec == Codec::INT4 || out_codec == Codec::INT8);
                break;
            case GGUFType::Q4_0:
            case GGUFType::Q4_1:
            case GGUFType::Q8_0:
                // Already quantized at the target bitwidth: need dequant for re-packing
                // into NEXUS's canonical INT4/INT8 layout.
                need_dequant = true;
                break;
            default:
                // K-quants and exotic types: always dequant.
                need_dequant = true;
                break;
        }

        // --- Path A: dequant to FP32, then quantize to target ---
        if (need_dequant && (out_codec == Codec::INT4 || out_codec == Codec::INT8)) {
            // Dequant to FP32
            size_t ne = static_cast<size_t>(num_elements);
            std::vector<float> fp32_buf(ne);
            dequant_gguf_to_fp32(fp32_buf.data(), src_buf.data(), td.type, ne);
            src_buf.clear();  // Free raw data

            if (out_codec == Codec::INT4) {
                // Quantize FP32 -> INT4 using NEXUS quant_int4
                size_t group_size   = 128;
                size_t num_groups   = (ne + group_size - 1) / group_size;
                size_t packed_bytes = (ne + 1) / 2;  // 2 values per byte

                std::vector<uint8_t> packed(packed_bytes);
                std::vector<float>   scales(num_groups);
                std::vector<float>   zeros(num_groups);

                quant::quant_int4(packed.data(), scales.data(), zeros.data(),
                                  fp32_buf.data(), static_cast<int>(ne),
                                  static_cast<int>(group_size));

                // Build the NXF chunk: [packed_data][scales][zeros]
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
                // INT8: simple round-and-clamp per group
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
        }
        // --- Path B: dequant to FP32, write as FP32 ---
        else if (need_dequant) {
            size_t ne = static_cast<size_t>(num_elements);
            std::vector<float> fp32_buf(ne);
            dequant_gguf_to_fp32(fp32_buf.data(), src_buf.data(), td.type, ne);
            src_buf.clear();

            size_t data_size = ne * sizeof(float);
            writer->begin_tensor(nxf_name, shape, DType::F32);
            writer->add_chunk(fp32_buf.data(), data_size, Codec::FP32, 0);
            writer->end_tensor();

            total_bytes_out += data_size;
        }
        // --- Path C: passthrough (FP32/FP16 source, FP32/FP16 target) ---
        else {
            // For FP16 passthrough with FP16 target:
            if (td.type == GGUFType::F16 && out_codec == Codec::FP16) {
                writer->begin_tensor(nxf_name, shape, DType::F16);
                writer->add_chunk(src_buf.data(), src_buf.size(), Codec::FP16, 0);
                writer->end_tensor();
                total_bytes_out += src_buf.size();
            }
            // FP32 source, FP32 target:
            else {
                writer->begin_tensor(nxf_name, shape, DType::F32);
                writer->add_chunk(src_buf.data(), src_buf.size(), Codec::FP32, 0);
                writer->end_tensor();
                total_bytes_out += src_buf.size();
            }
        }

        ++tensors_written;
    }

    // Finalize NXF file
    writer->finalize();

    // ─── Extract and save tokenizer vocabulary ────────────────────────────
    {
        VocabData vocab;
        if (VocabExtractor::extract_from_gguf(gguf_fd, gguf, vocab)) {
            // Derive output directory from nxf_path.
            std::string out_dir;
            size_t slash = nxf_path.find_last_of('/');
            if (slash != std::string::npos) {
                out_dir = nxf_path.substr(0, slash + 1);
            } else {
                out_dir = "./";
            }

            // Save vocab.txt next to the NXF file.
            VocabExtractor::save_vocab_file(vocab.tokens, out_dir + "vocab.txt");

            // Save merges.txt if merge rules were found.
            if (!vocab.merges.empty()) {
                VocabExtractor::save_merges_file(vocab.merges, out_dir + "merges.txt");
            }

            fprintf(stderr, "[nexus] Tokenizer vocabulary saved alongside NXF file\n");
        } else {
            fprintf(stderr, "[nexus] WARNING: Could not extract tokenizer vocabulary from GGUF\n");
        }
    }

    ::close(gguf_fd);

    fprintf(stderr, "\nnexus: conversion complete\n");
    fprintf(stderr, "  tensors written : %zu\n", tensors_written);
    fprintf(stderr, "  tensors skipped : %zu\n", tensors_skipped);
    fprintf(stderr, "  input size      : %.2f GB\n",
            static_cast<double>(total_bytes_in) / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "  output size     : %.2f GB\n",
            static_cast<double>(total_bytes_out) / (1024.0 * 1024.0 * 1024.0));

    return true;
}

}  // namespace nexus::import
