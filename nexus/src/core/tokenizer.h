#pragma once
/// NEXUS Tokenizer — BPE tokenizer for Qwen / tiktoken-style models.
///
/// Supports loading vocabulary from:
///   - GGUF metadata (tokenizer.ggml.tokens / tokenizer.ggml.merges)
///   - A plain-text vocab file (one token per line)
///   - An NXF manifest that references an external vocab file
///
/// Encoding uses greedy longest-match against the vocabulary, which is a
/// practical approximation of full BPE merge iteration and works well for
/// byte-level BPE vocabularies like those used by Qwen / GPT-2 / tiktoken.

#include "format/nxf.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nexus {

/// Byte-level BPE tokenizer.
class Tokenizer {
public:
    Tokenizer() = default;

    // ── Loading ────────────────────────────────────────────────────────────

    /// Load vocabulary from GGUF metadata key-value pairs.
    /// Looks for "tokenizer.ggml.tokens" (string array) and optionally
    /// "tokenizer.ggml.merges" (string array) and "tokenizer.ggml.token_type"
    /// (uint32 array).
    ///
    /// @param tokens   Ordered vector of token strings extracted from GGUF.
    /// @param merges   BPE merge rules (optional; empty = greedy mode only).
    /// @param types    Per-token type flags (optional; 0 = normal).
    /// @return true if at least one token was loaded.
    bool load_from_gguf_metadata(const std::vector<std::string>& tokens,
                                 const std::vector<std::string>& merges = {},
                                 const std::vector<uint32_t>& types = {});

    /// Load a plain-text vocab file (one token per line, UTF-8).
    /// Lines may contain raw bytes represented as hex escapes (\\xHH).
    bool load_from_vocab_file(const std::string& path);

    /// Try to find and load a vocab file referenced by the NXF manifest.
    /// Checks for "<model_dir>/vocab.txt" next to the NXF file.
    bool load_from_nxf_manifest(const format::ModelManifest& manifest,
                                const std::string& nxf_path);

    // ── Encode / Decode ────────────────────────────────────────────────────

    /// Encode UTF-8 text into a sequence of token IDs.
    /// Uses BPE merges when available, otherwise greedy longest-match.
    std::vector<int32_t> encode(const std::string& text) const;

    /// Decode a sequence of token IDs back into UTF-8 text.
    std::string decode(const std::vector<int32_t>& tokens) const;

    // ── Queries ────────────────────────────────────────────────────────────

    /// Number of tokens in the vocabulary.
    size_t vocab_size() const { return id_to_token_.size(); }

    /// Whether the tokenizer has been loaded successfully.
    bool is_loaded() const { return !id_to_token_.empty(); }

    /// Get the string representation of a single token ID.
    /// Returns empty string for out-of-range IDs.
    const std::string& id_to_token(int32_t id) const;

    /// Look up a token string and return its ID, or -1 if not found.
    int32_t token_to_id(const std::string& token) const;

    // ── Special tokens ─────────────────────────────────────────────────────

    /// Common special token IDs (set to -1 if not present).
    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    int32_t pad_id() const { return pad_id_; }
    int32_t unk_id() const { return unk_id_; }

private:
    /// Ordered list of token strings (index = token ID).
    std::vector<std::string> id_to_token_;

    /// Reverse map: token string -> token ID.
    std::unordered_map<std::string, int32_t> token_to_id_;

    /// BPE merge rules: each entry is a pair of token strings that merge.
    /// Ordered by priority (earlier = higher priority).
    struct MergeRule {
        std::string left;
        std::string right;
        std::string merged;  // left + right
    };
    std::vector<MergeRule> merges_;

    /// Merge priority lookup: "left right" -> priority index.
    std::unordered_map<std::string, int> merge_priority_;

    /// Per-token type flags (from GGUF tokenizer.ggml.token_type).
    /// 0 = normal, 1 = unknown, 2 = control, 3 = user_defined, etc.
    std::vector<uint32_t> token_types_;

    /// Special token IDs.
    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t pad_id_ = -1;
    int32_t unk_id_ = -1;

    /// Empty string returned for out-of-range lookups.
    static const std::string kEmptyToken;

    /// Build the reverse mapping and detect special tokens.
    void build_index();

    /// BPE encode: split text into byte tokens, then iteratively merge.
    std::vector<int32_t> bpe_encode(const std::string& text) const;

    /// Greedy longest-match encode (fallback when no merges available).
    std::vector<int32_t> greedy_encode(const std::string& text) const;

    /// Decode hex escape sequences in a token string (e.g. "<0x0A>" -> '\n').
    static std::string decode_token_escapes(const std::string& raw);
};

}  // namespace nexus
