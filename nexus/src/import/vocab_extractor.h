#pragma once
/// NEXUS Vocab Extractor — Extracts tokenizer vocabulary from GGUF metadata.
///
/// GGUF files store the tokenizer vocabulary as an array of strings under
/// the metadata key "tokenizer.ggml.tokens".  BPE merge rules are stored
/// under "tokenizer.ggml.merges" and token type flags under
/// "tokenizer.ggml.token_type".
///
/// This module reads those arrays directly from the GGUF file (using the
/// already-parsed metadata offsets) and serializes/deserializes them as
/// plain-text files so they can be bundled alongside NXF model files.

#include "import/gguf_importer.h"
#include <cstdint>
#include <string>
#include <vector>

namespace nexus::import {

/// Result of vocabulary extraction from a GGUF file.
struct VocabData {
    std::vector<std::string> tokens;    // Ordered token strings (index = token ID)
    std::vector<std::string> merges;    // BPE merge rules ("left right")
    std::vector<uint32_t>    types;     // Per-token type flags (0=normal, 3=control, ...)
};

class VocabExtractor {
public:
    /// Extract the tokenizer vocabulary from a GGUF file.
    ///
    /// This re-reads the GGUF metadata section to parse the string arrays
    /// that the normal GGUF importer skips (it only stores scalar KVs).
    ///
    /// @param gguf_fd     Open file descriptor for the GGUF file.
    /// @param metadata    Already-parsed metadata vector (used to locate
    ///                    the array entries and determine offsets).
    /// @param data        Output vocab data.
    /// @return true if at least the tokens array was successfully extracted.
    static bool extract_from_gguf(int gguf_fd,
                                  const GGUFFile& gguf,
                                  VocabData& data);

    /// Save vocabulary tokens to a plain-text file (one token per line).
    /// Byte tokens containing non-printable characters are stored as
    /// hex escapes: <0xHH>.
    static bool save_vocab_file(const std::vector<std::string>& tokens,
                                const std::string& path);

    /// Save BPE merge rules to a plain-text file (one rule per line).
    static bool save_merges_file(const std::vector<std::string>& merges,
                                 const std::string& path);

    /// Load vocabulary tokens from a plain-text file (one token per line).
    static std::vector<std::string> load_vocab_file(const std::string& path);

    /// Load BPE merge rules from a plain-text file (one rule per line).
    static std::vector<std::string> load_merges_file(const std::string& path);

private:
    /// Read a GGUF string-array metadata value at the given file offset.
    /// The offset should point to the start of the array header (element_type u32).
    static bool read_string_array(int fd, off_t offset,
                                  std::vector<std::string>& out);

    /// Read a GGUF uint32-array metadata value at the given file offset.
    static bool read_uint32_array(int fd, off_t offset,
                                  std::vector<uint32_t>& out);
};

}  // namespace nexus::import
