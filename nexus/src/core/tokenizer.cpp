/// NEXUS Tokenizer — BPE tokenizer implementation.

#include "core/tokenizer.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

namespace nexus {

const std::string Tokenizer::kEmptyToken;

// ─── Loading from GGUF metadata ───────────────────────────────────────────

bool Tokenizer::load_from_gguf_metadata(const std::vector<std::string>& tokens,
                                         const std::vector<std::string>& merges,
                                         const std::vector<uint32_t>& types) {
    if (tokens.empty()) {
        fprintf(stderr, "[tokenizer] WARNING: empty token list from GGUF metadata\n");
        return false;
    }

    id_to_token_ = tokens;
    token_types_ = types;

    // Parse merge rules: each line is "token_a token_b"
    merges_.clear();
    merge_priority_.clear();
    for (size_t i = 0; i < merges.size(); ++i) {
        const std::string& line = merges[i];
        // Find the first space that splits the merge rule.
        // BPE merge lines are "left right" where left and right can contain spaces
        // only in very rare cases; for byte-level BPE the tokens are short, so
        // first-space split is correct.
        size_t sp = line.find(' ');
        if (sp == std::string::npos || sp == 0 || sp == line.size() - 1) {
            continue;  // Malformed merge rule
        }
        MergeRule rule;
        rule.left   = line.substr(0, sp);
        rule.right  = line.substr(sp + 1);
        rule.merged = rule.left + rule.right;

        std::string key = rule.left + " " + rule.right;
        merge_priority_[key] = static_cast<int>(i);
        merges_.push_back(std::move(rule));
    }

    build_index();

    fprintf(stderr, "[tokenizer] Loaded %zu tokens, %zu merge rules from GGUF metadata\n",
            id_to_token_.size(), merges_.size());
    return true;
}

// ─── Loading from vocab file ──────────────────────────────────────────────

bool Tokenizer::load_from_vocab_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        fprintf(stderr, "[tokenizer] WARNING: cannot open vocab file: %s\n", path.c_str());
        return false;
    }

    id_to_token_.clear();
    std::string line;
    while (std::getline(ifs, line)) {
        // Remove trailing \r if present (Windows line endings)
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        // Decode hex escape sequences like <0xHH>
        std::string token = decode_token_escapes(line);
        id_to_token_.push_back(std::move(token));
    }

    if (id_to_token_.empty()) {
        fprintf(stderr, "[tokenizer] WARNING: vocab file is empty: %s\n", path.c_str());
        return false;
    }

    build_index();

    fprintf(stderr, "[tokenizer] Loaded %zu tokens from vocab file: %s\n",
            id_to_token_.size(), path.c_str());
    return true;
}

// ─── Loading from NXF manifest ────────────────────────────────────────────

bool Tokenizer::load_from_nxf_manifest(const format::ModelManifest& /*manifest*/,
                                        const std::string& nxf_path) {
    // Try to find vocab.txt next to the NXF file.
    // E.g. /path/to/model.nxf -> /path/to/vocab.txt
    std::string dir;
    size_t slash = nxf_path.find_last_of('/');
    if (slash != std::string::npos) {
        dir = nxf_path.substr(0, slash + 1);
    } else {
        dir = "./";
    }

    std::string vocab_path = dir + "vocab.txt";
    bool loaded = load_from_vocab_file(vocab_path);

    if (!loaded) {
        // Also try vocab.txt in parent directory (common layout)
        std::string parent_vocab = dir + "../vocab.txt";
        loaded = load_from_vocab_file(parent_vocab);
        if (loaded) dir = dir + "../";
    }

    if (!loaded) {
        fprintf(stderr, "[tokenizer] No vocab file found near: %s\n", nxf_path.c_str());
        return false;
    }

    // Try to also load BPE merge rules from merges.txt in the same directory.
    std::string merges_path = dir + "merges.txt";
    std::ifstream mfs(merges_path, std::ios::binary);
    if (mfs.is_open()) {
        std::vector<std::string> merge_lines;
        std::string line;
        while (std::getline(mfs, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) merge_lines.push_back(std::move(line));
        }
        mfs.close();

        if (!merge_lines.empty()) {
            merges_.clear();
            merge_priority_.clear();
            for (size_t i = 0; i < merge_lines.size(); ++i) {
                size_t sp = merge_lines[i].find(' ');
                if (sp == std::string::npos || sp == 0 || sp == merge_lines[i].size() - 1)
                    continue;
                MergeRule rule;
                rule.left   = merge_lines[i].substr(0, sp);
                rule.right  = merge_lines[i].substr(sp + 1);
                rule.merged = rule.left + rule.right;
                merge_priority_[rule.left + " " + rule.right] = static_cast<int>(i);
                merges_.push_back(std::move(rule));
            }
            fprintf(stderr, "[tokenizer] Loaded %zu merge rules from: %s\n",
                    merges_.size(), merges_path.c_str());
        }
    }

    return true;
}

// ─── Build index ──────────────────────────────────────────────────────────

void Tokenizer::build_index() {
    token_to_id_.clear();
    token_to_id_.reserve(id_to_token_.size());

    for (size_t i = 0; i < id_to_token_.size(); ++i) {
        token_to_id_[id_to_token_[i]] = static_cast<int32_t>(i);
    }

    // Detect common special tokens by content
    auto find_special = [&](const std::string& text) -> int32_t {
        auto it = token_to_id_.find(text);
        if (it != token_to_id_.end()) return it->second;
        return -1;
    };

    // Qwen uses these special tokens:
    bos_id_ = find_special("<|im_start|>");
    if (bos_id_ < 0) bos_id_ = find_special("<s>");
    if (bos_id_ < 0) bos_id_ = find_special("<bos>");

    eos_id_ = find_special("<|im_end|>");
    if (eos_id_ < 0) eos_id_ = find_special("<|endoftext|>");
    if (eos_id_ < 0) eos_id_ = find_special("</s>");
    if (eos_id_ < 0) eos_id_ = find_special("<eos>");

    pad_id_ = find_special("<|endoftext|>");
    if (pad_id_ < 0) pad_id_ = find_special("<pad>");

    unk_id_ = find_special("<unk>");

    // Also check token_types_ for special token detection
    if (!token_types_.empty() && token_types_.size() == id_to_token_.size()) {
        for (size_t i = 0; i < token_types_.size(); ++i) {
            // Type 3 is typically "control" in GGUF; check known strings
            if (token_types_[i] == 3) {  // control token
                const auto& t = id_to_token_[i];
                if (t == "<s>" || t == "<|im_start|>" || t == "<bos>") {
                    if (bos_id_ < 0) bos_id_ = static_cast<int32_t>(i);
                }
                if (t == "</s>" || t == "<|im_end|>" || t == "<|endoftext|>" || t == "<eos>") {
                    if (eos_id_ < 0) eos_id_ = static_cast<int32_t>(i);
                }
            }
        }
    }

    fprintf(stderr, "[tokenizer] Index built: vocab_size=%zu  bos=%d  eos=%d  unk=%d\n",
            id_to_token_.size(), bos_id_, eos_id_, unk_id_);
}

// ─── Queries ──────────────────────────────────────────────────────────────

const std::string& Tokenizer::id_to_token(int32_t id) const {
    if (id >= 0 && static_cast<size_t>(id) < id_to_token_.size()) {
        return id_to_token_[static_cast<size_t>(id)];
    }
    return kEmptyToken;
}

int32_t Tokenizer::token_to_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) return it->second;
    return -1;
}

// ─── Encode ───────────────────────────────────────────────────────────────

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    if (text.empty() || id_to_token_.empty()) return {};

    // Use BPE merges if available, otherwise greedy longest-match.
    if (!merge_priority_.empty()) {
        return bpe_encode(text);
    }
    return greedy_encode(text);
}

// ─── BPE encode ───────────────────────────────────────────────────────────

std::vector<int32_t> Tokenizer::bpe_encode(const std::string& text) const {
    // Step 1: Split text into individual UTF-8 bytes as initial tokens.
    // For byte-level BPE (Qwen/GPT-2 style), each byte is a token.
    std::vector<std::string> symbols;
    symbols.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        // Check if this byte is represented as a single-byte token in vocab.
        std::string byte_str(1, text[i]);
        symbols.push_back(byte_str);
    }

    // Step 2: Iteratively merge the highest-priority adjacent pair.
    // This is the classic BPE algorithm.
    bool changed = true;
    while (changed && symbols.size() >= 2) {
        changed = false;

        // Find the pair with the highest priority (lowest index in merges_).
        int best_priority = INT_MAX;
        size_t best_pos = 0;

        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            std::string key = symbols[i] + " " + symbols[i + 1];
            auto it = merge_priority_.find(key);
            if (it != merge_priority_.end() && it->second < best_priority) {
                best_priority = it->second;
                best_pos = i;
            }
        }

        if (best_priority < INT_MAX) {
            // Merge the pair at best_pos.
            symbols[best_pos] = symbols[best_pos] + symbols[best_pos + 1];
            symbols.erase(symbols.begin() + static_cast<long>(best_pos) + 1);
            changed = true;
        }
    }

    // Step 3: Convert merged symbols to token IDs.
    std::vector<int32_t> ids;
    ids.reserve(symbols.size());
    for (const auto& sym : symbols) {
        auto it = token_to_id_.find(sym);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            // Token not found in vocabulary; fall back to encoding each byte
            // as an individual token or use <unk>.
            bool found_bytes = true;
            std::vector<int32_t> byte_ids;
            for (unsigned char c : sym) {
                // Try the byte as a single-character token.
                std::string b(1, static_cast<char>(c));
                auto bit = token_to_id_.find(b);
                if (bit != token_to_id_.end()) {
                    byte_ids.push_back(bit->second);
                } else {
                    // Try GGUF byte token format: <0xHH>
                    char hex[8];
                    snprintf(hex, sizeof(hex), "<0x%02X>", c);
                    auto hit = token_to_id_.find(std::string(hex));
                    if (hit != token_to_id_.end()) {
                        byte_ids.push_back(hit->second);
                    } else {
                        found_bytes = false;
                        break;
                    }
                }
            }
            if (found_bytes) {
                ids.insert(ids.end(), byte_ids.begin(), byte_ids.end());
            } else if (unk_id_ >= 0) {
                ids.push_back(unk_id_);
            }
            // If no unk token either, silently drop the symbol.
        }
    }

    return ids;
}

// ─── Greedy longest-match encode ──────────────────────────────────────────

std::vector<int32_t> Tokenizer::greedy_encode(const std::string& text) const {
    std::vector<int32_t> ids;
    ids.reserve(text.size());

    size_t pos = 0;
    while (pos < text.size()) {
        // Try to match the longest token starting at pos.
        // Limit the maximum lookahead to a reasonable token length.
        int32_t best_id = -1;
        size_t best_len = 0;

        // Maximum token length to try (most BPE tokens are < 32 bytes).
        size_t max_len = std::min(text.size() - pos, static_cast<size_t>(64));

        for (size_t len = max_len; len >= 1; --len) {
            std::string candidate = text.substr(pos, len);
            auto it = token_to_id_.find(candidate);
            if (it != token_to_id_.end()) {
                best_id = it->second;
                best_len = len;
                break;
            }
        }

        if (best_id >= 0) {
            ids.push_back(best_id);
            pos += best_len;
        } else {
            // No token matched even a single byte; try byte-level fallback.
            unsigned char c = static_cast<unsigned char>(text[pos]);

            // Try GGUF byte token format: <0xHH>
            char hex[8];
            snprintf(hex, sizeof(hex), "<0x%02X>", c);
            auto hit = token_to_id_.find(std::string(hex));
            if (hit != token_to_id_.end()) {
                ids.push_back(hit->second);
            } else if (unk_id_ >= 0) {
                ids.push_back(unk_id_);
            }
            // If still nothing, skip the byte (avoids infinite loop).
            pos += 1;
        }
    }

    return ids;
}

// ─── Decode ───────────────────────────────────────────────────────────────

std::string Tokenizer::decode(const std::vector<int32_t>& tokens) const {
    std::string text;
    text.reserve(tokens.size() * 4);  // Rough estimate

    for (int32_t id : tokens) {
        if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) {
            continue;  // Skip invalid IDs
        }

        const std::string& token_str = id_to_token_[static_cast<size_t>(id)];

        // Skip control/special tokens during decode (they're metadata, not text).
        if (!token_types_.empty() && static_cast<size_t>(id) < token_types_.size()) {
            uint32_t ttype = token_types_[static_cast<size_t>(id)];
            if (ttype == 3) {
                // Control token — skip in output (e.g. <|im_start|>, <|im_end|>)
                continue;
            }
        }

        // Handle GGUF byte tokens like <0xHH>
        if (token_str.size() == 6 && token_str[0] == '<' && token_str[1] == '0'
            && token_str[2] == 'x' && token_str[5] == '>') {
            // Parse the hex byte.
            unsigned int byte_val = 0;
            if (sscanf(token_str.c_str(), "<0x%02X>", &byte_val) == 1 && byte_val < 256) {
                text += static_cast<char>(byte_val);
                continue;
            }
        }

        text += token_str;
    }

    return text;
}

// ─── Decode hex escapes in token strings ──────────────────────────────────

std::string Tokenizer::decode_token_escapes(const std::string& raw) {
    // Handle GGUF-style byte tokens: <0xHH>
    // These represent individual bytes and appear in the vocab as literal strings.
    // We keep them as-is in the token list since the encode/decode logic
    // handles them explicitly.
    return raw;
}

}  // namespace nexus
