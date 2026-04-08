#pragma once
/// NEXUS Engine — Top-level orchestrator for inference.
///
/// Coordinates: NXF loading, memory management, layer-by-layer streaming,
/// KV caching, compute dispatch, and sampling.

#include "core/config.h"
#include "format/nxf.h"
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace nexus {

/// Callback for streaming token output.
using TokenCallback = std::function<void(const std::string& token)>;

/// Engine configuration.
struct EngineConfig {
    MemoryConfig memory;
    SamplingParams sampling;
    int num_threads = 0;          // 0 = auto-detect
    bool use_metal = true;        // Use Metal GPU acceleration
    bool use_mmap = true;         // Use mmap for NXF loading
    int prefetch_layers = 2;      // Number of layers to prefetch ahead
};

/// Main inference engine.
class Engine {
public:
    ~Engine();

    /// Create an engine and load a model from an NXF file.
    static std::unique_ptr<Engine> create(const std::string& model_path,
                                           const EngineConfig& config = {});

    /// Generate text from a prompt. Calls callback for each token.
    /// Returns the full generated text.
    std::string generate(const std::string& prompt,
                         const SamplingParams& params = {},
                         TokenCallback callback = nullptr);

    /// Tokenize a string into token IDs.
    std::vector<int32_t> tokenize(const std::string& text) const;

    /// Decode token IDs back to text.
    std::string detokenize(const std::vector<int32_t>& tokens) const;

    /// Get model info.
    const format::ModelManifest& model_info() const;

    /// Get current memory usage in bytes.
    size_t memory_usage() const;

    /// Reset KV cache (start new conversation).
    void reset_cache();

private:
    Engine() = default;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nexus
