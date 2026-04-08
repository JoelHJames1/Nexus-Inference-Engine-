/// NEXUS Engine — Top-level orchestrator implementation.

#include "core/engine.h"
#include "core/tokenizer.h"
#include "memory/memory_manager.h"
#include "model/transformer.h"
#include "model/hybrid_model.h"
#include "compute/compute_dispatch.h"
#include <cstdio>
#include <chrono>
#include <thread>

namespace nexus {

/// All models now use HybridModel which has the INT4 dequant bridge,
/// fused GPU GEMV, resident mode, and handles both dense and hybrid architectures.
static bool is_hybrid_architecture(const std::string& /*arch*/) {
    return true;  // HybridModel handles everything
}

struct Engine::Impl {
    EngineConfig config;
    std::unique_ptr<format::NXFReader> reader;
    std::unique_ptr<MemoryManager> memory;
    std::unique_ptr<model::Transformer> model;
    std::unique_ptr<model::HybridModel> hybrid_model;
    std::unique_ptr<compute::ComputeDispatch> compute;
    Tokenizer tokenizer;
    format::ModelManifest manifest;
    std::string model_path;
    bool using_hybrid = false;
};

Engine::~Engine() = default;

std::unique_ptr<Engine> Engine::create(const std::string& model_path,
                                        const EngineConfig& config) {
    auto engine = std::unique_ptr<Engine>(new Engine());
    engine->impl_ = std::make_unique<Impl>();
    engine->impl_->config = config;

    // Open NXF file
    fprintf(stderr, "[nexus] Loading model: %s\n", model_path.c_str());
    engine->impl_->reader = format::NXFReader::open(model_path);
    if (!engine->impl_->reader) {
        fprintf(stderr, "[nexus] ERROR: Failed to open model file: %s\n", model_path.c_str());
        return nullptr;
    }

    engine->impl_->manifest = engine->impl_->reader->manifest();
    engine->impl_->model_path = model_path;
    fprintf(stderr, "[nexus] Model: %s (%s)\n",
            engine->impl_->manifest.name.c_str(),
            engine->impl_->manifest.architecture.c_str());
    fprintf(stderr, "[nexus] Layers: %u, Hidden: %u, Heads: %u/%u, Vocab: %u\n",
            engine->impl_->manifest.num_layers,
            engine->impl_->manifest.hidden_dim,
            engine->impl_->manifest.num_heads,
            engine->impl_->manifest.num_kv_heads,
            engine->impl_->manifest.vocab_size);

    // Load tokenizer vocabulary from files adjacent to the model.
    if (!engine->impl_->tokenizer.load_from_nxf_manifest(
            engine->impl_->manifest, model_path)) {
        fprintf(stderr, "[nexus] WARNING: No tokenizer vocabulary loaded; "
                "using byte-level fallback.\n");
        fprintf(stderr, "[nexus] To enable proper tokenization, run the GGUF "
                "converter with --extract-vocab or place vocab.txt next to the model.\n");
    } else {
        fprintf(stderr, "[nexus] Tokenizer: %zu tokens loaded\n",
                engine->impl_->tokenizer.vocab_size());
    }

    // Initialize memory manager
    engine->impl_->memory = std::make_unique<MemoryManager>(config.memory);
    fprintf(stderr, "[nexus] Memory limit: %.1f GB\n",
            config.memory.ram_limit / (1024.0 * 1024.0 * 1024.0));

    // Initialize GPU compute
    engine->impl_->compute = std::make_unique<compute::ComputeDispatch>();
    compute::set_global_compute(engine->impl_->compute.get());
    if (config.use_metal) {
        // Try to find metallib next to the executable or in share/nexus
        std::string shader_path = "nexus_shaders.metallib";
        if (engine->impl_->compute->init_gpu(shader_path)) {
            fprintf(stderr, "[nexus] GPU compute: %s\n",
                    engine->impl_->compute->gpu_name().c_str());
        }
    } else {
        fprintf(stderr, "[nexus] GPU disabled by user\n");
    }

    // Initialize model — choose HybridModel for hybrid architectures,
    // standard Transformer for everything else.
    if (is_hybrid_architecture(engine->impl_->manifest.architecture)) {
        fprintf(stderr, "[nexus] Detected hybrid SSM+Attention architecture: %s\n",
                engine->impl_->manifest.architecture.c_str());
        engine->impl_->hybrid_model = model::HybridModel::create(
            engine->impl_->manifest,
            *engine->impl_->reader,
            *engine->impl_->memory
        );
        if (!engine->impl_->hybrid_model) {
            fprintf(stderr, "[nexus] ERROR: Failed to initialize hybrid model\n");
            return nullptr;
        }
        engine->impl_->using_hybrid = true;
    } else {
        engine->impl_->model = model::Transformer::create(
            engine->impl_->manifest,
            *engine->impl_->reader,
            *engine->impl_->memory
        );
        if (!engine->impl_->model) {
            fprintf(stderr, "[nexus] ERROR: Failed to initialize transformer\n");
            return nullptr;
        }
    }

    fprintf(stderr, "[nexus] Engine ready.\n");
    return engine;
}

std::string Engine::generate(const std::string& prompt,
                              const SamplingParams& params,
                              TokenCallback callback) {
    if (!impl_) return "";
    if (!impl_->using_hybrid && !impl_->model) return "";
    if (impl_->using_hybrid && !impl_->hybrid_model) return "";

    auto start = std::chrono::high_resolution_clock::now();

    // Tokenize
    auto tokens = tokenize(prompt);
    fprintf(stderr, "[nexus] Prompt tokens: %zu\n", tokens.size());

    // Prefill
    auto prefill_start = std::chrono::high_resolution_clock::now();
    if (impl_->using_hybrid) {
        impl_->hybrid_model->prefill(tokens);
    } else {
        impl_->model->prefill(tokens);
    }
    auto prefill_end = std::chrono::high_resolution_clock::now();

    double prefill_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
    fprintf(stderr, "[nexus] Prefill: %.1f ms (%.1f tokens/s)\n",
            prefill_ms, tokens.size() * 1000.0 / prefill_ms);

    // Decode loop
    std::string output;
    int generated = 0;
    auto decode_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < params.max_tokens; i++) {
        int32_t next_token;
        if (impl_->using_hybrid) {
            next_token = impl_->hybrid_model->decode_step(params);
        } else {
            next_token = impl_->model->decode_step(params);
        }

        // Check for EOS (token -1 = error, EOS token varies by model)
        if (next_token < 0) break;

        std::string token_str = detokenize({next_token});
        output += token_str;
        generated++;

        if (callback) {
            callback(token_str);
        }
    }

    auto decode_end = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();

    fprintf(stderr, "\n[nexus] Generated %d tokens in %.1f ms (%.1f tokens/s)\n",
            generated, decode_ms, generated * 1000.0 / decode_ms);
    fprintf(stderr, "[nexus] Peak memory: %.1f MB\n",
            memory_usage() / (1024.0 * 1024.0));

    return output;
}

std::vector<int32_t> Engine::tokenize(const std::string& text) const {
    if (impl_->tokenizer.is_loaded()) {
        return impl_->tokenizer.encode(text);
    }
    // Byte-level fallback when no vocabulary is available.
    std::vector<int32_t> tokens;
    tokens.reserve(text.size());
    for (unsigned char c : text) {
        tokens.push_back(static_cast<int32_t>(c));
    }
    return tokens;
}

std::string Engine::detokenize(const std::vector<int32_t>& tokens) const {
    if (impl_->tokenizer.is_loaded()) {
        return impl_->tokenizer.decode(tokens);
    }
    // Byte-level fallback when no vocabulary is available.
    std::string text;
    for (int32_t t : tokens) {
        if (t >= 0 && t < 256) {
            text += static_cast<char>(t);
        }
    }
    return text;
}

const format::ModelManifest& Engine::model_info() const {
    return impl_->manifest;
}

size_t Engine::memory_usage() const {
    if (impl_ && impl_->memory) {
        return impl_->memory->current_rss();
    }
    return 0;
}

void Engine::reset_cache() {
    if (impl_) {
        if (impl_->using_hybrid && impl_->hybrid_model) {
            impl_->hybrid_model->reset_kv_cache();
        } else if (impl_->model) {
            impl_->model->reset_kv_cache();
        }
    }
}

}  // namespace nexus
