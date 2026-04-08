/// NEXUS CLI — Command-line interface for the inference engine.
///
/// Usage:
///   nexus run <model.nxf> --prompt "Hello"
///   nexus convert <input.gguf> <output.nxf> [--quant q4]
///   nexus info <model.nxf>
///   nexus serve <model.nxf> --port 8080
///   nexus bench <model.nxf>

#include "core/engine.h"
#include <cstdio>
#include <cstring>
#include <string>

// Defined in import/convert_cli.cpp
extern int cmd_convert(int argc, char** argv);

// Defined in api/bench_cli.cpp
extern int cmd_bench(int argc, char** argv);

// Defined in api/serve_cli.cpp
extern int cmd_serve(int argc, char** argv);

static void print_banner() {
    fprintf(stderr,
        "\n"
        "  ╔═══════════════════════════════════════════════════╗\n"
        "  ║  NEXUS Inference Engine v0.1.0                    ║\n"
        "  ║  Compression-native LLM runtime for Apple Silicon ║\n"
        "  ╚═══════════════════════════════════════════════════╝\n"
        "\n"
    );
}

static void print_usage() {
    fprintf(stderr,
        "Usage:\n"
        "  nexus run <model.nxf> [options]      Run inference\n"
        "  nexus convert <in> <out> [options]    Convert GGUF/safetensors to NXF\n"
        "  nexus info <model.nxf>                Show model information\n"
        "  nexus serve <model.nxf> [options]      Start OpenAI-compatible API server\n"
        "  nexus bench <model.nxf>               Run benchmarks\n"
        "\n"
        "Run options:\n"
        "  --prompt <text>       Input prompt\n"
        "  --max-tokens <n>      Maximum tokens to generate (default: 2048)\n"
        "  --temperature <f>     Sampling temperature (default: 0.7)\n"
        "  --top-p <f>           Top-p sampling (default: 0.9)\n"
        "  --top-k <n>           Top-k sampling (default: 40)\n"
        "  --ram-limit <gb>      RAM limit in GB (default: 48)\n"
        "  --no-metal            Disable Metal GPU acceleration\n"
        "  --threads <n>         Number of CPU threads (default: auto)\n"
        "\n"
        "Serve options:\n"
        "  --port <n>            Port to listen on (default: 8080)\n"
        "  --ram-limit <gb>      RAM limit in GB (default: 48)\n"
        "  --no-metal            Disable Metal GPU acceleration\n"
        "  --threads <n>         Number of CPU threads (default: auto)\n"
        "\n"
        "Convert options:\n"
        "  --quant <q4|q8|f16>   Quantization format (default: q4)\n"
        "\n"
    );
}

static int cmd_run(int argc, char** argv) {
    if (argc < 1) {
        fprintf(stderr, "Error: model path required\n");
        return 1;
    }

    std::string model_path = argv[0];
    nexus::EngineConfig config;
    nexus::SamplingParams params;
    std::string prompt;

    // Parse options
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            params.max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            params.temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            params.top_p = atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            params.top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ram-limit") == 0 && i + 1 < argc) {
            config.memory.ram_limit = static_cast<size_t>(atof(argv[++i]) * 1024 * 1024 * 1024);
        } else if (strcmp(argv[i], "--no-metal") == 0) {
            config.use_metal = false;
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            config.num_threads = atoi(argv[++i]);
        }
    }

    if (prompt.empty()) {
        fprintf(stderr, "Error: --prompt is required\n");
        return 1;
    }

    // Create engine
    auto engine = nexus::Engine::create(model_path, config);
    if (!engine) {
        fprintf(stderr, "Error: failed to load model\n");
        return 1;
    }

    // Generate with streaming output
    engine->generate(prompt, params, [](const std::string& token) {
        printf("%s", token.c_str());
        fflush(stdout);
    });

    printf("\n");
    return 0;
}

static int cmd_info(int argc, char** argv) {
    if (argc < 1) {
        fprintf(stderr, "Error: model path required\n");
        return 1;
    }

    auto reader = nexus::format::NXFReader::open(argv[0]);
    if (!reader) {
        fprintf(stderr, "Error: failed to open %s\n", argv[0]);
        return 1;
    }

    const auto& m = reader->manifest();
    printf("Model:         %s\n", m.name.c_str());
    printf("Architecture:  %s\n", m.architecture.c_str());
    printf("Layers:        %u\n", m.num_layers);
    printf("Hidden dim:    %u\n", m.hidden_dim);
    printf("Heads:         %u (KV: %u)\n", m.num_heads, m.num_kv_heads);
    printf("Head dim:      %u\n", m.head_dim);
    printf("Vocab size:    %u\n", m.vocab_size);
    printf("Max seq len:   %u\n", m.max_seq_len);
    printf("RoPE theta:    %.1f\n", m.rope_theta);

    if (m.num_experts > 0) {
        printf("MoE experts:   %u (active: %u)\n", m.num_experts, m.num_active_experts);
    }

    printf("Default codec: %u\n", static_cast<unsigned>(m.default_codec));
    printf("File size:     %.2f GB\n", reader->file_size() / (1024.0 * 1024.0 * 1024.0));

    auto names = reader->tensor_names();
    printf("Tensors:       %zu\n", names.size());

    return 0;
}

int main(int argc, char** argv) {
    print_banner();

    if (argc < 2) {
        print_usage();
        return 1;
    }

    const char* cmd = argv[1];

    if (strcmp(cmd, "run") == 0) {
        return cmd_run(argc - 2, argv + 2);
    } else if (strcmp(cmd, "info") == 0) {
        return cmd_info(argc - 2, argv + 2);
    } else if (strcmp(cmd, "convert") == 0) {
        return cmd_convert(argc - 2, argv + 2);
    } else if (strcmp(cmd, "serve") == 0) {
        return cmd_serve(argc - 2, argv + 2);
    } else if (strcmp(cmd, "bench") == 0) {
        return cmd_bench(argc - 2, argv + 2);
    } else if (strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) {
        print_usage();
        return 0;
    } else {
        fprintf(stderr, "Unknown command: %s\n", cmd);
        print_usage();
        return 1;
    }
}
