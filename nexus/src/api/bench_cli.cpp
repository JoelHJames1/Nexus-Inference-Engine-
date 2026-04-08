/// NEXUS Inference Engine — Benchmark CLI entry point (Phase 5)
///
/// Called from main CLI as: nexus bench <model.nxf> [options]
///
/// Options:
///   --prompt <text>       Prompt to benchmark (default: built-in)
///   --max-tokens <n>      Max tokens to generate (default: 128)
///   --output <path>       Export JSON results to file
///   --latency             Run latency benchmark only
///   --throughput          Run throughput benchmark only
///   --memory              Run memory benchmark only

#include "api/benchmark.h"
#include "core/engine.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

/// Default benchmark prompt when none is supplied.
static const char* kDefaultPrompt =
    "Explain the concept of unified memory architecture in Apple Silicon "
    "and how it benefits large language model inference.";

/// Additional prompts for throughput benchmarking.
static const std::vector<std::string> kThroughputPrompts = {
    "What is the time complexity of quicksort in the average case?",
    "Describe the transformer architecture and self-attention mechanism.",
    "Write a Python function that computes the Fibonacci sequence iteratively.",
    "Explain the difference between DRAM and SRAM in modern processors.",
};

int cmd_bench(int argc, char** argv) {
    if (argc < 1) {
        fprintf(stderr, "Error: model path required for bench command\n");
        fprintf(stderr, "Usage: nexus bench <model.nxf> [options]\n");
        return 1;
    }

    std::string model_path = argv[0];
    std::string prompt      = kDefaultPrompt;
    std::string output_path;
    int max_tokens = 128;

    bool run_latency    = false;
    bool run_throughput  = false;
    bool run_memory      = false;

    // Parse options
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (std::strcmp(argv[i], "--latency") == 0) {
            run_latency = true;
        } else if (std::strcmp(argv[i], "--throughput") == 0) {
            run_throughput = true;
        } else if (std::strcmp(argv[i], "--memory") == 0) {
            run_memory = true;
        } else {
            fprintf(stderr, "Warning: unknown option '%s'\n", argv[i]);
        }
    }

    // If no specific benchmark requested, run all
    bool run_all = !run_latency && !run_throughput && !run_memory;

    // Print system info first
    fprintf(stderr, "[bench] Collecting system information...\n");
    auto sys_info = nexus::BenchmarkRunner::collect_system_info();
    fprintf(stderr, "[bench] Device: %s\n", sys_info.device_name.c_str());
    fprintf(stderr, "[bench] Chip:   %s\n", sys_info.chip_name.c_str());
    fprintf(stderr, "[bench] RAM:    %.0f GB\n",
            sys_info.ram_bytes / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "\n");

    // Load model
    fprintf(stderr, "[bench] Loading model: %s\n", model_path.c_str());
    nexus::EngineConfig config;
    auto engine = nexus::Engine::create(model_path, config);
    if (!engine) {
        fprintf(stderr, "Error: failed to load model '%s'\n", model_path.c_str());
        return 1;
    }
    fprintf(stderr, "[bench] Model loaded: %s\n",
            engine->model_info().name.c_str());
    fprintf(stderr, "\n");

    nexus::BenchmarkRunner runner(*engine);
    nexus::BenchmarkResult final_result;

    // ── Latency benchmark ──────────────────────────────────────────────────
    if (run_all || run_latency) {
        fprintf(stderr, "[bench] === Latency Benchmark ===\n");
        fprintf(stderr, "[bench] Prompt: \"%.*s%s\"\n",
                60, prompt.c_str(), prompt.size() > 60 ? "..." : "");
        fprintf(stderr, "[bench] Max tokens: %d\n", max_tokens);
        fprintf(stderr, "[bench] Running...\n");

        auto lat_result = runner.run_latency_bench(prompt, max_tokens);
        nexus::BenchmarkRunner::print_report(lat_result);

        // Use latency result as the final result (merge later if needed)
        final_result = lat_result;
        final_result.model_path = model_path;
    }

    // ── Throughput benchmark ───────────────────────────────────────────────
    if (run_all || run_throughput) {
        fprintf(stderr, "[bench] === Throughput Benchmark ===\n");

        // Build prompt list: user prompt + built-in prompts
        std::vector<std::string> prompts;
        prompts.push_back(prompt);
        for (const auto& p : kThroughputPrompts) {
            prompts.push_back(p);
        }

        fprintf(stderr, "[bench] %zu prompts, %d max tokens each\n",
                prompts.size(), max_tokens);
        fprintf(stderr, "[bench] Running...\n");

        auto tp_result = runner.run_throughput_bench(prompts, max_tokens);

        // Merge throughput into final result
        final_result.throughput = tp_result.throughput;
        if (final_result.model_name.empty()) {
            final_result = tp_result;
            final_result.model_path = model_path;
        }

        fprintf(stderr, "\n");
        fprintf(stderr, "  Throughput (multi-prompt aggregate)\n");
        fprintf(stderr, "  ────────────────────────────────────────────────────────────\n");
        fprintf(stderr, "  %-24s %.1f tok/s\n", "Prefill:",
                tp_result.throughput.prefill_tok_per_s);
        fprintf(stderr, "  %-24s %.1f tok/s\n", "Decode:",
                tp_result.throughput.decode_tok_per_s);
        fprintf(stderr, "  %-24s %d\n", "Total prompt tokens:",
                tp_result.throughput.prompt_tokens);
        fprintf(stderr, "  %-24s %d\n", "Total generated tokens:",
                tp_result.throughput.generated_tokens);
        fprintf(stderr, "\n");
    }

    // ── Memory benchmark ───────────────────────────────────────────────────
    if (run_all || run_memory) {
        fprintf(stderr, "[bench] === Memory Benchmark ===\n");
        fprintf(stderr, "[bench] Running...\n");

        auto mem_result = runner.run_memory_bench(model_path);

        // Merge memory into final result
        final_result.memory = mem_result.memory;
        if (final_result.model_name.empty()) {
            final_result = mem_result;
        }

        fprintf(stderr, "\n");
        fprintf(stderr, "  Memory Breakdown\n");
        fprintf(stderr, "  ────────────────────────────────────────────────────────────\n");

        auto fmt = [](size_t b) -> std::string {
            char buf[64];
            if (b >= 1024ULL * 1024 * 1024)
                snprintf(buf, sizeof(buf), "%.2f GB", b / (1024.0*1024.0*1024.0));
            else if (b >= 1024ULL * 1024)
                snprintf(buf, sizeof(buf), "%.2f MB", b / (1024.0*1024.0));
            else
                snprintf(buf, sizeof(buf), "%zu B", b);
            return buf;
        };

        fprintf(stderr, "  %-24s %s\n", "Peak RSS:",  fmt(mem_result.memory.peak_rss_bytes).c_str());
        fprintf(stderr, "  %-24s %s\n", "Weights:",   fmt(mem_result.memory.weight_bytes).c_str());
        fprintf(stderr, "  %-24s %s\n", "KV Cache:",  fmt(mem_result.memory.kv_cache_bytes).c_str());
        fprintf(stderr, "  %-24s %s\n", "Scratch:",   fmt(mem_result.memory.scratch_bytes).c_str());
        fprintf(stderr, "  %-24s %s\n", "Other:",     fmt(mem_result.memory.other_bytes).c_str());
        fprintf(stderr, "\n");
    }

    // ── Export JSON ────────────────────────────────────────────────────────
    if (!output_path.empty()) {
        nexus::BenchmarkRunner::export_json(final_result, output_path);
    }

    fprintf(stderr, "[bench] Done.\n");
    return 0;
}
