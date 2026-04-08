#pragma once
/// NEXUS Inference Engine — Benchmarking Suite (Phase 5)
///
/// Measures latency (TTFT, ITL), throughput (prefill tok/s, decode tok/s),
/// memory footprint (peak RSS, weight/KV/scratch breakdown), and collects
/// Apple Silicon system information.

#include "core/engine.h"
#include "core/config.h"
#include <chrono>
#include <string>
#include <vector>

namespace nexus {

// ─── System information ────────────────────────────────────────────────────
struct SystemInfo {
    std::string device_name;        // e.g. "MacBook Pro (Apple M4 Max)"
    std::string chip_name;          // e.g. "Apple M4 Max"
    std::string os_version;         // e.g. "macOS 15.4.1"
    uint64_t    ram_bytes  = 0;     // Total physical RAM
    uint32_t    gpu_cores  = 0;     // Metal GPU core count
    uint32_t    cpu_cores  = 0;     // Physical CPU cores
    uint32_t    cpu_perf_cores = 0; // Performance cores
    uint32_t    cpu_eff_cores  = 0; // Efficiency cores
};

// ─── Memory breakdown ──────────────────────────────────────────────────────
struct MemoryBreakdown {
    size_t peak_rss_bytes   = 0;    // Peak RSS via mach_task_info
    size_t weight_bytes     = 0;    // Memory used by model weights
    size_t kv_cache_bytes   = 0;    // Memory used by KV cache
    size_t scratch_bytes    = 0;    // Memory used by scratch/activations
    size_t other_bytes      = 0;    // Everything else (overhead, etc.)
};

// ─── Latency measurements ──────────────────────────────────────────────────
struct LatencyStats {
    double ttft_ms          = 0.0;  // Time to first token (ms)
    double mean_itl_ms      = 0.0;  // Mean inter-token latency (ms)
    double p50_itl_ms       = 0.0;  // Median ITL
    double p90_itl_ms       = 0.0;  // 90th percentile ITL
    double p99_itl_ms       = 0.0;  // 99th percentile ITL
    double total_time_ms    = 0.0;  // Total generation time
};

// ─── Throughput measurements ───────────────────────────────────────────────
struct ThroughputStats {
    double prefill_tok_per_s = 0.0; // Prompt processing speed
    double decode_tok_per_s  = 0.0; // Token generation speed
    int    prompt_tokens      = 0;  // Number of input tokens
    int    generated_tokens   = 0;  // Number of output tokens
};

// ─── Full benchmark result ─────────────────────────────────────────────────
struct BenchmarkResult {
    std::string       model_name;
    std::string       model_path;
    std::string       timestamp;      // ISO-8601

    SystemInfo        system;
    MemoryBreakdown   memory;
    LatencyStats      latency;
    ThroughputStats   throughput;

    // Configuration used
    EngineConfig      engine_config;
    SamplingParams    sampling;
    std::string       prompt;
    int               max_tokens = 0;
};

// ─── Benchmark runner ──────────────────────────────────────────────────────
class BenchmarkRunner {
public:
    /// Construct with a reference to an already-loaded engine.
    explicit BenchmarkRunner(Engine& engine);
    ~BenchmarkRunner();

    /// Collect Apple Silicon system information.
    static SystemInfo collect_system_info();

    /// Measure peak RSS via mach_task_info (bytes).
    static size_t measure_peak_rss();

    /// Run a latency benchmark: single prompt, measure TTFT and ITL.
    BenchmarkResult run_latency_bench(const std::string& prompt,
                                      int max_tokens = 256);

    /// Run a throughput benchmark: multiple prompts, measure tok/s.
    BenchmarkResult run_throughput_bench(const std::vector<std::string>& prompts,
                                         int max_tokens = 256);

    /// Run a memory benchmark: measure RSS breakdown after model load.
    BenchmarkResult run_memory_bench(const std::string& model_path);

    /// Print a formatted report table to stderr.
    static void print_report(const BenchmarkResult& result);

    /// Export benchmark result as JSON to the given file path.
    static void export_json(const BenchmarkResult& result,
                            const std::string& path);

private:
    Engine& engine_;
};

}  // namespace nexus
