/// NEXUS Inference Engine — Benchmarking Suite Implementation (Phase 5)

#include "api/benchmark.h"
#include "memory/memory_manager.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// macOS / Apple Silicon headers
#include <mach/mach.h>
#include <mach/task_info.h>
#include <sys/sysctl.h>
#include <sys/types.h>

namespace nexus {

// ─── Helpers ───────────────────────────────────────────────────────────────

using Clock = std::chrono::high_resolution_clock;

/// Query a sysctl string value.
static std::string sysctl_string(const char* name) {
    size_t len = 0;
    if (sysctlbyname(name, nullptr, &len, nullptr, 0) != 0) return "";
    std::string buf(len, '\0');
    if (sysctlbyname(name, buf.data(), &len, nullptr, 0) != 0) return "";
    // Strip trailing null
    while (!buf.empty() && buf.back() == '\0') buf.pop_back();
    return buf;
}

/// Query a sysctl uint32 value.
static uint32_t sysctl_u32(const char* name) {
    uint32_t val = 0;
    size_t len = sizeof(val);
    sysctlbyname(name, &val, &len, nullptr, 0);
    return val;
}

/// Query a sysctl uint64 value.
static uint64_t sysctl_u64(const char* name) {
    uint64_t val = 0;
    size_t len = sizeof(val);
    sysctlbyname(name, &val, &len, nullptr, 0);
    return val;
}

/// Format bytes as human-readable string.
static std::string format_bytes(size_t bytes) {
    char buf[64];
    if (bytes >= 1024ULL * 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
    } else if (bytes >= 1024ULL * 1024) {
        snprintf(buf, sizeof(buf), "%.2f MB", bytes / (1024.0 * 1024.0));
    } else if (bytes >= 1024) {
        snprintf(buf, sizeof(buf), "%.2f KB", bytes / 1024.0);
    } else {
        snprintf(buf, sizeof(buf), "%zu B", bytes);
    }
    return buf;
}

/// Current ISO-8601 timestamp.
static std::string iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    struct tm tm_buf;
    gmtime_r(&t, &tm_buf);
    char buf[64];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);
    return buf;
}

/// Compute percentile from a sorted vector of doubles.
static double percentile(const std::vector<double>& sorted_vals, double p) {
    if (sorted_vals.empty()) return 0.0;
    if (sorted_vals.size() == 1) return sorted_vals[0];
    double idx = (p / 100.0) * static_cast<double>(sorted_vals.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) return sorted_vals[lo];
    double frac = idx - static_cast<double>(lo);
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac;
}

/// Escape a string for JSON output.
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char hex[8];
                    snprintf(hex, sizeof(hex), "\\u%04x", static_cast<unsigned>(c));
                    out += hex;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// ─── BenchmarkRunner ───────────────────────────────────────────────────────

BenchmarkRunner::BenchmarkRunner(Engine& engine)
    : engine_(engine) {}

BenchmarkRunner::~BenchmarkRunner() = default;

// ─── System info collection ────────────────────────────────────────────────

SystemInfo BenchmarkRunner::collect_system_info() {
    SystemInfo info;

    // Device / chip
    info.device_name = sysctl_string("hw.model");
    info.chip_name   = sysctl_string("machdep.cpu.brand_string");

    // If the brand string is empty (Apple Silicon), try to infer from model
    if (info.chip_name.empty()) {
        // On Apple Silicon, machdep.cpu.brand_string is "Apple" or empty.
        // We can read the chip name from IORegistry, but sysctl is simpler:
        info.chip_name = sysctl_string("machdep.cpu.brand_string");
        if (info.chip_name.empty()) {
            info.chip_name = "Apple Silicon";
        }
    }

    // OS version
    std::string os_release = sysctl_string("kern.osproductversion");
    if (!os_release.empty()) {
        info.os_version = "macOS " + os_release;
    } else {
        info.os_version = sysctl_string("kern.osrelease");
    }

    // RAM
    info.ram_bytes = sysctl_u64("hw.memsize");

    // CPU cores
    info.cpu_cores     = sysctl_u32("hw.physicalcpu");
    info.cpu_perf_cores = sysctl_u32("hw.perflevel0.physicalcpu");
    info.cpu_eff_cores  = sysctl_u32("hw.perflevel1.physicalcpu");

    // GPU cores — Apple does not expose this via sysctl directly.
    // We query IOKit indirectly; fall back to 0 if not available.
    // A common heuristic: read from machdep.gpu or use Metal API.
    // For now, try known sysctl keys.
    info.gpu_cores = sysctl_u32("machdep.gpu.core_count");
    if (info.gpu_cores == 0) {
        // Fallback: try alternate path used on some macOS versions
        info.gpu_cores = sysctl_u32("hw.gpu_core_count");
    }

    return info;
}

// ─── Peak RSS measurement ──────────────────────────────────────────────────

size_t BenchmarkRunner::measure_peak_rss() {
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(),
                                 MACH_TASK_BASIC_INFO,
                                 reinterpret_cast<task_info_t>(&info),
                                 &count);
    if (kr != KERN_SUCCESS) return 0;
    // resident_size is current RSS; there is no direct "peak" in this struct,
    // but we can use phys_footprint from task_vm_info for a better measure.
    task_vm_info_data_t vm_info;
    mach_msg_type_number_t vm_count = TASK_VM_INFO_COUNT;
    kr = task_info(mach_task_self(),
                   TASK_VM_INFO,
                   reinterpret_cast<task_info_t>(&vm_info),
                   &vm_count);
    if (kr == KERN_SUCCESS) {
        return static_cast<size_t>(vm_info.phys_footprint);
    }
    return static_cast<size_t>(info.resident_size);
}

// ─── Latency benchmark ────────────────────────────────────────────────────

BenchmarkResult BenchmarkRunner::run_latency_bench(const std::string& prompt,
                                                    int max_tokens) {
    BenchmarkResult result;
    result.model_name  = engine_.model_info().name;
    result.timestamp   = iso_timestamp();
    result.system      = collect_system_info();
    result.prompt      = prompt;
    result.max_tokens  = max_tokens;

    // Tokenize prompt to count input tokens
    auto prompt_tokens = engine_.tokenize(prompt);
    result.throughput.prompt_tokens = static_cast<int>(prompt_tokens.size());

    // Reset cache for a clean measurement
    engine_.reset_cache();

    // Set up timing
    std::vector<double> itl_samples;
    itl_samples.reserve(max_tokens);

    auto gen_start      = Clock::now();
    auto first_tok_time = Clock::time_point{};
    auto prev_tok_time  = Clock::time_point{};
    int  token_count    = 0;
    bool first_token    = true;

    // Configure sampling
    SamplingParams params;
    params.max_tokens = max_tokens;
    params.temperature = 0.0f;  // Greedy for reproducibility

    // Generate with per-token timing
    engine_.generate(prompt, params, [&](const std::string& /*token*/) {
        auto now = Clock::now();
        if (first_token) {
            first_tok_time = now;
            first_token = false;
        } else {
            double itl = std::chrono::duration<double, std::milli>(
                now - prev_tok_time).count();
            itl_samples.push_back(itl);
        }
        prev_tok_time = now;
        token_count++;
    });

    auto gen_end = Clock::now();

    // Compute latency stats
    double total_ms = std::chrono::duration<double, std::milli>(
        gen_end - gen_start).count();

    result.latency.total_time_ms = total_ms;
    result.latency.ttft_ms = std::chrono::duration<double, std::milli>(
        first_tok_time - gen_start).count();

    if (!itl_samples.empty()) {
        double sum = std::accumulate(itl_samples.begin(), itl_samples.end(), 0.0);
        result.latency.mean_itl_ms = sum / static_cast<double>(itl_samples.size());

        std::sort(itl_samples.begin(), itl_samples.end());
        result.latency.p50_itl_ms = percentile(itl_samples, 50.0);
        result.latency.p90_itl_ms = percentile(itl_samples, 90.0);
        result.latency.p99_itl_ms = percentile(itl_samples, 99.0);
    }

    // Throughput
    result.throughput.generated_tokens = token_count;

    // Prefill tok/s: prompt tokens processed in TTFT
    if (result.latency.ttft_ms > 0.0) {
        result.throughput.prefill_tok_per_s =
            static_cast<double>(result.throughput.prompt_tokens) /
            (result.latency.ttft_ms / 1000.0);
    }

    // Decode tok/s: generated tokens (excluding first) over time after TTFT
    double decode_ms = total_ms - result.latency.ttft_ms;
    if (decode_ms > 0.0 && token_count > 1) {
        result.throughput.decode_tok_per_s =
            static_cast<double>(token_count - 1) / (decode_ms / 1000.0);
    }

    // Memory snapshot
    result.memory.peak_rss_bytes = measure_peak_rss();
    result.memory.weight_bytes   = engine_.memory_usage();  // Approximate

    return result;
}

// ─── Throughput benchmark ──────────────────────────────────────────────────

BenchmarkResult BenchmarkRunner::run_throughput_bench(
        const std::vector<std::string>& prompts, int max_tokens) {
    BenchmarkResult result;
    result.model_name = engine_.model_info().name;
    result.timestamp  = iso_timestamp();
    result.system     = collect_system_info();
    result.max_tokens = max_tokens;

    int total_prompt_tokens = 0;
    int total_gen_tokens    = 0;
    double total_ttft_ms    = 0.0;
    double total_decode_ms  = 0.0;
    std::vector<double> all_itl;

    SamplingParams params;
    params.max_tokens  = max_tokens;
    params.temperature = 0.0f;

    for (const auto& prompt : prompts) {
        engine_.reset_cache();

        auto prompt_tokens = engine_.tokenize(prompt);
        total_prompt_tokens += static_cast<int>(prompt_tokens.size());

        auto gen_start     = Clock::now();
        auto first_tok     = Clock::time_point{};
        auto prev_tok      = Clock::time_point{};
        int  tok_count     = 0;
        bool is_first      = true;

        engine_.generate(prompt, params, [&](const std::string& /*token*/) {
            auto now = Clock::now();
            if (is_first) {
                first_tok = now;
                is_first = false;
            } else {
                double itl = std::chrono::duration<double, std::milli>(
                    now - prev_tok).count();
                all_itl.push_back(itl);
            }
            prev_tok = now;
            tok_count++;
        });

        auto gen_end = Clock::now();

        double ttft = std::chrono::duration<double, std::milli>(
            first_tok - gen_start).count();
        double total = std::chrono::duration<double, std::milli>(
            gen_end - gen_start).count();

        total_ttft_ms   += ttft;
        total_decode_ms += (total - ttft);
        total_gen_tokens += tok_count;
    }

    // Aggregate
    result.throughput.prompt_tokens    = total_prompt_tokens;
    result.throughput.generated_tokens = total_gen_tokens;

    double avg_ttft = prompts.empty() ? 0.0 : total_ttft_ms / prompts.size();
    result.latency.ttft_ms = avg_ttft;
    result.latency.total_time_ms = total_ttft_ms + total_decode_ms;

    if (!all_itl.empty()) {
        double sum = std::accumulate(all_itl.begin(), all_itl.end(), 0.0);
        result.latency.mean_itl_ms = sum / static_cast<double>(all_itl.size());
        std::sort(all_itl.begin(), all_itl.end());
        result.latency.p50_itl_ms = percentile(all_itl, 50.0);
        result.latency.p90_itl_ms = percentile(all_itl, 90.0);
        result.latency.p99_itl_ms = percentile(all_itl, 99.0);
    }

    if (total_ttft_ms > 0.0) {
        result.throughput.prefill_tok_per_s =
            static_cast<double>(total_prompt_tokens) /
            (total_ttft_ms / 1000.0);
    }
    if (total_decode_ms > 0.0 && total_gen_tokens > static_cast<int>(prompts.size())) {
        result.throughput.decode_tok_per_s =
            static_cast<double>(total_gen_tokens - static_cast<int>(prompts.size())) /
            (total_decode_ms / 1000.0);
    }

    result.memory.peak_rss_bytes = measure_peak_rss();

    return result;
}

// ─── Memory benchmark ──────────────────────────────────────────────────────

BenchmarkResult BenchmarkRunner::run_memory_bench(const std::string& model_path) {
    BenchmarkResult result;
    result.model_path = model_path;
    result.timestamp  = iso_timestamp();
    result.system     = collect_system_info();

    // Measure baseline RSS before model is loaded
    size_t baseline_rss = measure_peak_rss();

    // Create a fresh engine to measure loading cost
    EngineConfig cfg;
    auto fresh_engine = Engine::create(model_path, cfg);
    if (!fresh_engine) {
        fprintf(stderr, "[bench] Error: failed to load model for memory benchmark\n");
        return result;
    }

    size_t post_load_rss = measure_peak_rss();
    result.model_name = fresh_engine->model_info().name;

    // Weight memory approximation
    result.memory.weight_bytes = fresh_engine->memory_usage();

    // Run a short generation to populate KV cache
    SamplingParams params;
    params.max_tokens = 32;
    params.temperature = 0.0f;

    fresh_engine->generate("Hello, world!", params, nullptr);

    size_t post_gen_rss = measure_peak_rss();

    result.memory.peak_rss_bytes = post_gen_rss;
    result.memory.kv_cache_bytes = post_gen_rss - post_load_rss;

    // Scratch is approximated as total minus weights and KV
    size_t model_delta = post_load_rss - baseline_rss;
    if (model_delta > result.memory.weight_bytes) {
        result.memory.scratch_bytes = model_delta - result.memory.weight_bytes;
    }
    result.memory.other_bytes =
        result.memory.peak_rss_bytes -
        result.memory.weight_bytes -
        result.memory.kv_cache_bytes -
        result.memory.scratch_bytes;

    return result;
}

// ─── Report printing ───────────────────────────────────────────────────────

void BenchmarkRunner::print_report(const BenchmarkResult& r) {
    fprintf(stderr, "\n");
    fprintf(stderr, "  ╔═══════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "  ║              NEXUS Benchmark Report                       ║\n");
    fprintf(stderr, "  ╚═══════════════════════════════════════════════════════════╝\n");
    fprintf(stderr, "\n");

    // System info
    fprintf(stderr, "  System\n");
    fprintf(stderr, "  ────────────────────────────────────────────────────────────\n");
    fprintf(stderr, "  %-24s %s\n", "Device:",      r.system.device_name.c_str());
    fprintf(stderr, "  %-24s %s\n", "Chip:",        r.system.chip_name.c_str());
    fprintf(stderr, "  %-24s %s\n", "OS:",          r.system.os_version.c_str());
    fprintf(stderr, "  %-24s %s\n", "RAM:",         format_bytes(r.system.ram_bytes).c_str());
    fprintf(stderr, "  %-24s %u P + %u E = %u total\n", "CPU Cores:",
            r.system.cpu_perf_cores, r.system.cpu_eff_cores, r.system.cpu_cores);
    fprintf(stderr, "  %-24s %u\n", "GPU Cores:",   r.system.gpu_cores);
    fprintf(stderr, "\n");

    // Model info
    fprintf(stderr, "  Model\n");
    fprintf(stderr, "  ────────────────────────────────────────────────────────────\n");
    fprintf(stderr, "  %-24s %s\n", "Name:",  r.model_name.c_str());
    if (!r.model_path.empty())
        fprintf(stderr, "  %-24s %s\n", "Path:", r.model_path.c_str());
    fprintf(stderr, "  %-24s %s\n", "Timestamp:", r.timestamp.c_str());
    fprintf(stderr, "\n");

    // Memory
    fprintf(stderr, "  Memory\n");
    fprintf(stderr, "  ────────────────────────────────────────────────────────────\n");
    fprintf(stderr, "  %-24s %s\n", "Peak RSS:",     format_bytes(r.memory.peak_rss_bytes).c_str());
    fprintf(stderr, "  %-24s %s\n", "Weights:",      format_bytes(r.memory.weight_bytes).c_str());
    fprintf(stderr, "  %-24s %s\n", "KV Cache:",     format_bytes(r.memory.kv_cache_bytes).c_str());
    fprintf(stderr, "  %-24s %s\n", "Scratch:",      format_bytes(r.memory.scratch_bytes).c_str());
    fprintf(stderr, "  %-24s %s\n", "Other:",        format_bytes(r.memory.other_bytes).c_str());
    fprintf(stderr, "\n");

    // Latency
    fprintf(stderr, "  Latency\n");
    fprintf(stderr, "  ────────────────────────────────────────────────────────────\n");
    fprintf(stderr, "  %-24s %.2f ms\n", "TTFT:",        r.latency.ttft_ms);
    fprintf(stderr, "  %-24s %.2f ms\n", "Mean ITL:",    r.latency.mean_itl_ms);
    fprintf(stderr, "  %-24s %.2f ms\n", "P50 ITL:",     r.latency.p50_itl_ms);
    fprintf(stderr, "  %-24s %.2f ms\n", "P90 ITL:",     r.latency.p90_itl_ms);
    fprintf(stderr, "  %-24s %.2f ms\n", "P99 ITL:",     r.latency.p99_itl_ms);
    fprintf(stderr, "  %-24s %.2f ms\n", "Total time:",  r.latency.total_time_ms);
    fprintf(stderr, "\n");

    // Throughput
    fprintf(stderr, "  Throughput\n");
    fprintf(stderr, "  ────────────────────────────────────────────────────────────\n");
    fprintf(stderr, "  %-24s %.1f tok/s\n", "Prefill:",  r.throughput.prefill_tok_per_s);
    fprintf(stderr, "  %-24s %.1f tok/s\n", "Decode:",   r.throughput.decode_tok_per_s);
    fprintf(stderr, "  %-24s %d\n", "Prompt tokens:",    r.throughput.prompt_tokens);
    fprintf(stderr, "  %-24s %d\n", "Generated tokens:", r.throughput.generated_tokens);
    fprintf(stderr, "\n");
}

// ─── JSON export ───────────────────────────────────────────────────────────

void BenchmarkRunner::export_json(const BenchmarkResult& r,
                                   const std::string& path) {
    std::ofstream out(path);
    if (!out.is_open()) {
        fprintf(stderr, "[bench] Error: cannot open %s for writing\n", path.c_str());
        return;
    }

    out << "{\n";
    out << "  \"engine\": \"nexus\",\n";
    out << "  \"version\": \"0.1.0\",\n";
    out << "  \"model_name\": \"" << json_escape(r.model_name) << "\",\n";
    out << "  \"model_path\": \"" << json_escape(r.model_path) << "\",\n";
    out << "  \"timestamp\": \"" << json_escape(r.timestamp) << "\",\n";

    // System
    out << "  \"system\": {\n";
    out << "    \"device_name\": \"" << json_escape(r.system.device_name) << "\",\n";
    out << "    \"chip_name\": \"" << json_escape(r.system.chip_name) << "\",\n";
    out << "    \"os_version\": \"" << json_escape(r.system.os_version) << "\",\n";
    out << "    \"ram_bytes\": " << r.system.ram_bytes << ",\n";
    out << "    \"gpu_cores\": " << r.system.gpu_cores << ",\n";
    out << "    \"cpu_cores\": " << r.system.cpu_cores << ",\n";
    out << "    \"cpu_perf_cores\": " << r.system.cpu_perf_cores << ",\n";
    out << "    \"cpu_eff_cores\": " << r.system.cpu_eff_cores << "\n";
    out << "  },\n";

    // Memory
    out << "  \"memory\": {\n";
    out << "    \"peak_rss_bytes\": " << r.memory.peak_rss_bytes << ",\n";
    out << "    \"weight_bytes\": " << r.memory.weight_bytes << ",\n";
    out << "    \"kv_cache_bytes\": " << r.memory.kv_cache_bytes << ",\n";
    out << "    \"scratch_bytes\": " << r.memory.scratch_bytes << ",\n";
    out << "    \"other_bytes\": " << r.memory.other_bytes << "\n";
    out << "  },\n";

    // Latency
    out << "  \"latency\": {\n";
    out << "    \"ttft_ms\": " << r.latency.ttft_ms << ",\n";
    out << "    \"mean_itl_ms\": " << r.latency.mean_itl_ms << ",\n";
    out << "    \"p50_itl_ms\": " << r.latency.p50_itl_ms << ",\n";
    out << "    \"p90_itl_ms\": " << r.latency.p90_itl_ms << ",\n";
    out << "    \"p99_itl_ms\": " << r.latency.p99_itl_ms << ",\n";
    out << "    \"total_time_ms\": " << r.latency.total_time_ms << "\n";
    out << "  },\n";

    // Throughput
    out << "  \"throughput\": {\n";
    out << "    \"prefill_tok_per_s\": " << r.throughput.prefill_tok_per_s << ",\n";
    out << "    \"decode_tok_per_s\": " << r.throughput.decode_tok_per_s << ",\n";
    out << "    \"prompt_tokens\": " << r.throughput.prompt_tokens << ",\n";
    out << "    \"generated_tokens\": " << r.throughput.generated_tokens << "\n";
    out << "  },\n";

    // Config
    out << "  \"config\": {\n";
    out << "    \"max_tokens\": " << r.max_tokens << ",\n";
    out << "    \"prompt\": \"" << json_escape(r.prompt) << "\"\n";
    out << "  }\n";

    out << "}\n";

    out.close();
    fprintf(stderr, "[bench] Results exported to %s\n", path.c_str());
}

}  // namespace nexus
