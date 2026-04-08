#pragma once
/// NEXUS Scheduler — Coordinates layer streaming, prefetch, and compute dispatch.
///
/// Phase 2 upgrade: Hybrid CPU+GPU execution pipeline.
///   - CPU thread: entropy decode + dequantize via GCD queue
///   - GPU thread: Metal compute via command buffer pipeline
///   - Overlap: while GPU processes layer L, CPU prepares layer L+1
///
/// The scheduler maintains a state machine per layer:
///   COLD → PREFETCHING → READY → COMPUTING → EVICTED

#include "core/config.h"
#include <cstdint>
#include <functional>
#include <memory>

namespace nexus {

class MemoryManager;
class Prefetcher;

/// Layer execution callback: called with layer index.
using LayerCallback = std::function<void(uint32_t layer_idx)>;

/// Compute device preference.
enum class ComputeDevice {
    Auto,       // Let scheduler decide based on operation
    CPU,        // Force Accelerate/NEON (small batches, scalar ops)
    GPU,        // Force Metal (large batches, GEMM, attention)
};

/// Layer state in the streaming pipeline.
enum class LayerState {
    Cold,           // Not in memory
    Prefetching,    // Being loaded from SSD
    Ready,          // In memory, ready for compute
    Computing,      // Currently being processed on GPU/CPU
    Evicted,        // Marked for eviction
};

/// Scheduler configuration.
struct SchedulerConfig {
    int prefetch_window = 2;           // Layers to prefetch ahead
    ComputeDevice default_device = ComputeDevice::Auto;
    bool enable_double_buffer = true;  // Double-buffer weight streaming
    size_t weight_buffer_size = 256 * 1024 * 1024;  // 256 MB per buffer
};

/// Scheduler manages the pipeline of:
///   1. Prefetching next layer's weights from SSD (async, GCD)
///   2. CPU entropy-decode / dequantize (overlapped with GPU)
///   3. GPU compute current layer (Metal command buffer)
///   4. Evicting previous layer's weights (madvise DONTNEED)
class Scheduler {
public:
    Scheduler(MemoryManager& memory, uint32_t num_layers,
              const SchedulerConfig& config = {});
    ~Scheduler();

    /// Execute all layers in sequence with prefetch pipeline.
    void execute_layers(LayerCallback compute_fn);

    /// Signal that a layer's weights are no longer needed.
    void evict_layer(uint32_t layer_idx);

    /// Get current state of a layer.
    LayerState layer_state(uint32_t layer_idx) const;

    /// Get the prefetcher (for direct prefetch control).
    Prefetcher* prefetcher();

    /// Get config.
    const SchedulerConfig& config() const { return config_; }

    /// Choose compute device for an operation.
    /// Heuristic: GEMM with M>1 → GPU, single-token decode attention → GPU,
    /// small element-wise ops → CPU.
    static ComputeDevice choose_device(int M, int N, int K);

private:
    MemoryManager& memory_;
    uint32_t num_layers_;
    SchedulerConfig config_;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nexus
