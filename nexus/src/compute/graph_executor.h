#pragma once
/// NEXUS Graph Executor — MLX-style lazy evaluation for Metal.
///
/// Instead of dispatching each operation immediately (creating a command
/// buffer per dispatch), record operations into a graph, then execute
/// ALL of them on ONE command buffer with ONE commit.
///
/// Key difference from our previous token-batch attempt:
///   - Token-batch: each operation STILL copies CPU↔GPU per call
///   - Graph executor: operations reference GPU buffer IDs, NO CPU copies
///     between operations. Only copy at graph boundaries (input/output).
///
/// Usage:
///   graph.begin();
///   auto h = graph.input(hidden_state, dim);          // Upload once
///   auto n = graph.rmsnorm(h, norm_weight, dim, eps);  // Record, don't dispatch
///   auto q = graph.gemv_int4(n, wq, q_dim, dim);       // Record
///   auto k = graph.gemv_int4(n, wk, k_dim, dim);       // Record
///   ...
///   graph.output(result_buf, output_ptr, dim);          // Download once
///   graph.execute();  // ONE Metal commit for ALL recorded ops

#include "compute/metal/metal_backend.h"
#include <vector>
#include <cstdint>

namespace nexus::compute {

class ComputeDispatch;

/// A node in the computation graph — represents a GPU buffer.
using GraphNode = MetalBackend::buffer_id;

/// Operation types for the graph
enum class GraphOp {
    GEMV_INT4,       // INT4 GEMV: out = in @ weights
    RMSNORM,         // RMSNorm: out = norm(in) * weight
    SWIGLU,          // SwiGLU: out = silu(gate) * up
    RESIDUAL_ADD,    // Residual: out = a + b
    BUFFER_COPY,     // Copy: dst = src
    FUSED_FFN,       // Fused FFN: the whole thing
};

/// A recorded operation
struct GraphOperation {
    GraphOp op;
    MetalBackend::buffer_id inputs[4];   // Up to 4 input buffers
    MetalBackend::buffer_id output;       // Output buffer
    uint32_t params[4];                   // Integer params (dims, etc.)
    float fparams[2];                     // Float params (eps, etc.)
};

/// Graph executor — records operations, executes all in one commit.
class GraphExecutor {
public:
    GraphExecutor(ComputeDispatch& dispatch);

    /// Start recording a new graph
    void begin();

    /// Record a GEMV INT4 operation
    /// Returns the output buffer ID (for chaining)
    GraphNode gemv_int4(GraphNode input, MetalBackend::buffer_id weight,
                         GraphNode output, uint32_t N, uint32_t K);

    /// Record RMSNorm
    GraphNode rmsnorm(GraphNode input, MetalBackend::buffer_id weight,
                       GraphNode output, uint32_t dim, float eps);

    /// Record SwiGLU (gate = silu(gate) * up, in-place on gate)
    GraphNode swiglu(GraphNode gate, GraphNode up, uint32_t n);

    /// Record residual add (out = a + b)
    GraphNode residual_add(GraphNode a, GraphNode b, GraphNode output, uint32_t n);

    /// Execute ALL recorded operations on ONE Metal command buffer
    /// This is THE ONLY Metal commit for the entire graph.
    bool execute();

    /// Number of recorded operations
    size_t op_count() const { return ops_.size(); }

private:
    ComputeDispatch& dispatch_;
    std::vector<GraphOperation> ops_;
};

}  // namespace nexus::compute
