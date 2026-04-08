/// NEXUS Graph Executor — Record operations, execute all in ONE Metal commit.

#include "compute/graph_executor.h"
#include "compute/compute_dispatch.h"
#include <cstdio>

namespace nexus::compute {

GraphExecutor::GraphExecutor(ComputeDispatch& dispatch)
    : dispatch_(dispatch) {}

void GraphExecutor::begin() {
    ops_.clear();
}

GraphNode GraphExecutor::gemv_int4(GraphNode input, MetalBackend::buffer_id weight,
                                     GraphNode output, uint32_t N, uint32_t K) {
    GraphOperation op;
    op.op = GraphOp::GEMV_INT4;
    op.inputs[0] = input;
    op.inputs[1] = weight;
    op.output = output;
    op.params[0] = N;
    op.params[1] = K;
    ops_.push_back(op);
    return output;
}

GraphNode GraphExecutor::rmsnorm(GraphNode input, MetalBackend::buffer_id weight,
                                   GraphNode output, uint32_t dim, float eps) {
    GraphOperation op;
    op.op = GraphOp::RMSNORM;
    op.inputs[0] = input;
    op.inputs[1] = weight;
    op.output = output;
    op.params[0] = dim;
    op.fparams[0] = eps;
    ops_.push_back(op);
    return output;
}

GraphNode GraphExecutor::swiglu(GraphNode gate, GraphNode up, uint32_t n) {
    GraphOperation op;
    op.op = GraphOp::SWIGLU;
    op.inputs[0] = gate;
    op.inputs[1] = up;
    op.output = gate;  // In-place on gate
    op.params[0] = n;
    ops_.push_back(op);
    return gate;
}

GraphNode GraphExecutor::residual_add(GraphNode a, GraphNode b,
                                        GraphNode output, uint32_t n) {
    GraphOperation op;
    op.op = GraphOp::RESIDUAL_ADD;
    op.inputs[0] = a;
    op.inputs[1] = b;
    op.output = output;
    op.params[0] = n;
    ops_.push_back(op);
    return output;
}

bool GraphExecutor::execute() {
    if (ops_.empty()) return true;
    if (!dispatch_.has_gpu()) return false;

    // Start ONE persistent Metal command buffer
    dispatch_.begin_gpu_batch();

    // Encode ALL operations sequentially with memory barriers
    for (const auto& op : ops_) {
        switch (op.op) {
            case GraphOp::GEMV_INT4:
                dispatch_.gemm_int4_gpu(op.inputs[0], op.inputs[1],
                                         op.output, 1, op.params[0], op.params[1]);
                break;

            case GraphOp::RMSNORM:
                dispatch_.gpu_rmsnorm(op.inputs[0], op.inputs[1],
                                       op.output, op.params[0], op.fparams[0]);
                break;

            case GraphOp::SWIGLU:
                dispatch_.gpu_swiglu(op.inputs[0], op.inputs[1], op.params[0]);
                break;

            case GraphOp::RESIDUAL_ADD:
                dispatch_.gpu_residual_add(op.inputs[0], op.inputs[1],
                                            op.output, op.params[0]);
                break;

            default:
                break;
        }
    }

    // THE ONLY COMMIT for ALL operations
    dispatch_.end_gpu_batch();

    return true;
}

}  // namespace nexus::compute
