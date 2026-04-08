/// NEXUS Metal Shader — RMSNorm
///
/// RMSNorm(x) = x * weight / sqrt(mean(x^2) + eps)
/// Fused into a single GPU dispatch for minimal memory bandwidth.

#include <metal_stdlib>
using namespace metal;

/// RMSNorm kernel — one threadgroup per row.
/// Expects hidden_dim to fit within one threadgroup (max 1024 threads).
kernel void rmsnorm(
    device const float* input       [[buffer(0)]],
    device const float* weight      [[buffer(1)]],
    device float*       output      [[buffer(2)]],
    constant uint&      dim         [[buffer(3)]],
    constant float&     eps         [[buffer(4)]],
    uint                tid         [[thread_index_in_threadgroup]],
    uint                tg_size     [[threads_per_threadgroup]],
    uint                gid         [[threadgroup_position_in_grid]])
{
    // Each threadgroup processes one row
    uint row_offset = gid * dim;

    // Step 1: Compute sum of squares (parallel reduction)
    threadgroup float shared_sum[1024];
    float local_sum = 0.0f;

    for (uint i = tid; i < dim; i += tg_size) {
        float val = input[row_offset + i];
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 2: Compute 1/sqrt(mean + eps)
    float rms_inv = rsqrt(shared_sum[0] / float(dim) + eps);

    // Step 3: Normalize and apply weight
    for (uint i = tid; i < dim; i += tg_size) {
        output[row_offset + i] = input[row_offset + i] * rms_inv * weight[i];
    }
}
