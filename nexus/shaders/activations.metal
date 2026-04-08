/// NEXUS Metal Shaders — Activation functions
///
/// SiLU (Swish) and GELU for transformer FFN layers.
/// Fused SwiGLU: silu(gate) * up in a single dispatch.

#include <metal_stdlib>
using namespace metal;

/// SiLU activation: x * sigmoid(x)
kernel void silu(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant uint&      n      [[buffer(2)]],
    uint                gid    [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float x = input[gid];
    output[gid] = x / (1.0f + exp(-x));
}

/// GELU activation (approximate): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
kernel void gelu(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant uint&      n      [[buffer(2)]],
    uint                gid    [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float x = input[gid];
    float c = 0.7978845608f;  // sqrt(2/pi)
    float inner = c * (x + 0.044715f * x * x * x);
    output[gid] = 0.5f * x * (1.0f + tanh(inner));
}

/// Fused SwiGLU: output = silu(gate) * up
/// Used in LLaMA-style FFN: output = silu(x @ W1) * (x @ W3)
kernel void swiglu_fused(
    device const float* gate   [[buffer(0)]],   // x @ W1
    device const float* up     [[buffer(1)]],   // x @ W3
    device float*       output [[buffer(2)]],
    constant uint&      n      [[buffer(3)]],
    uint                gid    [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = gate[gid];
    float s = g / (1.0f + exp(-g));  // silu(gate)
    output[gid] = s * up[gid];
}

/// Elementwise add (residual connection)
kernel void residual_add(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device float*       output [[buffer(2)]],
    constant uint&      n      [[buffer(3)]],
    uint                gid    [[thread_position_in_grid]])
{
    if (gid >= n) return;
    output[gid] = a[gid] + b[gid];
}

/// Softmax over a row (one threadgroup per row)
kernel void softmax(
    device float*       data    [[buffer(0)]],
    constant uint&      dim     [[buffer(1)]],
    uint                tid     [[thread_index_in_threadgroup]],
    uint                tg_size [[threads_per_threadgroup]],
    uint                gid     [[threadgroup_position_in_grid]])
{
    uint row_offset = gid * dim;
    threadgroup float shared_val[1024];

    // Find max (parallel reduction)
    float local_max = -INFINITY;
    for (uint i = tid; i < dim; i += tg_size) {
        local_max = max(local_max, data[row_offset + i]);
    }
    shared_val[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_val[tid] = max(shared_val[tid], shared_val[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared_val[0];

    // Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float e = exp(data[row_offset + i] - max_val);
        data[row_offset + i] = e;
        local_sum += e;
    }
    shared_val[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_val[tid] += shared_val[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum = shared_val[0];

    // Normalize
    for (uint i = tid; i < dim; i += tg_size) {
        data[row_offset + i] /= sum;
    }
}
