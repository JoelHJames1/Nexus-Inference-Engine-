/// NEXUS Metal Shader — GPU-Resident Attention
///
/// Single-token decode attention where Q, K cache, and V cache all live on GPU.
/// Eliminates the CPU round-trip (flush/resume) that was the bottleneck for
/// achieving single-command-buffer-per-token execution.
///
/// The shader:
///   1. Writes the new K and V for the current token into the cache
///   2. Computes Q @ K^T scores with online softmax (no extra memory)
///   3. Accumulates weighted V to produce the output
///
/// Supports GQA (grouped-query attention): num_heads > num_kv_heads.
/// Each threadgroup handles one query head.
///
/// Buffer layout:
///   buffer(0): Q          [num_heads * head_dim_q]      — current token's query
///   buffer(1): K cache    [max_seq * kv_dim]             — all cached keys (this layer)
///   buffer(2): V cache    [max_seq * kv_dim]             — all cached values (this layer)
///   buffer(3): Output     [num_heads * head_dim_kv]      — attention output
///   buffer(4): params     GPUAttentionParams struct
///   buffer(5): new_K      [kv_dim]                       — current token's key
///   buffer(6): new_V      [kv_dim]                       — current token's value

#include <metal_stdlib>
using namespace metal;

struct GPUAttentionParams {
    uint seq_len;       // Number of KV entries AFTER writing new token
    uint num_heads;     // Number of query heads
    uint num_kv_heads;  // Number of KV heads (GQA)
    uint head_dim_q;    // Dimension per query head
    uint head_dim_kv;   // Dimension per KV head
    float scale;        // 1/sqrt(head_dim_q)
    uint seq_pos;       // Position to write new K/V in cache
    uint kv_stride;     // Stride between sequence positions (= num_kv_heads * head_dim_kv)
};

/// Step 1: Copy new K and V into the cache at seq_pos.
/// One threadgroup, threads iterate over kv_dim elements.
kernel void attention_gpu_cache_update(
    device const float*           new_K    [[buffer(0)]],
    device const float*           new_V    [[buffer(1)]],
    device float*                 K_cache  [[buffer(2)]],
    device float*                 V_cache  [[buffer(3)]],
    constant GPUAttentionParams&  params   [[buffer(4)]],
    uint                          tid      [[thread_index_in_threadgroup]],
    uint                          tg_size  [[threads_per_threadgroup]])
{
    uint kv_dim = params.kv_stride;
    uint offset = params.seq_pos * kv_dim;

    for (uint i = tid; i < kv_dim; i += tg_size) {
        K_cache[offset + i] = new_K[i];
        V_cache[offset + i] = new_V[i];
    }
}

/// Step 2: Compute GQA attention for one query head.
/// Each threadgroup computes attention for one query head:
///   - Dot product Q_head @ K[t] for all t in [0, seq_len)
///   - Online softmax (numerically stable, single pass)
///   - Weighted sum of V[t]
///
/// For GQA, multiple query heads share the same KV head:
///   kv_head_idx = query_head_idx * num_kv_heads / num_heads
kernel void attention_gpu_decode(
    device const float*           Q        [[buffer(0)]],
    device const float*           K_cache  [[buffer(1)]],
    device const float*           V_cache  [[buffer(2)]],
    device float*                 Output   [[buffer(3)]],
    constant GPUAttentionParams&  params   [[buffer(4)]],
    uint                          tid      [[thread_index_in_threadgroup]],
    uint                          tg_size  [[threads_per_threadgroup]],
    uint                          head_id  [[threadgroup_position_in_grid]])
{
    uint seq_len = params.seq_len;
    uint head_dim_q = params.head_dim_q;
    uint head_dim_kv = params.head_dim_kv;
    float scale = params.scale;
    uint kv_stride = params.kv_stride;

    // GQA mapping: which KV head does this query head use?
    uint kv_head = head_id * params.num_kv_heads / params.num_heads;

    // The dot product dimension is min(head_dim_q, head_dim_kv)
    uint dot_dim = min(head_dim_q, head_dim_kv);

    // Q offset for this head
    uint q_offset = head_id * head_dim_q;

    // Load Q into threadgroup memory for fast repeated access
    threadgroup float q_shared[512];  // max head_dim = 512 (Gemma uses 256)
    for (uint i = tid; i < dot_dim; i += tg_size) {
        q_shared[i] = Q[q_offset + i] * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread processes a SUBSET of sequence positions, maintaining
    // its own running online softmax state. At the end, we reduce across threads.
    //
    // Per-thread accumulators:
    float local_max = -INFINITY;
    float local_sum = 0.0f;

    // Per-thread output accumulator (one per head_dim_kv element)
    // We keep this in registers — for head_dim_kv <= 256 this is fine.
    // For larger dims, we'd tile, but Gemma 31B uses 256.
    float local_out[256] = {};  // Zero-initialized

    uint kv_head_offset = kv_head * head_dim_kv;

    for (uint t = tid; t < seq_len; t += tg_size) {
        // K[t] is at K_cache[t * kv_stride + kv_head * head_dim_kv]
        device const float* k_t = K_cache + t * kv_stride + kv_head_offset;

        // Dot product: Q_head @ K[t]
        float dot = 0.0f;
        for (uint d = 0; d < dot_dim; d++) {
            dot += q_shared[d] * k_t[d];
        }

        // Online softmax update
        float prev_max = local_max;
        local_max = max(local_max, dot);
        float correction = exp(prev_max - local_max);
        local_sum = local_sum * correction + exp(dot - local_max);

        // Accumulate weighted V (correcting previous accumulation)
        device const float* v_t = V_cache + t * kv_stride + kv_head_offset;
        float weight = exp(dot - local_max);
        for (uint d = 0; d < head_dim_kv; d++) {
            local_out[d] = local_out[d] * correction + weight * v_t[d];
        }
    }

    // ── Cross-thread reduction ──
    // We need to merge online-softmax results across threads in the threadgroup.
    // Use threadgroup memory for the reduction.
    //
    // For simplicity and correctness, we use a sequential reduction:
    // Each thread writes its (max, sum, out[]) to threadgroup memory,
    // then thread 0 merges all of them.
    //
    // This is fine for decode where seq_len is small (< 4096) and the
    // threadgroup size is moderate (256-1024 threads).

    threadgroup float tg_max[1024];
    threadgroup float tg_sum[1024];
    // We can't have a 2D threadgroup array of [1024][256], so we store
    // partial outputs in device memory via the output buffer (overwriting is fine
    // since only one threadgroup writes to each head's output region).
    // Instead, use a simpler approach: iterative pairwise reduction.

    tg_max[tid] = local_max;
    tg_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pairwise tree reduction for online softmax merge
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float max_a = tg_max[tid];
            float sum_a = tg_sum[tid];
            float max_b = tg_max[tid + stride];
            float sum_b = tg_sum[tid + stride];

            float new_max = max(max_a, max_b);
            float corr_a = exp(max_a - new_max);
            float corr_b = exp(max_b - new_max);

            tg_max[tid] = new_max;
            tg_sum[tid] = sum_a * corr_a + sum_b * corr_b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Now tg_max[0] and tg_sum[0] have the global max and sum.
    // Each thread corrects its local_out by its local_max vs global_max.
    float global_max = tg_max[0];
    float global_sum = tg_sum[0];
    float inv_sum = (global_sum > 0.0f) ? 1.0f / global_sum : 0.0f;
    float my_correction = exp(local_max - global_max);

    // Write corrected and normalized output.
    // Each thread writes the elements it "owns" — but since each thread
    // accumulated ALL head_dim_kv elements (just for its subset of seq positions),
    // we need to SUM across threads. Use atomics or a different strategy.
    //
    // Better approach: thread 0 does the final serial merge.
    // For decode performance this is acceptable since head_dim is small (256)
    // and we have at most a few hundred threads.

    // Store per-thread partial outputs in threadgroup memory, one thread at a time.
    // We'll use a staging area in threadgroup memory.
    threadgroup float merged_out[512];  // max head_dim_kv

    // Initialize merged output
    if (tid == 0) {
        for (uint d = 0; d < head_dim_kv; d++) {
            merged_out[d] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread atomically adds its contribution. Since Metal doesn't have
    // atomic float add in threadgroup memory on all devices, we serialize.
    // With small threadgroup sizes for decode, this is fast enough.
    for (uint t = 0; t < tg_size; t++) {
        if (tid == t) {
            float my_corr = exp(local_max - global_max);
            for (uint d = 0; d < head_dim_kv; d++) {
                merged_out[d] += local_out[d] * my_corr;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes the final normalized output
    if (tid == 0) {
        uint out_offset = head_id * head_dim_kv;
        for (uint d = 0; d < head_dim_kv; d++) {
            Output[out_offset + d] = merged_out[d] * inv_sum;
        }
    }
}
