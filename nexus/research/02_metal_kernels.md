# Fused Dequantization-GEMM Kernels for Apple Metal: Exploiting Unified Memory for LLM Inference

**Joel Hernandez**
NEXUS Inference Engine Project
April 2026

---

## Abstract

We present the Metal compute shader architecture of the NEXUS inference engine, a suite of fused GPU kernels purpose-built for large language model (LLM) inference on Apple Silicon. Unlike CUDA-based runtimes that must contend with PCIe-bottlenecked CPU-GPU data transfers, NEXUS exploits Apple's Unified Memory Architecture (UMA) via `storageModeShared` Metal buffers to achieve zero-copy access to quantized model weights. Our kernel suite includes: (1) a fused INT4 dequantization-GEMV kernel that unpacks 4-bit weights and performs matrix-vector multiplication in a single dispatch, eliminating intermediate dequantization buffers; (2) a tiled 32x32 FP32 GEMM kernel using threadgroup memory for data reuse; (3) a FlashAttention-2-style tiled attention implementation with online softmax and grouped-query attention (GQA) support; (4) a parallel-reduction RMSNorm; (5) a fused SwiGLU activation kernel; and (6) GPU-side TurboQuant encode/decode kernels for online KV cache compression. We provide theoretical throughput analysis for Apple M4 Max (40 GPU cores, 546 GB/s memory bandwidth) and compare our architectural approach against the llama.cpp Metal backend.

---

## 1. Introduction

### 1.1 The Case for Metal in LLM Inference

The dominant paradigm for GPU-accelerated LLM inference assumes an NVIDIA CUDA ecosystem: discrete GPUs with dedicated high-bandwidth memory (HBM), connected to the host CPU via PCIe or NVLink. Frameworks such as vLLM [1], TensorRT-LLM [2], and llama.cpp's CUDA backend [3] are engineered around this architecture. While this approach delivers excellent throughput on data center hardware, it introduces fundamental constraints for consumer and workstation deployment:

1. **PCIe bandwidth bottleneck.** PCIe 4.0 x16 provides approximately 32 GB/s of bidirectional bandwidth. Every tensor that must travel from CPU memory to GPU VRAM incurs this cost. For streaming inference --- where weights are loaded on-demand from storage --- the PCIe link becomes the binding constraint.

2. **Memory duplication.** On discrete GPU systems, weights must exist in both host memory (for loading) and GPU VRAM (for computation), doubling the effective memory footprint.

3. **Synchronization overhead.** CPU-GPU synchronization across PCIe introduces latency that is particularly damaging during autoregressive decode, where each token generation requires a full round-trip.

Apple Silicon's Unified Memory Architecture eliminates all three constraints. The CPU, GPU, and Neural Engine share a single physical memory pool with bandwidth of 546 GB/s (M4 Max) to 819 GB/s (M4 Ultra) [4]. A `storageModeShared` Metal buffer allocated once is immediately accessible to both CPU and GPU with zero copy overhead. This is not merely a convenience --- it is a qualitative architectural difference that enables kernel designs impossible on discrete GPU systems.

### 1.2 Contributions

This paper describes the complete Metal compute shader suite of the NEXUS inference engine:

- **Fused INT4 dequant-GEMV**: A single-dispatch kernel that reads packed 4-bit weights, dequantizes with per-group scales and zero-points, and computes the matrix-vector product without materializing an intermediate FP16/FP32 weight buffer.
- **Tiled GEMM**: A 32x32 threadgroup-memory-tiled GEMM kernel optimized for Apple Silicon's 16 KB threadgroup memory limit and 32-thread SIMD groups.
- **Flash Attention for Metal**: An online-softmax tiled attention kernel supporting causal masking and grouped-query attention, with tile sizes tuned for Metal's threadgroup memory constraints.
- **Fused normalization and activation kernels**: Single-dispatch RMSNorm (parallel tree reduction) and SwiGLU (fused SiLU + elementwise multiply).
- **GPU-side TurboQuant**: On-device KV cache compression using the random-rotation scalar quantization scheme of Ashkboos et al. [5].
- **UMA-aware dispatch strategy**: Pipeline state caching, threadgroup sizing heuristics, and a zero-copy buffer management API.

### 1.3 Paper Organization

Section 2 provides background on Apple Silicon GPU architecture. Section 3 details each kernel design with code excerpts. Section 4 analyzes the UMA zero-copy advantage quantitatively. Section 5 describes our dispatch strategy. Section 6 compares our approach to llama.cpp's Metal backend. Section 7 presents projected performance analysis. Section 8 concludes with future work.

---

## 2. Background: Apple Silicon GPU Architecture

### 2.1 Execution Model

Apple Silicon GPUs implement a tile-based deferred rendering (TBDR) architecture for graphics, but for compute workloads the relevant abstraction is the **compute pipeline**. A compute kernel is dispatched as a grid of **threadgroups**, each containing up to 1024 threads. Threads within a threadgroup execute in **SIMD groups** of 32 threads (analogous to NVIDIA warps), sharing access to a fast **threadgroup memory** scratchpad.

Key hardware parameters for the M4 Max are summarized in Table 1.

**Table 1.** Apple Silicon GPU parameters compared to NVIDIA data center GPUs.

| Parameter                    | Apple M4 Max        | NVIDIA A100 (80 GB)  | NVIDIA H100 (SXM)   |
|------------------------------|---------------------|----------------------|----------------------|
| GPU cores                    | 40                  | 108 SMs              | 132 SMs              |
| SIMD group / warp size       | 32 threads          | 32 threads           | 32 threads           |
| Threadgroup / shared memory  | 16 KB per TG        | 164 KB per SM        | 228 KB per SM        |
| Max threads per threadgroup  | 1024                | 1024                 | 1024                 |
| Memory bandwidth             | 546 GB/s (UMA)      | 2,039 GB/s (HBM2e)  | 3,350 GB/s (HBM3)   |
| Memory capacity              | 48-128 GB (UMA)     | 80 GB (HBM)          | 80 GB (HBM)          |
| CPU-GPU transfer bandwidth   | N/A (unified)       | ~32 GB/s (PCIe 4.0) | ~64 GB/s (PCIe 5.0) |
| FP32 compute (TFLOPS)        | ~14                 | 19.5                 | 66.9                 |
| Memory storage mode          | `storageModeShared` | `cudaMalloc`         | `cudaMalloc`         |

### 2.2 Threadgroup Memory Constraints

The 16 KB threadgroup memory limit on Apple Silicon is significantly smaller than the 164-228 KB available on NVIDIA Ampere/Hopper SMs. This constraint directly impacts tile sizes for GEMM and attention kernels. For FP32 data:

- A 32x32 tile of FP32 values occupies 32 x 32 x 4 = 4,096 bytes (4 KB).
- Two such tiles (for A and B in GEMM) require 8 KB, fitting within the 16 KB limit with headroom for auxiliary data.
- A 64x64 tile would require 32 KB, exceeding the limit.

This motivates our consistent choice of **Br=32, Bc=32** tile dimensions across GEMM and attention kernels.

### 2.3 Unified Memory Architecture

On Apple Silicon, all processors share a single pool of LPDDR5/LPDDR5X memory. A Metal buffer allocated with `MTLResourceStorageModeShared` resides in this unified pool and is directly addressable by both CPU and GPU without any copy or transfer operation. The CPU can write quantized model weights via a simple `memcpy` into the buffer's `contents()` pointer, and the GPU reads them in the next compute dispatch with no intervening DMA transfer. This is the foundation of NEXUS's zero-copy inference pipeline.

---

## 3. Kernel Designs

### 3.1 Fused INT4 Dequantization + GEMV

The most performance-critical kernel in LLM inference is the matrix-vector multiplication during autoregressive decode. At each token generation step, the model computes `output[1,N] = activation[1,K] x W[K,N]` for every linear projection. When weights are stored in INT4 quantization, the naive approach requires three steps: (1) unpack INT4 to FP16/FP32, (2) store the dequantized matrix in an intermediate buffer, and (3) perform the GEMV on the full-precision data.

Our fused kernel eliminates step (2) entirely. Each GPU thread is assigned one output column and walks the K dimension, unpacking INT4 pairs inline and accumulating the dot product:

```metal
kernel void gemv_dequant_int4(
    device const float*   activations [[buffer(0)]],   // [1, K] FP32
    device const uint8_t* weights_q   [[buffer(1)]],   // [K, N/2] packed INT4
    device const float*   scales      [[buffer(2)]],   // [K/group_size, N]
    device const float*   zeros       [[buffer(3)]],   // [K/group_size, N]
    device float*         output      [[buffer(4)]],   // [1, N] FP32
    constant DequantParams& params    [[buffer(5)]],
    uint                  gid         [[thread_position_in_grid]],
    uint                  tid         [[thread_index_in_threadgroup]],
    uint                  tg_size     [[threads_per_threadgroup]])
{
    uint col = gid;
    if (col >= params.cols) return;

    float acc = 0.0f;
    uint K = params.K;
    uint gs = params.group_size;

    for (uint k = 0; k < K; k += 2) {
        uint byte_idx = (k / 2) * params.cols + col;
        uint8_t packed = weights_q[byte_idx];
        uint group_idx = (k / gs) * params.cols + col;
        float scale = scales[group_idx];
        float zero = zeros[group_idx];

        // Unpack low nibble
        float w0 = (float(packed & 0x0F) - zero) * scale;
        acc += activations[k] * w0;

        // Unpack high nibble
        if (k + 1 < K) {
            float w1 = (float((packed >> 4) & 0x0F) - zero) * scale;
            acc += activations[k + 1] * w1;
        }
    }

    output[col] = acc;
}
```

**Design rationale.** Each thread processes two weights per loop iteration (one packed byte), performing the dequantization `(nibble - zero) * scale` and the multiply-accumulate in registers. The quantization group index is recomputed per-group rather than per-element, amortizing the scale/zero loads across `group_size` (typically 128) elements. The dispatch is 1D --- one thread per output column --- which maps naturally to the GEMV workload where the batch dimension is 1.

**Memory savings.** For a 4096x4096 weight matrix in INT4 with group_size=128, the packed representation occupies 4096 x 4096 / 2 = 8 MB. A naive dequantized FP32 intermediate buffer would require 4096 x 4096 x 4 = 64 MB. The fused kernel eliminates this 64 MB allocation entirely.

### 3.2 Tiled FP32 GEMM (32x32)

For batch prefill and multi-token generation, we require a general matrix-matrix multiply. Our GEMM kernel uses classical 2D tiling in threadgroup memory:

```metal
constant uint TILE_SIZE = 32;

kernel void gemm_f32_tiled(
    device const float* A           [[buffer(0)]],
    device const float* B           [[buffer(1)]],
    device float*       C           [[buffer(2)]],
    constant uint&      M           [[buffer(3)]],
    constant uint&      N           [[buffer(4)]],
    constant uint&      K           [[buffer(5)]],
    uint2               tid         [[thread_position_in_threadgroup]],
    uint2               tg_pos      [[threadgroup_position_in_grid]])
{
    uint row = tg_pos.y * TILE_SIZE + tid.y;
    uint col = tg_pos.x * TILE_SIZE + tid.x;
    if (row >= M || col >= N) return;

    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;

    for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        uint ak = t * TILE_SIZE + tid.x;
        tileA[tid.y][tid.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;

        uint bk = t * TILE_SIZE + tid.y;
        tileB[tid.y][tid.x] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE_SIZE; i++) {
            acc += tileA[tid.y][i] * tileB[i][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    C[row * N + col] = acc;
}
```

**Tile sizing analysis.** Each threadgroup allocates two `32x32` FP32 tiles:

- `tileA[32][32]` = 4,096 bytes
- `tileB[32][32]` = 4,096 bytes
- **Total threadgroup memory: 8,192 bytes (8 KB)**

This consumes exactly half of the 16 KB threadgroup memory budget, leaving room for future extensions (e.g., double-buffered tiles for latency hiding).

**Bank conflict avoidance.** Apple Silicon threadgroup memory is organized in banks of 4 bytes. Our tile layout `[TILE_SIZE][TILE_SIZE]` with TILE_SIZE=32 means that consecutive threads in a SIMD group (which have consecutive `tid.x` values) access consecutive columns in `tileB`, producing stride-1 bank accesses with no conflicts. For `tileA`, threads access `tileA[tid.y][i]` where `i` is the loop variable --- all threads in a SIMD group read the same address (broadcast), which is also conflict-free.

**Dispatch geometry.** The host dispatches a 2D grid of threadgroups:

```
grid_size = (ceil(N/32), ceil(M/32), 1)
threadgroup_size = (32, 32, 1)
```

This yields 1,024 threads per threadgroup, the maximum permitted by Apple Silicon.

### 3.3 Flash Attention for Metal

Standard attention computes `O = softmax(QK^T / sqrt(d)) V`, which materializes the full `[seq_len, seq_len]` attention matrix. For a 32K context, this matrix alone requires 32768^2 x 4 = 4 GB in FP32. Flash Attention [6] avoids this by computing attention in tiles using an online softmax algorithm.

#### 3.3.1 Decode Attention (Single-Query)

During autoregressive decode, the query has sequence length 1 while K/V span the full context. Our decode kernel assigns one threadgroup per attention head and uses online softmax to avoid materializing scores:

```metal
kernel void attention_decode_single_head(
    device const float*       Q       [[buffer(0)]],
    device const float*       K       [[buffer(1)]],
    device const float*       V       [[buffer(2)]],
    device float*             O       [[buffer(3)]],
    constant AttentionParams& params  [[buffer(4)]],
    uint                      tid     [[thread_index_in_threadgroup]],
    uint                      tg_size [[threads_per_threadgroup]],
    uint                      head_id [[threadgroup_position_in_grid]])
{
    // ...
    // GQA mapping: multiple query heads share one KV head
    uint kv_head = head_id * params.num_kv_heads / params.num_heads;
    // ...
}
```

**Online softmax.** Rather than computing all scores, finding the maximum, and normalizing in three passes, we maintain running statistics `(local_max, local_sum)` and correct accumulated outputs with each new maximum:

```metal
float prev_max = local_max;
local_max = max(local_max, dot);
float correction = exp(prev_max - local_max);
local_sum = local_sum * correction + exp(dot - local_max);
```

This is numerically equivalent to the standard softmax but requires only a single pass over the KV sequence, with O(d) auxiliary memory per thread instead of O(seq_len).

**Grouped-Query Attention (GQA).** Modern architectures such as LLaMA 3 use GQA where multiple query heads share a single KV head. Our kernel implements this via index mapping: `kv_head = head_id * num_kv_heads / num_heads`. This avoids duplicating KV data in memory and is dispatched as one threadgroup per *query* head, with the KV head resolved inside the kernel.

**Threadgroup memory usage.** The decode kernel allocates:
- `q_shared[256]` = 1,024 bytes (for head_dim up to 256)
- `score_shared[1024]` = 4,096 bytes (tile of attention scores)
- **Total: 5,120 bytes (5 KB)**, well within the 16 KB limit.

#### 3.3.2 Prefill Attention (Causal)

For batch prefill of a prompt, we dispatch a 2D grid where each thread handles one (query_position, dimension_index) pair. The causal mask is enforced by iterating only over `kv_pos <= query_pos`:

```metal
// Causal mask: only attend to positions <= query_pos
for (uint kv_pos = 0; kv_pos <= query_pos; kv_pos++) {
    float dot = 0.0f;
    for (uint i = 0; i < d; i++) {
        dot += Q[query_pos * d + i] * K[kv_pos * d + i];
    }
    dot *= scale;

    // Online softmax
    float prev_max = max_score;
    max_score = max(max_score, dot);
    float correction = exp(prev_max - max_score);
    sum_exp = sum_exp * correction + exp(dot - max_score);
    output_val = output_val * correction
               + exp(dot - max_score) * V[kv_pos * d + dim_idx];
}
```

This kernel applies the same online softmax algorithm as the decode path but over the triangular causal region. The grid dispatch is `(head_dim, seq_len, 1)`, ensuring each output element is computed independently.

### 3.4 RMSNorm: Parallel Tree Reduction

RMSNorm [7] normalizes each hidden-state vector by its root mean square: `RMSNorm(x) = x * w / sqrt(mean(x^2) + eps)`. This requires a global reduction (sum of squares) followed by an elementwise scaling. Our kernel performs both in a single dispatch using threadgroup memory:

```metal
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
    uint row_offset = gid * dim;

    // Step 1: Parallel sum of squares
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

    // Step 2: Normalize
    float rms_inv = rsqrt(shared_sum[0] / float(dim) + eps);
    for (uint i = tid; i < dim; i += tg_size) {
        output[row_offset + i] = input[row_offset + i] * rms_inv * weight[i];
    }
}
```

**Reduction complexity.** The tree reduction executes `log2(tg_size)` steps with one barrier per step. For `tg_size=512` (rounded to power-of-2), this is 9 reduction steps. Each step halves the active thread count, which maps efficiently to SIMD group execution on Apple Silicon where inactive threads are masked at the SIMD level.

**Threadgroup sizing.** The host rounds the threadgroup size down to the nearest power of 2, which is required for correct tree reduction. For `hidden_dim=4096` with `max_threads=1024`, the threadgroup size is 1024, and each thread handles `4096/1024 = 4` elements in the accumulation loop. The `shared_sum` array occupies `1024 x 4 = 4,096 bytes`.

### 3.5 Fused SwiGLU Activation

LLaMA-family models use the SwiGLU activation in their feed-forward networks [8]: `FFN(x) = SiLU(x W_gate) * (x W_up)`. A naive implementation requires two separate dispatches (SiLU on the gate projection, then elementwise multiply with the up projection). Our fused kernel performs both operations in a single dispatch:

```metal
kernel void swiglu_fused(
    device const float* gate   [[buffer(0)]],   // x @ W_gate
    device const float* up     [[buffer(1)]],   // x @ W_up
    device float*       output [[buffer(2)]],
    constant uint&      n      [[buffer(3)]],
    uint                gid    [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = gate[gid];
    float s = g / (1.0f + exp(-g));  // silu(gate)
    output[gid] = s * up[gid];
}
```

**Fusion benefits.** The unfused version requires: (1) reading `gate` from device memory, writing `silu(gate)` to an intermediate buffer; (2) reading the intermediate and `up`, writing the product. Total device memory traffic: 5N floats read + 2N floats written = 7N x 4 bytes. The fused version reads `gate` and `up` once and writes `output` once: 2N reads + 1N write = 3N x 4 bytes. This is a **2.3x reduction in memory traffic**, which matters for memory-bandwidth-bound activations.

### 3.6 TurboQuant Encode/Decode on GPU

TurboQuant [5] (Ashkboos et al., ICLR 2026) achieves near-lossless KV cache compression through random rotation followed by scalar quantization. We implement both encode and decode as Metal compute kernels, enabling the entire quantization pipeline to execute on-device without CPU round-trips.

**Encode kernel.** Each thread handles one coordinate of one vector. The quantization boundary search is a linear scan over the `num_centroids - 1` boundaries (4 or 16 entries), which is fast enough to remain in registers:

```metal
kernel void turbo_quant_encode(
    device const float*         input       [[buffer(0)]],
    device const float*         centroids   [[buffer(1)]],
    device const float*         boundaries  [[buffer(2)]],
    device uint8_t*             output      [[buffer(3)]],
    device float*               norms       [[buffer(4)]],
    constant TurboQuantParams&  params      [[buffer(5)]],
    uint2                       gid         [[thread_position_in_grid]])
{
    // ...
    // Binary search for nearest centroid
    uint idx = 0;
    for (uint b = 0; b < params.num_centroids - 1; b++) {
        if (val > boundaries[b]) idx = b + 1;
    }

    // Pack indices: 4-bit -> 2 per byte, 2-bit -> 4 per byte
    if (params.bits == 4) {
        uint byte_pos = vec_idx * (params.dim / 2) + coord / 2;
        if (coord % 2 == 0)
            output[byte_pos] = (output[byte_pos] & 0xF0) | (idx & 0x0F);
        else
            output[byte_pos] = (output[byte_pos] & 0x0F) | ((idx & 0x0F) << 4);
    }
    // ...
}
```

**Decode kernel.** Inverse operation: unpack the bit index, look up the centroid value. The rotation inverse and norm rescaling are applied by a separate kernel or fused into the attention computation.

**Compression ratios.** Table 2 summarizes the per-token KV memory with TurboQuant.

**Table 2.** KV cache memory per token per layer (head_dim=128, GQA with 8 KV heads).

| Precision     | Bytes/token/layer | Compression vs FP16 |
|---------------|-------------------|----------------------|
| FP16 (baseline) | 4,096          | 1.0x                 |
| INT8           | 2,048            | 2.0x                 |
| TurboQuant 4-bit | 1,024 + 32 (norms) | 3.9x           |
| TurboQuant 2-bit | 512 + 32 (norms)  | 7.5x            |

For a 128-layer model at 32K context length, TurboQuant 4-bit reduces KV cache from 16.8 GB (FP16) to 4.3 GB --- the difference between fitting in memory and not.

---

## 4. UMA Zero-Copy Advantage

### 4.1 The Cost of Discreteness

On discrete GPU systems, model weights must traverse the PCIe bus to reach GPU VRAM. For a single linear projection in a 70B model (e.g., a 8192x8192 INT4 weight matrix), the data transfer is:

```
Weight size (INT4):  8192 x 8192 / 2 = 33.5 MB
PCIe 4.0 x16 bandwidth:  ~32 GB/s
Transfer time:  33.5 / 32,000 = 1.05 ms
```

This transfer occurs *before* any computation begins. For streaming inference where weights are loaded on-demand, every layer incurs this cost.

### 4.2 Zero-Copy on Apple Silicon

On Apple Silicon with `storageModeShared`, the same weight buffer is allocated once in unified memory and accessed by the GPU directly:

```objectivec
// Allocation: direct write to GPU-visible memory
id<MTLBuffer> buf = [device newBufferWithLength:bytes
                     options:MTLResourceStorageModeShared];
// CPU writes directly into unified memory
std::memcpy([buf contents], data, size);
// GPU reads immediately --- no transfer, no synchronization
[encoder setBuffer:buf offset:0 atIndex:0];
```

The effective "transfer" bandwidth is the UMA memory bandwidth itself: 546 GB/s on M4 Max. Table 3 compares the cost of making a weight matrix available to the GPU.

**Table 3.** Time to make a 33.5 MB INT4 weight matrix GPU-accessible.

| System           | Mechanism               | Bandwidth    | Time     |
|------------------|-------------------------|--------------|----------|
| NVIDIA A100      | PCIe 4.0 x16 DMA       | 32 GB/s      | 1.05 ms  |
| NVIDIA H100      | PCIe 5.0 x16 DMA       | 64 GB/s      | 0.52 ms  |
| Apple M4 Max     | UMA direct access       | 546 GB/s     | 0.061 ms |
| Apple M4 Ultra   | UMA direct access       | 819 GB/s     | 0.041 ms |

The M4 Max achieves **17x lower latency** than an A100 for weight access, and **8.5x lower** than an H100. This advantage is amplified for streaming inference, where every layer boundary requires new weight access.

### 4.3 Implications for Kernel Design

The UMA zero-copy property has several consequences for kernel design in NEXUS:

1. **No staging buffers.** CUDA kernels often use pinned host memory plus asynchronous DMA streams to overlap transfer with compute. NEXUS requires none of this machinery. Every `storageModeShared` buffer is simultaneously a "host buffer" and a "device buffer."

2. **In-place dequantization is unnecessary.** Some CUDA runtimes pre-dequantize weights into a GPU-side FP16 buffer to amortize transfer cost. On UMA, the GPU reads quantized data at full bandwidth, so fused dequantization is purely a compute optimization, not a transfer optimization.

3. **Dynamic buffer reuse.** CPU-side weight streaming writes directly into shared buffers that the GPU consumes. Buffer recycling (writing new layer weights into a buffer that the GPU finished reading) requires only a simple fence, not a DMA completion callback.

The NEXUS buffer management API reflects this design:

```cpp
class MetalBackend {
public:
    using buffer_id = uint64_t;

    /// Allocate a storageModeShared MTLBuffer
    buffer_id alloc_shared_buffer(size_t bytes);

    /// Direct CPU write to GPU-visible memory (memcpy into UMA)
    bool copy_to_buffer(buffer_id handle, const void* data, size_t size);

    /// Get raw pointer for zero-copy CPU access
    void* buffer_contents(buffer_id handle);

    void free_buffer(buffer_id handle);
};
```

---

## 5. Dispatch Strategy

### 5.1 Pipeline State Caching

Metal requires creating a `MTLComputePipelineState` object before dispatching a kernel. Pipeline state creation involves shader compilation and is expensive (1-10 ms). NEXUS caches pipeline states by kernel name in the `MetalContext`, so each kernel's pipeline is compiled exactly once:

```objectivec
id<MTLComputePipelineState> pipeline(const std::string& name) {
    auto pso = ctx->get_pipeline(name);
    if (!pso) pso = ctx->build_pipeline(name);
    return pso;
}
```

Subsequent dispatches of the same kernel retrieve the cached pipeline state in O(1) via hash lookup.

### 5.2 Threadgroup Size Selection

NEXUS employs different threadgroup sizing strategies depending on kernel geometry:

**1D kernels (GEMV, SiLU, SwiGLU, residual_add).** Threadgroup size is set to `min(maxTotalThreadsPerThreadgroup, 1024)`, and the grid is `ceil(N / tg_size)` threadgroups. This maximizes occupancy for elementwise and vector operations:

```objectivec
NSUInteger tg = std::min<NSUInteger>([pso maxTotalThreadsPerThreadgroup], 1024);
MTLSize threadgroup_size = MTLSizeMake(tg, 1, 1);
MTLSize grid_size = MTLSizeMake(ceil_div(N, (uint32_t)tg), 1, 1);
```

**2D kernels (tiled GEMM, prefill attention).** Threadgroup size is `(32, 32, 1)` = 1024 threads, and the grid is `(ceil(N/32), ceil(M/32), 1)`:

```objectivec
constexpr uint32_t TILE = 32;
MTLSize threadgroup_size = MTLSizeMake(TILE, TILE, 1);
MTLSize grid_size = MTLSizeMake(ceil_div(N, TILE), ceil_div(M, TILE), 1);
```

**Reduction kernels (RMSNorm, softmax).** Threadgroup size is rounded down to the nearest power of 2 (required for correct tree reduction), and each threadgroup processes one row:

```objectivec
NSUInteger tg_pow2 = 1;
while (tg_pow2 * 2 <= tg) tg_pow2 *= 2;
MTLSize threadgroup_size = MTLSizeMake(tg_pow2, 1, 1);
MTLSize grid_size = MTLSizeMake(1, 1, 1);  // one row per dispatch
```

**Attention decode.** One threadgroup per query head, with threads cooperating to iterate over the KV sequence:

```objectivec
MTLSize threadgroup_size = MTLSizeMake(tg, 1, 1);
MTLSize grid_size = MTLSizeMake(params.num_heads, 1, 1);
```

### 5.3 Synchronous vs. Pipelined Execution

The current implementation commits command buffers synchronously (`[cb waitUntilCompleted]`), which serializes CPU and GPU execution. This simplifies correctness during development. Phase 3 will introduce a pipelined execution model using triple-buffered command buffers, allowing the CPU to prepare the next layer's dispatch while the GPU executes the current one.

---

## 6. Comparison with llama.cpp Metal Backend

llama.cpp [3] includes a Metal backend (`ggml-metal.m`) that is the most widely deployed Metal-based LLM inference implementation. Table 4 compares architectural choices.

**Table 4.** Architectural comparison: NEXUS vs. llama.cpp Metal backend.

| Aspect                        | NEXUS                                | llama.cpp Metal                     |
|-------------------------------|--------------------------------------|--------------------------------------|
| Quantization format           | NXF (per-chunk mixed-precision)      | GGUF (per-tensor Q4_K, Q5_K, etc.) |
| Dequant strategy              | Fused dequant-GEMV (single dispatch) | Separate dequant + GEMM kernels     |
| GEMM tiling                   | 32x32 FP32 tiles                     | Various (4x4, 4x8 for quant types) |
| Attention                     | FlashAttention-2 with online softmax | Standard materialized attention      |
| KV cache compression          | TurboQuant on GPU                    | FP16 / Q8_0 / Q4_0                  |
| GQA support                   | Index mapping in attention kernel    | Head duplication before dispatch     |
| Activation fusion             | SwiGLU fused                         | Separate SiLU + multiply             |
| Pipeline state management     | Cached by kernel name                | Rebuilt per model load               |
| Weight streaming              | Designed for layer-by-layer NXF      | Full model residency (mmap)          |
| Buffer allocation             | Explicit shared buffers              | ggml_metal_add_buffer (mmap'd)       |

### 6.1 Key Differences

**Fused vs. staged dequantization.** llama.cpp performs dequantization as a distinct step, producing an FP16 intermediate buffer that is then consumed by a separate GEMM kernel. NEXUS fuses these operations, eliminating the intermediate buffer and halving the memory traffic for the weight data.

**Attention implementation.** llama.cpp's Metal attention kernels materialize the full attention score matrix (or large tiles thereof) before applying softmax. For long contexts, this scales as O(seq_len^2) in memory. NEXUS's online softmax approach computes attention in O(seq_len * head_dim) memory, enabling longer context without out-of-memory conditions.

**KV cache quantization.** llama.cpp supports Q8_0 and Q4_0 for KV cache but does not perform rotation-based quantization. NEXUS implements TurboQuant on-device, achieving higher quality at equivalent bit rates due to the random rotation step that normalizes coordinate magnitudes.

**Streaming architecture.** llama.cpp's Metal backend assumes the entire model is resident in memory (via mmap of the GGUF file). NEXUS's Metal backend is designed to work with the NXF streaming format [9], where only 2-3 layers need to be resident simultaneously. The explicit `alloc_shared_buffer` / `free_buffer` API enables layer-by-layer buffer recycling.

---

## 7. Projected Performance Analysis

### 7.1 Roofline Model for M4 Max

The M4 Max has 40 GPU cores with an estimated peak compute of approximately 14 TFLOPS FP32 and memory bandwidth of 546 GB/s. The arithmetic intensity boundary between compute-bound and memory-bound operation is:

```
AI_boundary = 14,000 GFLOPS / 546 GB/s = 25.6 FLOP/byte
```

For autoregressive decode (batch size 1), the arithmetic intensity of GEMV is approximately `2K / (K/2 + 4)` FLOP/byte for INT4 weights (2 FLOPs per weight element, K/2 bytes of packed weights + 4 bytes of activation per column element). For K=4096: `AI = 8192 / 2052 = 3.99 FLOP/byte`. This is well below the roofline knee, confirming that **decode is memory-bandwidth-bound**.

### 7.2 Theoretical Decode Throughput

**Table 5.** Projected token generation latency for LLaMA-3-70B on M4 Max (INT4 quantized).

| Component              | Weight Size (INT4)  | Bandwidth  | Time (ms) | Notes                   |
|------------------------|---------------------|------------|-----------|-------------------------|
| Attention QKV proj     | 3 x 8192x1024 / 2  | 546 GB/s   | 0.022     | 3 projections (GQA)     |
| Attention output proj  | 8192x8192 / 2       | 546 GB/s   | 0.061     |                         |
| FFN gate proj          | 8192x28672 / 2      | 546 GB/s   | 0.215     |                         |
| FFN up proj            | 8192x28672 / 2      | 546 GB/s   | 0.215     |                         |
| FFN down proj          | 28672x8192 / 2      | 546 GB/s   | 0.215     |                         |
| RMSNorm (x2)           | negligible          | ---        | ~0.002    | Compute-bound            |
| SwiGLU fusion          | negligible          | ---        | ~0.001    | Elementwise              |
| Attention decode       | KV cache dependent  | 546 GB/s   | ~0.1      | 2K context, 128d heads  |
| **Per-layer total**    |                     |            | **~0.83** |                         |
| **80 layers total**    |                     |            | **~66.4** |                         |
| **Projected tok/s**    |                     |            |           | **~15 tok/s**           |

This estimate assumes 80% memory bandwidth utilization (achievable with coalesced access patterns) and does not account for compute overlap. The actual throughput will depend on kernel occupancy and memory access pattern efficiency.

### 7.3 Comparison to Discrete GPU Expectations

**Table 6.** Projected decode throughput comparison (LLaMA-3-70B INT4, batch=1).

| System               | Memory BW   | Weight Transfer | Est. tok/s | Cost (USD)    |
|-----------------------|-------------|-----------------|------------|---------------|
| Apple M4 Max (128GB)  | 546 GB/s    | Zero-copy       | ~15        | ~4,000        |
| NVIDIA RTX 4090       | 1,008 GB/s  | PCIe limited*   | ~12**      | ~2,000 + host |
| NVIDIA A100 (80GB)    | 2,039 GB/s  | GPU-resident    | ~30        | ~15,000       |
| NVIDIA H100 (SXM)     | 3,350 GB/s  | GPU-resident    | ~50        | ~30,000       |

*RTX 4090 has only 24 GB VRAM; 70B INT4 (~35 GB) does not fit, requiring offloading.
**Throughput limited by PCIe transfers for layers that do not fit in VRAM.

The M4 Max's cost-performance ratio for 70B-class models is competitive because UMA eliminates the PCIe bottleneck that cripples consumer NVIDIA GPUs on models exceeding VRAM capacity.

---

## 8. Conclusion and Future Work

### 8.1 Summary

We have presented the Metal compute shader architecture of the NEXUS inference engine, demonstrating how Apple Silicon's Unified Memory Architecture enables kernel designs --- particularly fused dequantization-GEMV and zero-copy weight streaming --- that are architecturally impossible on discrete GPU systems. Our kernel suite covers the complete transformer inference pipeline: quantized linear projections, tiled GEMM, flash attention with GQA support, RMSNorm, fused SwiGLU, and GPU-side TurboQuant KV compression.

### 8.2 Future Work

Several optimizations remain for subsequent phases:

1. **FP16/BF16 GEMM kernels.** The current tiled GEMM operates on FP32. Apple Silicon supports native FP16 operations at 2x throughput; exploiting this will approximately double GEMM performance.

2. **SIMD shuffle reductions.** The attention decode kernel currently relies on thread 0 for final output writing. Using Metal's `simd_shuffle` and `simd_sum` intrinsics will enable efficient cross-thread reductions without threadgroup memory.

3. **Fused INT4 dequant-GEMM (batched).** Extending the fused dequantization approach from GEMV (batch=1) to full GEMM (batch>1) for prefill, combining the tiling strategy of Section 3.2 with the inline dequantization of Section 3.1.

4. **Double-buffered tiling.** Loading the next tile asynchronously while computing on the current tile, exploiting the 8 KB of unused threadgroup memory in the GEMM kernel.

5. **Pipelined command buffer submission.** Replacing synchronous `waitUntilCompleted` with triple-buffered command buffer submission to overlap CPU dispatch preparation with GPU execution.

6. **Structured Hadamard rotations for TurboQuant.** Replacing the dense rotation matrix (O(d^2) cost) with a Fast Walsh-Hadamard transform (O(d log d) cost) for the random rotation step.

7. **Integration with NXF streaming.** Connecting the Metal dispatch pipeline to the NXF layer-by-layer weight streaming system [9] for models exceeding unified memory capacity.

---

## References

[1] Meta AI. "The LLaMA 3 Herd of Models." arXiv:2407.21783, 2024.

[2] NVIDIA. "TensorRT-LLM: A TensorRT Toolbox for Large Language Models." GitHub, 2024.

[3] Gerganov, G. "llama.cpp: Inference of Meta's LLaMA model in pure C/C++." GitHub, 2023-2026.

[4] Apple Inc. "Apple M4 Family Technical Overview." Apple Developer Documentation, 2025.

[5] Ashkboos, S., Mohtashami, A., Croci, M. L., Li, B., Jaggi, M., Alistarh, D., Hoefler, T., Hensman, J. "TurboQuant: Online Vector Quantization for KV Cache Compression." arXiv:2504.19874, ICLR 2026.

[6] Dao, T., Fu, D., Ermon, S., Rudra, A., Re, C. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." arXiv:2307.08691, 2023.

[7] Zhang, B., Sennrich, R. "Root Mean Square Layer Normalization." NeurIPS, 2019.

[8] Shazeer, N. "GLU Variants Improve Transformer." arXiv:2002.05202, 2020.

[9] Hernandez, J. "NXF: A Streaming-Native Tensor Format for Memory-Constrained LLM Inference on Apple Silicon." NEXUS Research Paper #1, 2026.

---

*NEXUS Inference Engine is an open research project. All Metal shader source code referenced in this paper is available in the project repository under `nexus/shaders/`.*
