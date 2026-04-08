# NEXUS Inference Engine: Current Architecture and Performance Analysis

**Date**: April 2026
**Status**: Active development — functional inference on Apple Silicon
**Codebase**: 73 source files, ~18,900 lines of C++/Objective-C++/Metal

---

## 1. Executive Summary

NEXUS is a from-scratch LLM inference engine purpose-built for Apple Silicon Macs. It exploits UMA (Unified Memory Architecture) to run large language models with minimal RAM overhead by streaming quantized weights directly from SSD through GPU compute with zero-copy memory access.

**What it does today:**

- Runs Gemma 4 31B at **2.7 tok/s** with 168 MB -- 3.8 GB peak RAM
- Runs Qwen3-Coder-Next 80B at **3.0 tok/s** with 8.6 GB peak RAM
- Prefill latency: **0.8--1.5 seconds**
- Supports dense transformer models (LLaMA, Gemma) and hybrid SSM+MoE+Attention architectures (Qwen3-Coder-Next)
- Imports from GGUF (all K-quant types) and safetensors (single-file and multi-shard)
- Custom NXF binary format with 16 KB page-aligned chunking for mmap streaming
- Fused INT4 GEMV on Metal GPU with UMA zero-copy buffer wrapping
- Full KV cache subsystem: flat cache (active), paged attention, TurboQuant compression, prefix caching, H2O/SnapKV eviction (all implemented)

---

## 2. System Architecture

### 2.1 Component Overview

```
+---------------------------+
|        CLI / HTTP API     |   api/  (2,127 lines)
+---------------------------+
            |
+---------------------------+
|        Engine             |   core/  (1,164 lines)
|  (orchestrator, scheduler,|
|   tokenizer)              |
+---------------------------+
       |          |
+------+---+ +---+----------+
| Model    | | NXF Format   |   model/ (3,956 lines)  format/ (1,010 lines)
| Transformer| NXFReader    |
| HybridModel| NXFWriter    |
| MoE Router |              |
+----------+ +--------------+
       |
+---------------------------+
|   ComputeDispatch         |   compute/  (2,952 lines)
|  +-------+  +-----------+ |
|  | Metal |  | Accelerate| |
|  | GEMV  |  | cblas     | |
|  +-------+  +-----------+ |
+---------------------------+
       |
+---------------------------+
|   Memory / KV             |   memory/ (733 lines)  kv/ (2,606 lines)
|  UMA Allocator            |
|  MemoryManager (slab+LRU) |
|  KVStore, PagedKVCache    |
|  TurboQuant, PrefixCache  |
|  H2O / SnapKV Eviction    |
+---------------------------+
       |
+---------------------------+
|   Import / Quantization   |   import/ (3,290 lines)  quant/ (1,046 lines)
|  GGUF Importer            |
|  Safetensors Importer     |
|  GPTQ, AWQ, QuIP#, ANS   |
+---------------------------+
```

### 2.2 Data Flow: Model File to Token Output

1. **Import**: GGUF or safetensors file is converted to NXF format. Weights are quantized to INT4 (or passed through for K-quant types). Output is a single `.nxf` file with 16 KB-aligned chunks.

2. **Load**: `Engine::create()` opens the NXF file via `NXFReader::open()`. Reads the 64-byte header, JSON manifest, and tensor index. Tensor data is NOT loaded -- it will be mmap'd on demand.

3. **Model construction**: Based on the manifest's `architecture` field, either a `Transformer` (dense) or `HybridModel` (SSM+MoE) is created. Layer weights are mapped lazily.

4. **Prefill**: All prompt tokens are processed sequentially through all layers. Each layer's weights are mmap'd, wrapped as MTLBuffers (zero-copy), computed via fused INT4 GEMV on Metal, and KV entries are written to the cache.

5. **Decode loop**: For each output token:
   - Forward pass through all layers (streaming weights layer-by-layer)
   - Each layer: RMSNorm -> Attention/SSM -> Residual -> RMSNorm -> FFN -> Residual
   - Final RMSNorm -> output projection -> logits -> sampling
   - Token callback fires with the decoded text

6. **Streaming**: Only 2--3 layers of weights are resident at any time. The `Scheduler` prefetches the next layer's weights from SSD while the current layer computes on GPU. Previous layers are evicted via `madvise(MADV_DONTNEED)`.

---

## 3. NXF Format

### 3.1 File Layout

```
Offset 0:     NXFHeader (64 bytes, fixed)
              +-- magic: "NXF1" (0x3146584E)
              +-- version: 1
              +-- manifest_offset / manifest_size
              +-- tensor_index_offset / tensor_index_size
              +-- data_offset
              +-- total_file_size

Manifest:     JSON blob with model architecture metadata
              (ModelManifest: architecture, dims, MoE config, codec info)

Tensor Index: Array of TensorInfo entries, each with:
              +-- name (e.g., "layers.0.attention.wq.weight")
              +-- shape (e.g., [4096, 4096])
              +-- dtype (original precision)
              +-- chunks: ordered list of ChunkDesc

Data Section: Chunk data, each chunk at 16 KB alignment
```

### 3.2 Chunk Design

Each tensor is split into one or more chunks described by `ChunkDesc` (24 bytes):

- `file_offset`: absolute position in file
- `compressed_size` / `decompressed_size`: on-disk vs in-memory
- `checksum`: xxHash32 of compressed data
- `codec`: per-chunk codec (allows mixed precision within a tensor)
- `group_size`: quantization group size (e.g., 128 for INT4)

### 3.3 Supported Codecs

| Codec ID | Name | Description |
|----------|------|-------------|
| 0 | FP32 | 32-bit float |
| 1 | FP16 | 16-bit float |
| 2 | BF16 | Brain float 16 |
| 3 | INT8 | 8-bit integer |
| 4 | INT4 | Block quantized, group_size=128 |
| 5 | GPTQ | Per-group 4-bit with scales/zeros |
| 6 | AWQ | Activation-aware weight quantization |
| 7 | QUIP3 | QuIP# 3-bit E8 lattice |
| 8 | AQLM2 | AQLM 2-bit additive codebooks |
| 9 | ANS | Entropy coded (post-quant lossless) |
| 10 | TURBO_Q | TurboQuant (KV cache) |
| 11 | Q3K | Q3_K native passthrough (3.4 bits) |
| 12 | Q4K | Q4_K native passthrough (4.5 bits) |
| 13 | Q2K | Q2_K native passthrough (2.6 bits) |

### 3.4 16 KB Alignment

All chunk data is aligned to 16 KB boundaries -- the arm64 macOS VM page size. This is critical for:

- `mmap()` to work with page-granularity faulting
- `MTLDevice.newBufferWithBytesNoCopy()` which requires page-aligned pointers
- Zero-copy UMA access: the same physical page is visible to both CPU and GPU without any data movement

This is a deliberate departure from x86 conventions (4 KB pages). GGUF uses unaligned data; safetensors uses no alignment. NXF pays the padding cost to enable true zero-copy on Apple Silicon.

---

## 4. Compute Pipeline

### 4.1 Metal GPU Path: Fused INT4 GEMV

The primary compute path is `gemv_int4_uniform()` on the `MetalBackend`. This kernel:

1. Takes a raw mmap'd INT4 weight pointer (2 values packed per byte)
2. Wraps it as an `MTLBuffer` via `wrap_pointer()` -- no allocation, no copy on UMA
3. Dispatches a Metal compute shader that reads INT4 nibbles, dequantizes inline via `(nibble - 8) * 0.125`, and accumulates the dot product
4. Output is written to a storageModeShared buffer readable by CPU

This is the **fastest path** because:
- No CPU dequantization step
- No memory allocation per dispatch
- INT4 means 4x less memory bandwidth than FP32 (the bottleneck on Apple Silicon)
- The GPU reads directly from the mmap'd file pages

A second kernel, `gemv_dequant_int4()`, handles the case with explicit scales/zeros buffers (for GPTQ-style quantization).

### 4.2 ComputeDispatch

`ComputeDispatch` is the unified CPU/GPU dispatch layer. Key features:

**GPU Buffer Pool**: Pre-allocated persistent MTLBuffers for intermediate activations:
- `hidden`, `residual`, `norm_buf`: hidden state pipeline
- `q_buf`, `k_buf`, `v_buf`, `attn_out`: attention intermediates
- `ffn_gate`, `ffn_up`, `ffn_out`: FFN intermediates
- `logits`: vocabulary-sized output buffer
- `kv_keys`, `kv_values`: GPU-resident KV cache (infrastructure built)

Allocated once at model load, reused every token.

**Batch Mode** (`begin_gpu_batch()` / `end_gpu_batch()`): Multiple GPU dispatches share a single Metal command buffer. Reduces commit overhead by batching operations within a layer (e.g., the FFN's 4 GEMVs).

**Token-Batch Mode** (`begin_token()` / `end_token()`): Infrastructure for encoding an ENTIRE token's worth of GPU work (all layers, all dispatches) into a single Metal command buffer. Currently built but effectively disabled because attention requires CPU-side softmax which forces a `flush_token()` / `resume_token()` mid-token, losing much of the benefit.

**Buffer Caching**: `wrapped_buffer_cache_` avoids recreating MTLBuffers for the same mmap'd weight pointers across tokens. Without this, 288 `wrap_pointer` + `free_buffer` calls per token were needed.

**Pre-cached MTLBuffers**: `upload_small_buffer()` caches small tensors (RMSNorm weights, etc.) as persistent GPU buffers.

### 4.3 Per-Layer Execution: Dense Transformer (Gemma)

For a standard dense transformer layer:

```
Input: hidden_state [hidden_dim]

1. RMSNorm(hidden_state, attn_norm_weight) -> norm_out
2. GEMV_INT4(norm_out, wq) -> Q   [num_heads * head_dim]
3. GEMV_INT4(norm_out, wk) -> K   [num_kv_heads * head_dim]
4. GEMV_INT4(norm_out, wv) -> V   [num_kv_heads * head_dim]
5. RoPE(Q, K, seq_pos)
6. KV cache write (K, V at seq_pos)
7. GQA attention: Q x K^T -> scores -> softmax -> scores x V -> attn_out
8. GEMV_INT4(attn_out, wo) -> proj_out
9. Residual: hidden_state += proj_out

10. RMSNorm(hidden_state, ffn_norm_weight) -> norm_out
11. GEMV_INT4(norm_out, w1) -> gate   [ffn_dim]
12. GEMV_INT4(norm_out, w3) -> up     [ffn_dim]
13. SwiGLU: silu(gate) * up -> ffn_act
14. GEMV_INT4(ffn_act, w2) -> ffn_out [hidden_dim]
15. Residual: hidden_state += ffn_out
```

That is **7 INT4 GEMVs per layer**. For a 60-layer model, that is **420 GPU dispatches per token**.

### 4.4 Per-Layer Execution: Hybrid Model (Qwen3-Coder-Next)

Qwen3-Coder-Next uses a repeating pattern of 48 layers: `12 x (3 x SSM+MoE, 1 x Attention+MoE)`.

**Type A layers (SSM+MoE)** -- 36 of 48:
- Pre-attention RMSNorm
- Fused QKV projection via GEMV_INT4 (gated DeltaNet, simplified as linear attention)
- Gating projection via GEMV_INT4
- SSM output projection via GEMV_INT4
- Post-attention RMSNorm
- MoE FFN: router selects top-10 of 512 experts, each expert runs SwiGLU (3 GEMVs), results weighted-summed

**Type B layers (Attention+MoE)** -- 12 of 48:
- Pre-attention RMSNorm
- Separate Q, K, V projections with Q/K RMSNorm
- Full GQA attention with RoPE
- Output projection
- Post-attention RMSNorm
- MoE FFN (same as Type A)

**MoE routing** uses DeepSeek-V3 style auxiliary-loss-free load balancing with per-expert bias terms adjusted at runtime. The router also supports predictive prefetch of next-token experts.

Expert weights are stored as raw INT4 in the NXF file. Per-expert slicing happens on demand from the full `[num_experts, hidden_dim, ffn_dim]` packed tensor. A `batched_moe_ffn()` path executes all active experts in a single GPU dispatch.

---

## 5. Memory Management

### 5.1 UMA Allocator

The `UMAAllocator` provides page-aligned shared CPU/GPU memory via `mmap(MAP_ANON | MAP_PRIVATE)`. On Apple Silicon UMA, this memory is physically the same DRAM seen by both CPU cores and GPU cores -- there is no discrete VRAM or PCIe transfer.

### 5.2 MemoryManager

The `MemoryManager` is the central allocation authority:

- **Page allocation**: `alloc_pages()` / `free_pages()` with automatic LRU eviction when approaching the RAM cap (default 48 GB)
- **Slab pools**: Fixed-size slab allocators for weight chunks (1 MB slabs) and KV pages (64 KB slabs). Eliminates fragmentation and allocation overhead on the hot path.
- **LRU eviction**: Tracks all allocated regions by last-access time. When `alloc_pages()` would exceed the RAM limit, the oldest regions are evicted first.
- **Prefetch**: `fcntl(F_RDADVISE)` advisory read-ahead for upcoming weight chunks
- **Async I/O**: GCD `dispatch_io` for background reads from SSD

### 5.3 Memory Budget Configuration

```cpp
struct MemoryConfig {
    size_t ram_limit        = 48 GB;
    size_t weight_buffer_mb = 8192;   // Double-buffered weight streaming
    size_t kv_hot_mb        = 2048;   // FP16 KV
    size_t kv_warm_mb       = 6144;   // TurboQuant 3.5-bit KV
    size_t kv_cool_mb       = 4096;   // TurboQuant 2.5-bit KV
    size_t scratch_mb       = 4096;   // Activations / intermediates
};
```

### 5.4 Resident Mode

For models that fit in RAM, all weight buffers can be pre-cached as MTLBuffers at load time. `preload_all_buffers()` touches every page to fault them into physical RAM, and `pre_allocate_activation_buffer()` avoids reallocation on the per-token path.

### 5.5 Streaming Mode

For models too large for RAM (e.g., 80B at INT4 = ~40 GB), the `Scheduler` orchestrates layer-by-layer streaming:

```
Layer L-1: EVICTED (madvise DONTNEED)
Layer L:   COMPUTING on GPU
Layer L+1: PREFETCHING from SSD (async via GCD)
Layer L+2: COLD (not yet touched)
```

State machine per layer: `COLD -> PREFETCHING -> READY -> COMPUTING -> EVICTED`

---

## 6. KV Cache

### 6.1 CPU-Side Flat Cache (Active -- Current Production Path)

`KVStore` is a simple flat FP32 buffer: `[num_layers][max_seq_len][num_kv_heads * head_dim]`. Separate key and value buffers per layer. This is what runs today.

- `key_at(layer, seq_pos)` / `value_at(layer, seq_pos)`: direct pointer access
- `advance(layer)`: increment sequence length
- Per-layer sequence tracking

### 6.2 GPU KV Cache (Infrastructure Built)

The `GPUBufferPool` in `ComputeDispatch` includes `kv_keys` and `kv_values` buffer IDs for a GPU-resident KV cache: `[num_layers * max_seq * kv_dim]` floats. `init_kv_cache()` allocates these, and `attention_gpu()` writes K/V directly into the GPU cache and runs attention without any CPU round-trip.

**Status**: The plumbing is complete (buffer allocation, GPU attention kernel, cache indexing). Performance was worse than CPU attention due to commit overhead (see Section 8), so the engine currently uses the CPU flat cache path.

### 6.3 TurboQuant Compression (Implemented)

`TurboQuantKV` implements a novel KV cache compression scheme:

1. **Random rotation**: `y = Pi * x` where Pi is a random orthogonal matrix (generated via QR decomposition of a Gaussian matrix)
2. **Lloyd-Max scalar quantization**: Each rotated coordinate is quantized to b bits (2, 3, or 4) using a precomputed optimal codebook
3. **Norm preservation**: L2 norm stored per vector for rescaling on dequantization

Supported bit widths:
- 4-bit: ~4x compression (Warm tier)
- 3-bit: ~5.3x compression (Cool tier)
- 2-bit: ~8x compression (theoretical, aggressive)

The rotation step is key: it makes each coordinate approximately i.i.d. regardless of the original vector's structure, so a single scalar codebook works well for all dimensions.

### 6.4 Paged Attention (Implemented)

`PagedKVCache` provides tiered compression with automatic promotion/demotion:

```
Hot  (FP16)            -- active attention window, zero decompression cost
Warm (TurboQuant 4-bit) -- recent context, ~4x compression
Cool (TurboQuant 3-bit) -- older context, ~5.3x compression
Cold (evicted)          -- spilled to SSD or freed
```

Pages are fixed-size (default 256 tokens). Keyed by `(layer, head_group, page_index, is_key)`. Thread-safe with per-cache mutex. Decompression happens on-the-fly into a scratch buffer during attention.

### 6.5 Prefix Cache (Implemented)

`PrefixCache` is a compressed radix tree (trie) indexed by token sequences. When a new request shares a prompt prefix with a previous one, the engine can skip recomputation and reuse cached KV pages.

Features:
- Compressed radix tree: edges carry multiple tokens (chains without branching are collapsed)
- LRU eviction with soft cap on node count (default 65,536 entries)
- Serialization to disk (`save_to_disk` / `load_from_disk`) for persistence across sessions
- Hit rate tracking

### 6.6 Eviction Strategies (Implemented)

`EvictionManager` supports four strategies:

- **H2O (Heavy-Hitter Oracle)**: Keeps initial tokens (system prompt), recent window, and tokens with highest cumulative attention scores. Sums attention weights across all layers.
- **SnapKV**: Uses an observation window of recent tokens to identify which KV positions each head consistently attends to. Votes across all (layer, head) pairs.
- **Combined**: Union of H2O and SnapKV keep-sets.
- **LRU**: Simple timestamp-based least-recently-used.

Eviction can run asynchronously on a background thread via `request_async_eviction()`.

---

## 7. Model Support

### 7.1 GGUF Import

`GGUFImporter` handles llama.cpp GGUF v3 files. Supported quantization types:

| GGUF Type | Handling |
|-----------|----------|
| F32, F16 | Direct conversion or re-quantize to INT4 |
| Q4_0, Q4_1 | Repack to NXF INT4 |
| Q8_0 | Passthrough as INT8 |
| Q2_K | Native passthrough (Codec::Q2K) |
| Q3_K | Native passthrough (Codec::Q3K) |
| Q4_K | Native passthrough (Codec::Q4K) |
| Q5_K, Q5_0, Q5_1 | Supported via block size tables |
| Q6_K | Supported |
| IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS, IQ1_S/M | Recognized (type enum defined) |

K-quant types (Q2_K through Q6_K) are the most important in practice since most community GGUF models use these formats. The importer reads the GGUF header and metadata, maps tensor names to NEXUS conventions, and streams tensor data into the NXF file with appropriate codec tags.

### 7.2 Safetensors Import

`SafetensorsImporter` handles HuggingFace safetensors files:

- Single file: `model.safetensors`
- Multi-shard: directory with `model-00001-of-00003.safetensors` etc.
- Reads `config.json` for architecture metadata (fallback: infer from tensor shapes)
- Supports F16, BF16, F32 source dtypes
- Quantizes to INT4 or INT8, or passes through as FP16/FP32

### 7.3 Architecture Support

**Dense transformers** (`Transformer` class):
- Standard Q/K/V/O attention projections
- SwiGLU FFN (gate + up + down projections)
- RMSNorm, RoPE
- GQA (grouped query attention)
- Tested with: Gemma, LLaMA-family models

**Hybrid SSM+MoE+Attention** (`HybridModel` class):
- Qwen3-Coder-Next's `12 x (3 x (Gated DeltaNet + MoE) + 1 x (GQA + MoE))` layout
- 48 layers total: 36 Type A (SSM+MoE) + 12 Type B (Attention+MoE)
- Type A: fused QKV + gated DeltaNet (simplified as linear attention) + MoE
- Type B: separate Q/K/V with Q/K RMSNorm + full GQA + MoE
- MoE: 512 experts, top-10 routing, DeepSeek-V3 style load balancing
- Shared expert gating
- Raw INT4 expert slicing from packed `[num_experts, ...]` tensors

### 7.4 Additional Components

- **CoreML draft model** (`draft_model.h/mm`): Stub for speculative decoding with a small CoreML model
- **Speculative decoding** (`speculative.h/cpp`): Framework for draft-verify speculation
- **MoE layer** (`moe_layer.h/cpp`): Standalone MoE FFN execution

---

## 8. Current Performance

### 8.1 Measured Numbers

| Model | Parameters | Quantization | Decode Speed | Peak RAM | Notes |
|-------|-----------|-------------|-------------|----------|-------|
| Gemma 4 31B | 31B | INT4 | **2.7 tok/s** | 168 MB -- 3.8 GB | Streaming mode, layers evicted after use |
| Qwen3-Coder-Next 80B | 80B | INT4 | **3.0 tok/s** | 8.6 GB | MoE: only active experts loaded |
| Prefill | -- | -- | **0.8--1.5s** | -- | Depends on prompt length |

The Qwen3-Coder-Next 80B result is notable: despite being 2.5x larger than Gemma 31B, it runs *faster* because MoE means only 10 of 512 experts are active per token, so effective compute per token is much smaller.

The Gemma 31B RAM range (168 MB -- 3.8 GB) reflects streaming mode: at minimum only the current layer's weights are resident; at peak, prefetched layers and KV cache accumulate.

### 8.2 Hardware Context

All numbers measured on Apple Silicon (M-series) with UMA. The memory bandwidth ceiling is ~200 GB/s (M2 Pro/Max class) to ~400 GB/s (M3 Ultra class). INT4 GEMV is memory-bandwidth bound: for a 31B INT4 model, ~16 GB of weights must be read per token.

At 200 GB/s bandwidth and 16 GB per token: theoretical floor is ~12.5 tokens/second. Current 2.7 tok/s means we are using roughly **22% of theoretical bandwidth** -- the rest is overhead.

---

## 9. Performance Analysis

### 9.1 The Bottleneck: Metal Command Buffer Commit Overhead

The dominant cost is not computation but **Metal command buffer synchronization**.

Each Metal command buffer `commit()` + `waitUntilCompleted()` costs approximately **0.5 ms** in overhead (kernel scheduling, GPU clock ramp, synchronization fence). For a 60-layer model with 7 GEMVs per layer:

```
420 dispatches x 0.5 ms/commit = 210 ms/token overhead
1000 ms / 210 ms = ~4.8 tok/s theoretical ceiling (overhead alone)
```

Add actual GEMV compute time on top and we get the observed 2.7--3.0 tok/s.

### 9.2 GPU Attention Attempt: 1.1 tok/s

Moving the full attention computation (including KV cache reads and softmax) to GPU was attempted via `attention_gpu()` on the `MetalBackend`. Result: **1.1 tok/s** -- worse than the CPU attention path.

Why: GPU attention added many fine-grained dispatches (KV cache write, Q*K^T, softmax, score*V) each requiring their own Metal command buffer commit. The overhead of these additional commits overwhelmed the compute savings.

The `flush_token()` / `resume_token()` mechanism was built specifically to handle this: mid-token, flush the command buffer so GPU results are CPU-readable (for softmax), then resume with a new command buffer. But each flush is another commit, and each commit is another 0.5 ms.

### 9.3 AMX/CPU Attempt: 0.2 tok/s

Running GEMV on CPU via Apple's AMX (through Accelerate/cblas_sgemm) was also tried. Result: **0.2 tok/s** -- dramatically worse.

Why: AMX operates on FP32 data. INT4 weights must be dequantized to FP32 on CPU before AMX can process them. This means:

- 4x bandwidth expansion (INT4 -> FP32)
- CPU dequantization compute cost
- The expanded FP32 data saturates memory bandwidth at 4x the rate

The Metal INT4 path avoids this entirely by dequantizing in the GPU shader, reading 4x less data from memory.

### 9.4 Why Metal INT4 Direct Read is Fastest

On Apple Silicon UMA, there is one memory bus shared by CPU and GPU. The key constraint is **memory bandwidth**, not compute. The winning strategy is:

1. **Read as little data as possible** -- INT4 = 0.5 bytes per weight vs 4 bytes for FP32 (8x less)
2. **Never copy data** -- mmap'd NXF weights are wrapped as MTLBuffers (same physical pages)
3. **Dequantize at the point of use** -- the GPU shader reads INT4 nibbles and converts inline
4. **Avoid CPU involvement** -- no CPU dequant, no intermediate buffers

This makes the Metal INT4 GEMV the clear winner despite the commit overhead.

---

## 10. Path to 30 tok/s

### 10.1 Fused Multi-Head Attention Kernel

The single largest win: replace per-head attention dispatches with ONE kernel that processes ALL attention heads in a single Metal dispatch.

Current: `num_heads` dispatches for Q*K^T, 1 dispatch for softmax (CPU), `num_heads` dispatches for score*V.
Target: 1 dispatch that does Q*K^T + softmax + score*V for ALL heads.

Expected savings: eliminate ~30--60 commits per layer for attention, potentially reducing per-token overhead from 210 ms to ~80 ms.

### 10.2 FP16 KV Cache

Currently KV cache is FP32 on CPU. Moving to FP16 halves the memory bandwidth for attention's K/V reads. This matters most for long sequences where attention cost grows linearly with sequence length.

### 10.3 Fewer, Fatter Kernels

The general principle: every Metal commit costs 0.5 ms. The path to speed is fewer commits with more work per commit.

**Fused FFN**: `gpu_ffn_fused()` already encodes W1 GEMV + W3 GEMV + SwiGLU + W2 GEMV as 4 dispatches on ONE command buffer with memory barriers. This reduces 4 commits to 1.

**Full-layer fusion**: Encode an entire layer (attention + FFN) as a single command buffer. The barrier is attention's softmax, which currently requires CPU involvement.

**Token-level batching**: The `begin_token()` / `end_token()` infrastructure exists but is not fully usable until attention runs entirely on GPU.

### 10.4 Separate Prefill vs Decode Strategies

Prefill processes many tokens and can use batched GEMM (M > 1) which has much better GPU utilization. Decode processes 1 token and is bandwidth-bound.

- **Prefill**: Use batched GEMM kernels, potentially with tiling optimized for M=prompt_length
- **Decode**: Use GEMV kernels optimized for M=1, minimize dispatches

### 10.5 Projected Impact

| Optimization | Commits Saved / Token | Speed Estimate |
|-------------|----------------------|---------------|
| Current baseline | 420 commits | 2.7 tok/s |
| Batch FFN (1 commit per FFN) | -180 | ~5 tok/s |
| Fused all-head attention (1 dispatch) | -180 | ~10 tok/s |
| Token-level single commit | -remaining | ~15-20 tok/s |
| FP16 KV + bandwidth optimizations | -- | ~25-30 tok/s |

These are rough estimates. The 30 tok/s target requires nearly all of the above plus careful kernel tuning.

---

## 11. What's Built vs What's Planned

### Fully Functional (Shipping Today)

| Component | Status | Notes |
|-----------|--------|-------|
| NXF format (reader + writer) | **Working** | 16 KB aligned, mmap streaming |
| GGUF importer (all K-quants) | **Working** | Q2_K through Q6_K native passthrough |
| Safetensors importer | **Working** | Single + multi-shard, config.json |
| Metal INT4 fused GEMV | **Working** | UMA zero-copy, primary compute path |
| ComputeDispatch (CPU/GPU routing) | **Working** | Auto-detection, fallback |
| Dense transformer (Gemma, LLaMA) | **Working** | Full prefill + decode |
| Hybrid SSM+MoE+Attention (Qwen3) | **Working** | 48-layer hybrid architecture |
| MoE router (DeepSeek-V3 style) | **Working** | Top-k routing, load balancing |
| CPU flat KV cache | **Working** | FP32, per-layer tracking |
| Scheduler (layer streaming) | **Working** | Prefetch + evict pipeline |
| Memory manager (slab + LRU) | **Working** | Page-aligned, eviction |
| CLI interface | **Working** | Interactive generation |
| HTTP server | **Working** | API endpoint |
| Tokenizer | **Working** | Encode + decode |
| Buffer caching (wrapped MTLBuffers) | **Working** | Eliminates per-token re-wrapping |
| Batch mode (per-FFN commit batching) | **Working** | Reduces commit count |

### Implemented but Not Active in Production Path

| Component | Status | Notes |
|-----------|--------|-------|
| GPU buffer pool | **Built** | Allocated at load, reused per token |
| GPU-resident KV cache | **Built** | Buffers allocated, attention_gpu() works |
| GPU attention kernel | **Built, slower** | 1.1 tok/s due to commit overhead |
| Token-batch mode | **Built, disabled** | Needs GPU-only attention to be useful |
| TurboQuant KV compression | **Implemented** | 2/3/4-bit Lloyd-Max + rotation |
| Paged attention (tiered) | **Implemented** | Hot/Warm/Cool/Cold tiers |
| Prefix cache (radix tree) | **Implemented** | Insert, lookup, LRU evict, persistence |
| H2O eviction | **Implemented** | Cumulative attention scoring |
| SnapKV eviction | **Implemented** | Observation window voting |
| Eviction manager (async) | **Implemented** | Background thread, callback API |
| Fused FFN (gpu_ffn_fused) | **Built** | 4 dispatches, 1 commit |
| GPU RMSNorm / SwiGLU / residual_add | **Built** | GPU-resident activation ops |
| Batched MoE FFN (single dispatch) | **Built** | All experts in one GPU dispatch |

### Planned / Stub

| Component | Status | Notes |
|-----------|--------|-------|
| Fused multi-head attention kernel | **Not started** | Critical for 30 tok/s |
| FP16 KV cache | **Planned** | Config exists, not wired |
| Speculative decoding | **Stub** | Framework exists, no draft model |
| CoreML draft model | **Stub** | Header only |
| ANS entropy coding (runtime) | **Codec defined** | Encoder/decoder in quant/ |
| QuIP# / AQLM runtime | **Codec defined** | Quantizer in quant/, no inference path |
| AWQ runtime | **Codec defined** | Quantizer in quant/, no inference path |

---

## 12. Codebase Statistics

| Directory | Lines | Files | Purpose |
|-----------|-------|-------|---------|
| `model/` | 3,956 | 8 | Transformer, HybridModel, MoE router/layer, speculative |
| `import/` | 3,290 | 6 | GGUF importer, safetensors importer, vocab extractor |
| `compute/` | 2,952 | 10 | ComputeDispatch, Metal backend/context, Accelerate GEMM, NEON dequant, CoreML |
| `kv/` | 2,606 | 10 | KVStore, TurboQuant, paged attention, prefix cache, eviction |
| `api/` | 2,127 | 5 | CLI, HTTP server, benchmark |
| `core/` | 1,164 | 5 | Engine, scheduler, tokenizer, config |
| `quant/` | 1,046 | 6 | GPTQ, AWQ, QuIP#, entropy coding |
| `format/` | 1,010 | 3 | NXF reader, writer, header definitions |
| `memory/` | 733 | 4 | UMA allocator, memory manager, prefetcher |
| **Total** | **18,884** | **73** | |

---

## 13. Lessons Learned

1. **Metal commit overhead dominates everything.** The GPU is fast; getting work to and from it is not. Every `commit()` + `waitUntilCompleted()` is ~0.5 ms. At 420 dispatches per token, this alone caps throughput at ~5 tok/s before any compute is done.

2. **UMA zero-copy is real and essential.** `MTLDevice.newBufferWithBytesNoCopy()` with mmap'd NXF data means weights go from SSD -> page cache -> GPU shader with zero intermediate copies. This is the single most important architectural decision.

3. **INT4 beats FP32 on bandwidth-bound workloads.** Dequantizing on the GPU shader is nearly free compared to the 4x bandwidth savings. The AMX/CPU path (0.2 tok/s) vs Metal INT4 path (2.7 tok/s) is a 13.5x difference entirely explained by bandwidth.

4. **MoE models have surprisingly good memory efficiency.** Qwen3-Coder-Next 80B uses 8.6 GB peak RAM because only 10 of 512 experts are loaded per token. The parameter count is misleading -- effective compute is much smaller.

5. **Build the infrastructure even if you can't use it yet.** GPU buffer pool, token-batch mode, and GPU KV cache were built before the fused attention kernel that makes them useful. When that kernel lands, the plumbing is ready.

6. **Fewer, fatter kernels is the optimization mantra.** Every API boundary (commit, sync, fence) has overhead. The path to performance is eliminating boundaries, not optimizing what happens between them.

7. **NXF's 16 KB alignment pays for itself.** The padding waste is negligible compared to the zero-copy mmap + Metal buffer wrapping it enables. GGUF's unaligned data requires copying; NXF's aligned data does not.
