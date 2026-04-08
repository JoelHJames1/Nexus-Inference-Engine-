# NXF: A Streaming-Native Tensor Format for Memory-Constrained LLM Inference on Apple Silicon

**Joel Hernandez**
NEXUS Inference Engine Project
April 2026

---

## Abstract

Large language models (LLMs) with 400B+ parameters require 200+ GB of storage even under aggressive 4-bit quantization, far exceeding the 48-128 GB unified memory available on consumer Apple Silicon machines. Existing tensor formats --- GGUF (llama.cpp), safetensors (HuggingFace), ONNX --- assume that the quantized model resides entirely in memory, leading to out-of-memory failures or catastrophic mmap thrashing on memory-constrained hardware. We present **NXF (Nexus Format)**, a streaming-native binary tensor container designed from first principles for layer-by-layer weight streaming from NVMe SSD to Apple Silicon's Unified Memory Architecture (UMA). NXF introduces a 64-byte fixed header, a binary tensor index with 24-byte chunk descriptors, 16 KB chunk alignment matching the arm64 macOS virtual memory page size, per-chunk codec selection supporting 11 compression schemes (FP16 through entropy-coded 2-bit quantization), MoE expert routing metadata, and a KV cache sidecar section for persistent prefix caches. We provide a quantitative memory footprint analysis showing that NXF enables streaming inference of a 405B-parameter model within a 48 GB memory envelope, requiring only 2-3 transformer layers resident at any time, compared to GGUF's requirement of full model residency. NXF is the foundation of the NEXUS inference engine, a C++20/Metal runtime purpose-built for Apple Silicon.

---

## 1. Introduction

### 1.1 Problem Statement

The rapid scaling of large language models has produced architectures --- Meta's LLaMA 3.1 405B [1], DeepSeek-V3 671B [2] --- whose parameter counts vastly exceed the memory capacity of consumer and workstation-class hardware. A 405B-parameter model stored in FP16 requires approximately 810 GB. Even with aggressive 4-bit weight quantization (GPTQ [3], AWQ [4]), the compressed representation exceeds 200 GB. Apple's highest-end consumer machine, the Mac Studio with M4 Ultra, provides 512 GB of unified memory; the more common M4 Max configurations offer 48-128 GB.

Every existing inference runtime and tensor format was designed under the assumption of **full model residency**: the entire quantized weight set must fit in addressable memory simultaneously. llama.cpp's GGUF format [5] relies on memory-mapped I/O where the operating system pages the entire file into the address space. HuggingFace's safetensors [6] provides a flat memory-mapped view of tensors. ONNX [7] and TensorRT [8] serialize computation graphs with embedded weights for GPU-resident execution. When the model exceeds available RAM, these approaches degrade to OS-level page thrashing --- a pathology that reduces throughput by 10-100x due to random I/O patterns that conflict with sequential SSD access [9].

### 1.2 Motivation

Apple Silicon's Unified Memory Architecture (UMA) presents a unique opportunity. The CPU, GPU, and Neural Engine share a single physical memory pool with bandwidths of 546-819 GB/s (M4 Max/Ultra) [10]. The NVMe subsystem delivers 5-7 GB/s sequential read throughput. This bandwidth ratio (roughly 100:1) means that if inference computation is sufficiently dense, we can overlap weight streaming from SSD with GPU execution on previously loaded weights, keeping the GPU utilization high while never materializing more than a small fraction of the model in memory.

However, exploiting this pipeline requires a **purpose-built tensor format** that supports:

1. **Chunk-level random access** --- individual weight blocks can be loaded independently without reading surrounding data.
2. **Page-aligned storage** --- chunks align to the operating system's virtual memory page size to enable zero-copy mmap without wasted memory.
3. **Per-chunk codec selection** --- different layers, experts, or even sub-blocks of a single tensor can use different quantization schemes.
4. **Architecture metadata** --- MoE routing tables, expert counts, and KV cache configuration must be accessible without parsing model code.
5. **Streaming-first I/O** --- the format must support prefetch hints, double-buffered loading, and partial reads without requiring a global file scan.

No existing format satisfies all five requirements. NXF does.

### 1.3 Contributions

This paper makes the following contributions:

- **NXF format specification**: A complete binary tensor container with a 64-byte header, binary tensor index using 24-byte chunk descriptors, and 16 KB page-aligned chunk data, designed for streaming inference on Apple Silicon.
- **Multi-codec architecture**: Per-chunk codec selection supporting 11 codecs from FP32 to entropy-coded 2-bit additive quantization, enabling mixed-precision inference within a single model.
- **Streaming I/O design**: Detailed integration with macOS-specific APIs (`mmap`, `madvise`, `fcntl(F_RDADVISE)`, GCD `dispatch_io`) for overlapped I/O and compute.
- **Quantitative analysis**: Memory footprint calculations demonstrating that a 405B model can be served within a 48 GB envelope using NXF streaming, compared to GGUF's minimum requirement of ~203 GB resident memory.

---

## 2. Background and Related Work

### 2.1 GGUF (llama.cpp)

GGUF (GPT-Generated Unified Format) [5] is the dominant open-source tensor format, used by llama.cpp and its derivatives (ollama, llamafile, koboldcpp). GGUF stores model metadata as key-value pairs followed by a flat array of quantized tensors. The format supports multiple quantization types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, etc.) but applies a **single quantization scheme per tensor**. The entire file is memory-mapped via `mmap()`, and llama.cpp assumes that the OS will page tensors into physical RAM as they are accessed.

**Limitations for streaming:**
- **No chunk-level granularity.** Each tensor is stored as a contiguous block. Loading a single 4096x4096 weight matrix requires mapping the entire tensor region; partial reads require application-level offset management.
- **4 KB alignment assumption.** GGUF aligns tensor data to 32 bytes (the GGML tensor alignment), not to VM page boundaries. On Apple Silicon where the VM page size is 16 KB (not 4 KB as on x86_64), this means tensor boundaries rarely coincide with page boundaries, causing wasted memory when selectively paging.
- **No per-chunk codec.** A tensor is either all Q4_0 or all Q8_0. Mixed precision within a tensor (e.g., attention heads at higher precision, FFN at lower) is not supported.
- **Full-residency assumption.** llama.cpp's `llama_model_load()` maps the entire file. For a 405B Q4_K_M model (~203 GB), this requires 203 GB of virtual address space and, in practice, physical RAM to avoid thrashing.
- **No MoE routing metadata.** Expert routing must be inferred from tensor naming conventions.

### 2.2 Safetensors (HuggingFace)

Safetensors [6] is HuggingFace's format designed for safe, fast tensor loading. It uses a JSON header followed by raw tensor data. The format's primary contribution is **safety** --- it avoids Python pickle deserialization attacks --- and **speed** --- direct memory mapping of tensor data without parsing overhead.

**Limitations for streaming:**
- **Flat memory map.** Like GGUF, the entire data region is memory-mapped as a contiguous block. The format provides no chunking mechanism.
- **No compression.** Safetensors stores raw dtype values (FP16, BF16, FP32). All quantization must happen at load time, which defeats the purpose of pre-compressed storage for streaming.
- **JSON header overhead.** The header is variable-length JSON, requiring parsing before any tensor access. For streaming, a fixed-size binary header with known offsets is preferable.
- **No architecture metadata.** Model architecture (layer count, head dimensions, etc.) is not part of the format; it must come from a separate `config.json`.

### 2.3 ONNX and TensorRT

ONNX (Open Neural Network Exchange) [7] serializes both the computation graph and weight tensors in a Protocol Buffers container. TensorRT [8] compiles ONNX models into optimized GPU execution plans.

**Limitations for streaming:**
- **Graph-centric.** ONNX encodes operator topology, not just weights. This couples the format to a specific computation graph, making it impossible to stream individual weight tensors independently.
- **GPU-resident assumption.** TensorRT engines assume all weights reside in GPU VRAM. There is no mechanism for demand-paging from storage.
- **NVIDIA-specific.** TensorRT is unavailable on Apple Silicon. ONNX Runtime supports CoreML but does not expose chunk-level I/O control.
- **No mixed-precision per-tensor.** While ONNX supports multiple data types, the quantization scheme is global or per-operator, not per-chunk.

### 2.4 Other Related Work

**vLLM's PagedAttention** [11] introduced virtual memory concepts to KV cache management, treating KV blocks as fixed-size pages with a page table for non-contiguous allocation. NXF extends this paging philosophy to *weight storage*, not just KV caches.

**DeepCompression** [12] demonstrated that combining pruning, quantization, and Huffman coding can reduce model size by 35-49x. NXF's multi-codec architecture generalizes this by allowing arbitrary codec pipelines per chunk, including post-quantization entropy coding with ANS [13].

**FlexGen** [14] explored offloading LLM inference to SSD with a linear programming-based scheduling algorithm. However, FlexGen operates on unmodified model files and suffers from I/O inefficiency due to non-aligned reads. NXF's page-aligned chunking addresses this directly.

---

## 3. NXF Format Design

### 3.1 Design Principles

NXF is governed by four design principles:

1. **Streaming first.** Every design decision optimizes for sequential, layer-by-layer weight loading with minimal random I/O.
2. **Apple Silicon native.** Alignment, page sizes, and I/O APIs target arm64 macOS specifically.
3. **Codec-agnostic.** The format is a container; it does not mandate a quantization scheme. New codecs can be added without format version changes.
4. **Self-describing.** A single NXF file contains all information needed to run inference: architecture, quantization parameters, expert routing, and KV cache configuration.

### 3.2 File Layout

An NXF file consists of four contiguous regions:

```
Offset 0                                    Offset N
+----------+-----------+--------------+------...------+----------+
|  Header  | Manifest  | Tensor Index | Chunk Data... | KV Side- |
| (64 B)   | (JSON/FB) | (binary)     | (16KB-aligned)| car      |
+----------+-----------+--------------+------...------+----------+
```

Each region's offset and size are recorded in the header, enabling direct seeking to any section without scanning.

### 3.3 Header Structure (64 bytes)

The NXF header is a fixed 64-byte structure at file offset 0:

```c++
struct NXFHeader {
    uint32_t magic;              // 0x3146584E ("NXF1" little-endian)
    uint16_t version;            // 1
    uint16_t flags;              // Reserved (encryption, endianness)
    uint64_t manifest_offset;    // Byte offset to manifest blob
    uint64_t manifest_size;      // Size of manifest in bytes
    uint64_t tensor_index_offset;// Byte offset to tensor index
    uint64_t tensor_index_size;  // Size of tensor index in bytes
    uint64_t data_offset;        // Byte offset to first chunk
    uint64_t total_file_size;    // Integrity: expected file size
    uint8_t  reserved[8];        // Padding to 64 bytes
};
static_assert(sizeof(NXFHeader) == 64);
```

**Design rationale:**

- **Fixed 64 bytes.** The header fits in a single cache line on most architectures. Reading it requires exactly one I/O operation. The `static_assert` guarantees binary compatibility.
- **Magic number `NXF1`.** Encodes format name and major version. Readers reject files with unknown magic immediately, before any further parsing.
- **Explicit offsets.** Every section is located by absolute file offset, not by scanning from the end of the previous section. This enables parallel reads: the manifest and tensor index can be loaded concurrently.
- **Total file size.** Serves as a quick integrity check. If the file is truncated (e.g., interrupted download), the reader detects the mismatch before attempting to parse corrupt data.
- **Reserved bytes.** The 8-byte reserved field and the 16-bit flags field provide space for future extensions (e.g., per-file encryption flags, endianness markers, or checksum algorithm selectors) without breaking the header layout.

### 3.4 Manifest

The manifest is a structured metadata blob (JSON for human readability in v1, with FlatBuffers [15] planned for v2) encoding the model architecture:

```c++
struct ModelManifest {
    std::string architecture;     // "llama", "deepseek_v3", "mixtral"
    std::string name;             // "LLaMA-3.1-405B"
    uint32_t num_layers;          // 126 for LLaMA 405B
    uint32_t hidden_dim;          // 16384
    uint32_t num_heads;           // 128
    uint32_t num_kv_heads;        // 8 (GQA)
    uint32_t head_dim;            // 128
    uint32_t vocab_size;          // 128256
    uint32_t max_seq_len;         // 131072
    float    rope_theta;          // 500000.0
    float    rms_norm_eps;        // 1e-5

    // MoE fields (0 for dense models)
    uint32_t num_experts;         // e.g., 256 for DeepSeek-V3
    uint32_t num_active_experts;  // e.g., 8 (top-k routing)

    // Default quantization
    Codec    default_codec;       // e.g., Codec::GPTQ
    uint8_t  default_group_size;  // e.g., 128
};
```

The manifest enables the inference engine to configure its compute pipeline (attention head counts, GQA ratios, RoPE parameters) and memory budget (layer sizes, expert counts) before reading any tensor data. For MoE models, the `num_experts` and `num_active_experts` fields allow the scheduler to pre-compute memory requirements: only `num_active_experts` expert weight sets need to be resident simultaneously.

### 3.5 Tensor Index

The tensor index is a binary array of tensor metadata entries. Each tensor's chunks are described by `ChunkDesc` structures:

```c++
struct ChunkDesc {
    uint64_t file_offset;        // Absolute byte offset in NXF file
    uint32_t compressed_size;    // On-disk size (after codec)
    uint32_t decompressed_size;  // In-memory size (after decode)
    uint32_t checksum;           // xxHash32 of compressed data
    Codec    codec;              // 1-byte codec identifier
    uint8_t  group_size;         // Quantization group size (e.g., 128)
    uint8_t  reserved[2];        // Padding to 24 bytes
};
static_assert(sizeof(ChunkDesc) == 24);
```

Each `TensorInfo` entry associates a tensor name (e.g., `"layers.42.attention.wq.weight"`) with its shape, original dtype, and an ordered list of `ChunkDesc` entries.

**Design rationale for 24-byte ChunkDesc:**

- **Compact.** A model with 10,000 chunks (typical for 405B) uses only 240 KB of index data, easily fitting in L2 cache.
- **Self-contained.** Each chunk carries its own codec and checksum. The reader does not need global state to decode any individual chunk.
- **xxHash32 checksum.** xxHash [16] provides collision-resistant integrity checking at >10 GB/s throughput on arm64, adding negligible overhead to chunk validation. The 32-bit hash provides a $2^{-32}$ false-positive rate per chunk, sufficient for corruption detection (not cryptographic security).
- **Group size per chunk.** Different chunks of the same tensor can use different quantization group sizes. This enables adaptive precision: attention projection weights might use group_size=64 for higher accuracy, while FFN weights use group_size=128 for better compression.

### 3.6 16 KB Chunk Alignment

All chunk data regions in NXF are aligned to 16 KB boundaries:

```c++
constexpr size_t kPageSize       = 16 * 1024;  // arm64 macOS VM page
constexpr size_t kChunkAlignment = kPageSize;
```

This is one of NXF's most consequential design decisions. The justification is rooted in Apple Silicon's virtual memory architecture:

**Why 16 KB, not 4 KB?**

On x86_64 Linux and Windows, the VM page size is 4 KB. On arm64 macOS (all Apple Silicon), the VM page size is **16 KB** [17]. When a memory-mapped file is accessed, the OS loads data in page-sized units. If a chunk starts at a non-page-aligned offset:

1. The OS must load the containing page, which includes data from adjacent chunks --- wasting memory and bandwidth.
2. Selective `munmap()` or `madvise(MADV_DONTNEED)` cannot release the chunk's memory without affecting adjacent data that shares the same page.
3. The Translation Lookaside Buffer (TLB) is indexed by page number. Misaligned chunks increase TLB pressure because a single logical chunk spans multiple pages unnecessarily.

By aligning every chunk to 16 KB:

- **Zero-copy mmap** works optimally. Each chunk occupies an integer number of VM pages. Loading chunk data via `mmap()` maps exactly the pages containing that chunk, with no wasted memory.
- **Selective eviction** is efficient. After a layer's computation completes, its chunks can be individually unmapped via `munmap()` or marked as `MADV_DONTNEED`, and the kernel reclaims exactly those physical pages.
- **Prefetch granularity** matches hardware. `madvise(MADV_WILLNEED)` and `fcntl(F_RDADVISE)` operate on page-aligned regions; 16 KB-aligned chunks require no rounding.

**Padding overhead analysis:**

The worst case is a chunk whose compressed data is 1 byte over a 16 KB boundary, wasting nearly 16 KB. In practice, weight chunks are 1-16 MB (the `kWeightSlabSize` default is 1 MB), so the padding overhead is:

$$\text{Overhead} = \frac{16\text{ KB}}{1\text{ MB}} = 1.56\%$$

For a 130 GB NXF file (405B model at QuIP# 3-bit + ANS), this is approximately 2 GB of padding --- a modest cost for the I/O efficiency gained.

### 3.7 Multi-Codec Architecture

NXF defines 11 codec identifiers:

```c++
enum class Codec : uint8_t {
    FP32     = 0,   // IEEE 754 single precision
    FP16     = 1,   // IEEE 754 half precision
    BF16     = 2,   // Brain floating point
    INT8     = 3,   // Symmetric 8-bit integer quantization
    INT4     = 4,   // Block quantized 4-bit, group_size=128
    GPTQ     = 5,   // GPTQ: per-group 4-bit with scales/zeros [3]
    AWQ      = 6,   // Activation-aware weight quantization [4]
    QUIP3    = 7,   // QuIP# 3-bit E8 lattice quantization [18]
    AQLM2    = 8,   // AQLM 2-bit additive codebooks [19]
    ANS      = 9,   // Entropy coded (lossless post-quant) [13]
    TURBO_Q  = 10,  // TurboQuant (KV cache codec) [20]
};
```

Each codec is described below with its role in the NXF ecosystem:

| Codec | Bits/Param | Use Case | Quality Impact |
|-------|-----------|----------|---------------|
| FP32 | 32 | Embedding tables, final LM head | Baseline |
| FP16 | 16 | Norm weights, small tensors | Negligible |
| BF16 | 16 | Alternative to FP16 for training-format compat | Negligible |
| INT8 | 8 | First/last layers, attention projections | <0.1% perplexity |
| INT4 | 4 | General weight storage | <0.5% perplexity |
| GPTQ | 4 | Compatibility with existing GPTQ models [3] | <0.5% perplexity |
| AWQ | 4 | Activation-aware, better on outlier-heavy layers [4] | <0.3% perplexity |
| QuIP# | 3 | Primary weight codec for extreme compression [18] | <1.0% perplexity |
| AQLM | 2 | Non-critical FFN layers, maximum compression [19] | 1-3% perplexity |
| ANS | Variable | Lossless post-quant entropy coding [13] | 0% (lossless) |
| TurboQuant | 2.5-3.5 | KV cache compression at runtime [20] | <0.3% (3.5-bit) |

**Per-chunk codec selection** is the key differentiator from GGUF's per-tensor codec. Consider a single weight matrix `layers.42.ffn.w1.weight` of shape [16384, 53248] (LLaMA 405B FFN up-projection). In GGUF, this entire 3.3 GB (FP16) tensor must use one quantization type. In NXF, the tensor is split into chunks, and each chunk independently specifies its codec:

- Chunks containing outlier-heavy columns (identified during offline calibration) can use INT8 or AWQ for higher fidelity.
- The majority of chunks use QuIP# 3-bit for maximum compression.
- All chunks are post-processed with ANS entropy coding for an additional 10-20% lossless compression [12].

This **mixed-precision within a single tensor** is a capability unique to NXF among open tensor formats.

### 3.8 MoE Routing Metadata

For Mixture-of-Experts models (e.g., DeepSeek-V3 [2] with 256 experts, 8 active per token; Mixtral [21] with 8 experts, 2 active), the manifest encodes:

- `num_experts`: Total expert count.
- `num_active_experts`: Top-k experts activated per token.

The tensor index uses a naming convention (`layers.{L}.experts.{E}.{param}`) that the scheduler parses to identify expert-specific weight tensors. When the router selects experts for a given token, the streaming engine loads **only** the selected experts' chunks, avoiding the bandwidth cost of loading all experts.

For a model like DeepSeek-V3 (671B total, 37B active per token), this reduces the per-token weight streaming requirement by approximately 18x:

$$\frac{671\text{B}}{37\text{B}} \approx 18.1\text{x reduction}$$

Combined with 4-bit quantization, the active parameter set is approximately:

$$37\text{B} \times 0.5 \text{ bytes/param (INT4)} = 18.5 \text{ GB}$$

This fits comfortably within a 48 GB memory budget.

### 3.9 KV Sidecar Section

NXF reserves a section (or companion sidecar file) for persistent KV cache storage. This enables:

1. **Prefix cache persistence.** After computing KV for a system prompt or few-shot prefix, the resulting KV pages can be serialized to the NXF sidecar. Subsequent inference sessions reuse this prefix without re-computation --- a technique inspired by SGLang's RadixAttention [22].
2. **Session resumption.** A conversation's KV cache can be checkpointed to disk and restored, enabling pause/resume of long-context interactions.
3. **Tiered KV compression.** The sidecar stores KV pages compressed with TurboQuant [20] at varying bitwidths:

| Tier | Token Age | Codec | Bits/Channel | Compression vs FP16 |
|------|-----------|-------|-------------|---------------------|
| Hot | Current token | FP16 | 16 | 1.0x |
| Warm | Recent (<1K tokens) | TurboQuant MSE | 3.5 | 4.6x |
| Cool | Older (1K-8K tokens) | TurboQuant Prod | 2.5 | 6.4x |
| Cold | Oldest | Evicted (H2O [23]) | 0 | --- |

---

## 4. Streaming Architecture

### 4.1 Layer-by-Layer Weight Streaming

Traditional inference loads all model weights into memory before generating the first token. NXF enables a fundamentally different execution model: **layer-by-layer streaming**.

During autoregressive decoding, each token generation requires a forward pass through all $L$ transformer layers sequentially. At any given moment, the compute backend (Metal GPU or Accelerate CPU) operates on at most one layer's weights. NXF exploits this temporal locality:

```
Time ──────────────────────────────────────────────>

GPU:   [Compute Layer L-1] [Compute Layer L  ] [Compute Layer L+1]
SSD:        [Load Layer L  ] [Load Layer L+1 ] [Load Layer L+2 ]
Evict: [Evict Layer L-2 ]  [Evict Layer L-1]  [Evict Layer L  ]
```

At any time, at most 2-3 layers' weights reside in memory:
- **Layer L**: Currently being computed.
- **Layer L+1**: Being prefetched from SSD (double-buffer).
- **Layer L-1**: Being evicted (asynchronous `munmap` / `MADV_DONTNEED`).

### 4.2 Memory-Mapped I/O with Prefetch

The NXF reader uses `mmap()` to map chunk regions on demand:

```c++
const void* NXFReader::map_chunk(const ChunkDesc& desc) {
    // Page-align the offset (NXF guarantees alignment)
    size_t aligned_offset = desc.file_offset;  // Already 16KB-aligned
    size_t length = /* round up to page boundary */ ...;

    void* ptr = mmap(nullptr, length, PROT_READ,
                     MAP_PRIVATE | MAP_FILE, fd_, aligned_offset);
    // Record for later unmapping
    mapped_regions_.push_back({ptr, length});
    return ptr;
}
```

Because NXF chunks are 16 KB-aligned, the `mmap` offset is always page-aligned --- a **requirement** of the `mmap` system call on all POSIX systems. Formats without page-aligned data (GGUF, safetensors) must round down to the nearest page boundary, mapping extra data and complicating offset calculations.

### 4.3 Prefetch Pipeline

Before computation of layer $L$ begins, the streaming engine issues prefetch hints for layer $L+1$:

```c++
// macOS-specific: fcntl F_RDADVISE for read-ahead
struct radvisory ra;
ra.ra_offset = next_layer_offset;
ra.ra_count  = next_layer_size;
fcntl(fd_, F_RDADVISE, &ra);

// Also: madvise on already-mapped regions
madvise(ptr, length, MADV_WILLNEED);
```

The `fcntl(F_RDADVISE)` call is macOS-specific and tells the unified buffer cache to begin reading ahead from the specified offset. Unlike `posix_fadvise` (unavailable on macOS), `F_RDADVISE` is the idiomatic mechanism for I/O prefetching on Darwin systems [17].

### 4.4 Double-Buffered Streaming

The NEXUS memory manager allocates two weight buffers (Buffer A and Buffer B), each sized to hold one transformer layer's weights:

```
Buffer A: [Layer L weights]     -> GPU computing
Buffer B: [Layer L+1 weights]   -> SSD loading (GCD dispatch_io)
                                    ↓ (swap)
Buffer A: [Layer L+2 weights]   -> SSD loading
Buffer B: [Layer L+1 weights]   -> GPU computing
```

For a 405B model with 126 layers, each layer's weight set (attention projections + FFN) at QuIP# 3-bit occupies approximately:

$$\frac{405\text{B params}}{126\text{ layers}} \times 0.375 \text{ bytes/param (3-bit)} \approx 1.2 \text{ GB/layer}$$

Two buffers require approximately 2.4 GB --- well within budget. The `MemoryConfig` in NEXUS allocates 8 GB for weight buffers (`weight_buffer_mb = 8192`), providing generous headroom for prefetch windows of 2-3 layers.

---

## 5. Apple Silicon Optimizations

### 5.1 16 KB Virtual Memory Pages

As discussed in Section 3.6, Apple Silicon uses 16 KB VM pages [17]. This affects NXF in three ways:

1. **Alignment.** All chunk offsets are multiples of 16,384 bytes.
2. **mmap granularity.** The minimum mappable unit is 16 KB. Sub-page-sized chunks (rare in practice) still consume a full page of physical memory.
3. **TLB efficiency.** With 16 KB pages, a 1 MB weight chunk requires only 64 TLB entries, vs. 256 entries with 4 KB pages. The M4's TLB is finite; reducing pressure improves address translation performance.

### 5.2 UMA Zero-Copy

Apple Silicon's Unified Memory Architecture allows CPU and GPU to share the same physical memory without explicit copies. NXF exploits this through Metal's `storageModeShared` buffers:

```
NXF on SSD → mmap() → Virtual Memory → MTLBuffer(storageModeShared) → GPU
                           ↑
                    No copy required.
                    CPU and GPU see the same physical pages.
```

On discrete-GPU systems (NVIDIA, AMD), weight data must be explicitly copied from host RAM to GPU VRAM via PCIe (12-64 GB/s). On Apple Silicon, the mmap'd NXF chunk data is **directly accessible** by Metal compute shaders through a shared-mode buffer wrapping the same virtual address. This eliminates an entire copy step and its associated latency.

### 5.3 GCD Dispatch I/O

Grand Central Dispatch (GCD) provides `dispatch_io` channels for asynchronous, system-optimized file I/O [24]. The NXF streaming engine uses `dispatch_io_read()` to issue non-blocking reads that the kernel coalesces and schedules optimally:

```c++
dispatch_io_t channel = dispatch_io_create(
    DISPATCH_IO_RANDOM, fd_, queue_,
    ^(int error) { /* cleanup */ });

dispatch_io_read(channel, offset, length, queue_,
    ^(bool done, dispatch_data_t data, int error) {
        if (done && !error) {
            // Data is ready; signal compute thread
            dispatch_semaphore_signal(layer_ready_sem);
        }
    });
```

GCD I/O channels coalesce small reads, manage read-ahead, and integrate with the macOS I/O scheduler. This is the Darwin equivalent of Linux's `io_uring` [25], providing kernel-bypassed asynchronous I/O without the overhead of thread-per-read models.

### 5.4 fcntl(F_RDADVISE) for Predictive Prefetch

The macOS-specific `fcntl(F_RDADVISE)` system call [17] informs the kernel that a specific byte range will be needed soon. The kernel can then schedule I/O operations proactively, populating the unified buffer cache before the application faults on those pages. NXF issues `F_RDADVISE` for the next 2 layers' chunk regions during computation of the current layer:

$$\text{Prefetch window} = \text{Layer}_{L+1} + \text{Layer}_{L+2} \approx 2.4 \text{ GB}$$

At 5-7 GB/s NVMe throughput, this 2.4 GB prefetch completes in 340-480 ms, which must be less than the computation time for layer $L$. For a 405B model on the M4 Max GPU (achieving ~1-2 TFLOPS effective throughput at INT4 matmul), each layer's computation takes approximately:

$$\frac{2 \times 3.2\text{B params/layer} \times 2 \text{ FLOPs/MAC}}{1.5 \times 10^{12} \text{ FLOPS}} \approx 8.5 \text{ ms (prefill, batch=1)}$$

This presents a challenge: at batch size 1, computation is faster than I/O. The solution is to increase the effective batch size through continuous batching or speculative decoding, or to accept that SSD streaming introduces a latency floor of ~3.4 ms per layer (1.2 GB / 5 GB/s = 240 ms per layer when I/O-bound, amortized across tokens via prefetch overlap). During decode, where each token requires a full pass through all 126 layers, the total per-token I/O time is:

$$126 \text{ layers} \times 240 \text{ ms/layer} \div \text{overlap factor} \approx 5\text{-}15 \text{ s/token (I/O bound)}$$

This motivates aggressive prefetching and expert-only loading (for MoE models) to reduce the I/O requirement per token.

---

## 6. Comparison with Existing Formats

Table 1 provides a comprehensive comparison of NXF with existing tensor formats across dimensions critical to streaming inference.

**Table 1: Tensor Format Comparison**

| Feature | NXF | GGUF | safetensors | ONNX |
|---------|-----|------|-------------|------|
| **Primary use case** | Streaming inference | Memory-resident inference | Safe model storage | Graph exchange |
| **Streaming support** | Native (chunk-level mmap) | None (full-file mmap) | None (full-file mmap) | None |
| **Per-tensor codec** | Per-*chunk* (finer) | Per-tensor | None (raw dtype) | Per-operator |
| **Codec count** | 11 | ~15 (Q-types) | 0 | 3-4 |
| **Page alignment** | 16 KB (arm64 native) | 32 B (GGML align) | None | None |
| **Chunk granularity** | 16 KB - 16 MB | Whole tensor | Whole tensor | Whole tensor |
| **MoE routing metadata** | Manifest fields | None (naming convention) | None | Graph-embedded |
| **KV cache persistence** | Sidecar section | None | None | None |
| **Compression flexibility** | Quantization + entropy | Quantization only | None | Quantization only |
| **Integrity checking** | xxHash32 per chunk | None | None | Protobuf CRC |
| **Header size** | 64 B (fixed) | Variable (KV pairs) | Variable (JSON) | Variable (Protobuf) |
| **Architecture metadata** | Full manifest | Partial (KV pairs) | None | Full graph |
| **Apple Silicon optimized** | Yes (16KB, UMA, GCD) | No | No | No |
| **File overhead** | ~1.5% (alignment padding) | ~0% | ~0% | ~5-10% (graph) |
| **Random access cost** | O(1) via index | O(n) tensor scan | O(1) via JSON header | O(n) graph parse |

---

## 7. Preliminary Results: Memory Footprint Analysis

### 7.1 Methodology

We analyze the memory requirements for running Meta's LLaMA 3.1 405B model [1] under three scenarios: (A) GGUF with full residency, (B) safetensors with full residency, and (C) NXF with layer-by-layer streaming. All scenarios assume a context length of 8,192 tokens and greedy decoding (batch size 1).

**Model parameters:**
- 405B parameters
- 126 transformer layers
- Hidden dimension: 16,384
- Attention heads: 128 (Q), 8 (KV) --- Grouped Query Attention
- Head dimension: 128
- FFN intermediate: 53,248
- Vocabulary: 128,256

### 7.2 Weight Memory

| Configuration | Codec | Bits/Param | Weight Size | Resident Requirement |
|--------------|-------|-----------|-------------|---------------------|
| GGUF Q4_K_M | INT4 mixed | ~4.5 | ~228 GB | **228 GB** (full) |
| GGUF Q4_0 | INT4 | 4.0 | ~203 GB | **203 GB** (full) |
| safetensors FP16 | FP16 | 16.0 | ~810 GB | **810 GB** (full) |
| NXF QuIP# 3-bit | QuIP# | 3.0 | ~152 GB | **~2.4 GB** (2 layers) |
| NXF QuIP# + ANS | QuIP# + ANS | ~2.5 | ~127 GB | **~2.0 GB** (2 layers) |
| NXF AQLM 2-bit | AQLM | 2.0 | ~101 GB | **~1.6 GB** (2 layers) |

Under NXF streaming, only the double-buffer (2 layers) must be resident. The remaining ~125-225 GB resides on SSD and is streamed on demand.

### 7.3 KV Cache Memory

For 8,192 tokens with GQA (8 KV heads, dimension 128, 126 layers):

$$\text{KV (FP16)} = 2 \times 126 \times 8 \times 128 \times 8192 \times 2 \text{ bytes} = 4.13 \text{ GB}$$

With TurboQuant tiered compression (Section 3.9):

| Tier | Tokens | Bits | Size |
|------|--------|------|------|
| Hot (FP16) | 256 | 16 | 0.13 GB |
| Warm (3.5-bit) | 2,048 | 3.5 | 0.22 GB |
| Cool (2.5-bit) | 5,888 | 2.5 | 0.46 GB |
| **Total** | **8,192** | --- | **0.81 GB** |

TurboQuant reduces the 4.13 GB KV cache to 0.81 GB --- a **5.1x** compression.

### 7.4 Total Memory Budget (48 GB Mac)

**Table 2: Memory Budget --- LLaMA 3.1 405B on 48 GB Mac**

| Component | GGUF (full) | NXF Streaming |
|-----------|-------------|---------------|
| OS + system overhead | 6 GB | 6 GB |
| Model weights (resident) | 203 GB | 2.4 GB |
| KV cache | 4.1 GB | 0.8 GB |
| Compute scratch (activations) | 4 GB | 4 GB |
| Draft model (speculative) | --- | 2 GB |
| Prefetch headroom | --- | 8 GB |
| **Total** | **217 GB** | **23.2 GB** |
| **Fits in 48 GB?** | **No** | **Yes** |

NXF streaming reduces the resident memory from 217 GB to 23.2 GB --- a **9.4x reduction** --- enabling 405B inference on a 48 GB machine with 24.8 GB of headroom for OS file cache, larger context windows, or batch processing.

### 7.5 Storage and I/O Requirements

The full NXF file (405B at QuIP# 3-bit + ANS) occupies approximately 130 GB on SSD, including alignment padding. At the M4 Max's NVMe throughput of ~5-7 GB/s:

- **Time to stream one layer:** $1.2 \text{ GB} / 6 \text{ GB/s} \approx 200 \text{ ms}$
- **Time for full forward pass (126 layers):** $126 \times 200 \text{ ms} = 25.2 \text{ s}$ (I/O only)
- **With compute overlap (double-buffering):** Effective latency is $\max(\text{compute}, \text{I/O})$ per layer. During decode with sufficient batching or MoE routing, significant overlap is achievable.

For MoE models (e.g., DeepSeek-V3 with 8 active experts out of 256), the per-token I/O is reduced by ~18x, bringing per-layer streaming time to approximately 11 ms --- well within the compute-bound regime.

---

## 8. Security and Integrity

NXF incorporates security measures at multiple levels:

1. **Per-chunk checksums.** Every `ChunkDesc` includes an xxHash32 [16] checksum of the compressed data. The reader validates each chunk before decompression, detecting corruption from truncated downloads, bit-rot, or tampering.
2. **Total file size check.** The header's `total_file_size` field enables instant detection of truncated files.
3. **No code execution.** Unlike Python pickle-based formats (which safetensors was designed to replace), NXF contains only numerical data and structured metadata. The parser performs bounds-checked reads with no dynamic code loading.
4. **Future: digital signatures.** The reserved header flags include space for a signature scheme identifier, enabling model publishers to sign NXF files for provenance verification.

---

## 9. Conclusion and Future Work

We have presented NXF, a streaming-native tensor format designed for memory-constrained LLM inference on Apple Silicon. NXF's key innovations --- 16 KB page-aligned chunking, per-chunk multi-codec support, and integrated MoE/KV metadata --- enable a fundamentally different inference paradigm where models are streamed from SSD rather than loaded into memory. Our analysis shows that NXF enables a 405B-parameter model to run within a 48 GB memory envelope, requiring only 23.2 GB of resident memory compared to GGUF's 217 GB.

### 9.1 Future Work

**Phase 2-3 integration.** The NXF format will be integrated with Metal compute shaders for fused dequantization-GEMM kernels (Research Paper #2), enabling direct GPU consumption of NXF chunk data through UMA zero-copy buffers.

**Entropy coding optimization.** The ANS codec (Codec::ANS) provides 10-20% lossless compression on top of quantization. We will implement FiniteStateEntropy [13] decoding optimized for arm64 NEON, targeting >5 GB/s decode throughput to avoid I/O pipeline stalls.

**TurboQuant KV compression.** The KV sidecar will implement the full TurboQuant [20] tiered compression pipeline with Metal shader acceleration for the random rotation and scalar quantization stages.

**Adaptive codec selection.** An offline calibration tool will analyze per-layer sensitivity (using a calibration dataset) and automatically assign optimal codecs per chunk, balancing compression ratio against quality loss.

**NXF v2.** Future format revisions will transition the manifest from JSON to FlatBuffers [15] for zero-copy deserialization, add support for tensor parallelism metadata (enabling multi-device inference), and introduce an optional encryption layer for proprietary model protection.

---

## References

[1] Dubey, A., et al. "The Llama 3 Herd of Models." *arXiv preprint arXiv:2407.21783*, 2024.

[2] Liu, A., et al. "DeepSeek-V3 Technical Report." *arXiv preprint arXiv:2412.19437*, 2024.

[3] Frantar, E., et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *ICLR*, 2023.

[4] Lin, J., et al. "AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration." *MLSys*, 2024.

[5] Gerganov, G., et al. "llama.cpp and GGUF format specification." *GitHub*, 2023-2025. https://github.com/ggerganov/llama.cpp

[6] HuggingFace. "Safetensors: A simple, safe way to store and distribute tensors." *GitHub*, 2023. https://github.com/huggingface/safetensors

[7] Bai, J., et al. "ONNX: Open Neural Network Exchange." *GitHub*, 2019. https://github.com/onnx/onnx

[8] NVIDIA. "TensorRT: High-Performance Deep Learning Inference." https://developer.nvidia.com/tensorrt

[9] Sheng, Y., et al. "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU." *ICML*, 2023.

[10] Apple Inc. "Apple M4 Max and M4 Ultra." *Apple Newsroom*, 2025.

[11] Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP*, 2023.

[12] Han, S., et al. "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding." *ICLR*, 2016.

[13] Collet, Y. "Finite State Entropy: A new breed of entropy coder." *GitHub*, 2013-2024. https://github.com/Cyan4973/FiniteStateEntropy

[14] Sheng, Y., et al. "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU." *ICML*, 2023.

[15] Google. "FlatBuffers: Memory Efficient Serialization Library." https://google.github.io/flatbuffers/

[16] Collet, Y. "xxHash: Extremely fast non-cryptographic hash algorithm." *GitHub*, 2012-2024. https://github.com/Cyan4973/xxHash

[17] Apple Inc. "Kernel Programming Guide: Virtual Memory System." *Apple Developer Documentation*, 2024.

[18] Tseng, A., et al. "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks." *ICML*, 2024.

[19] Egiazarian, V., et al. "Extreme Compression of Large Language Models via Additive Quantization." *ICML*, 2024.

[20] Zandieh, A., et al. "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." *arXiv preprint arXiv:2504.19874*, 2025. (ICLR 2026.)

[21] Jiang, A.Q., et al. "Mixtral of Experts." *arXiv preprint arXiv:2401.04088*, 2024.

[22] Zheng, L., et al. "SGLang: Efficient Execution of Structured Language Model Programs." *arXiv preprint arXiv:2312.07104*, 2023.

[23] Zhang, Z., et al. "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models." *NeurIPS*, 2023.

[24] Apple Inc. "Grand Central Dispatch (GCD) Reference." *Apple Developer Documentation*, 2024.

[25] Axboe, J. "Efficient IO with io_uring." *Kernel.org*, 2019.

---

*This paper is Research Paper #1 in the NEXUS Inference Engine series. Subsequent papers will address Metal compute shaders (#2), the multi-layer compression stack (#3), TurboQuant-Paged KV cache (#4), EAGLE-ANE speculative decoding (#5), and the comprehensive NEXUS system paper (#6).*
