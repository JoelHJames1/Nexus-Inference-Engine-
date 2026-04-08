Compression-Native Inference Runtime for 400B-Class Models on 48 GB RAM
Executive Summary
Running a dense 400B-parameter Transformer on a single 48 GB-RAM machine is infeasible with traditional inference: even extreme 4-bit quantization yields ~200 GB of weights
, plus a growing KV cache per token. Instead, we must treat LLM inference as a streaming, caching, and compression problem, not a monolithic memory load.

We propose a novel C++ runtime, “NEXUS”, built around these pillars:

NXF format: A new on-disk tensor container with block-level chunking and multiple codecs (mixed precision, entropy coding, product/AQ quantization). This lets us page in only needed weight blocks for each layer or expert.
Tiered memory manager: Treat weights and KV pages like virtual memory. Use mmap/io_uring/madvise (Linux) or Windows APIs to async-stream chunks from SSD, overlap I/O with compute, and enforce a strict RAM cap.
KV-as-DB: Store KV cache as persistent, addressable pages. A radix-tree index over token prefixes enables reuse (RadixAttention
) and safe overflow to disk. Compress or evict stale KV (via H2O/SnapKV heuristics) to bound memory.
Layered precision: Fuse on-the-fly dequant into computation kernels (FlashInfer/FlashAttention) for weights/KV. Support heterogeneous quant (GPTQ/AWQ, SmoothQuant, FP8, NVFP4) per tensor.
Execution optimizations: Predictive prefetch of next-layer weights or selected experts; block-sparse kernels; speculative decoding with small models (knowledge distillation
); and MoE routing.
Our target is “400B-class capability”: matching a 400–700B model’s task performance (e.g. LLaMA 3.1 405B
 or DeepSeek V3 671B) while never loading all parameters. Success metrics include strict peak RAM usage ≤48 GB; competitive time-to-first-token (TTFT) and tokens/sec; and negligible quality drop (e.g. <1% on benchmarks like MMLU/GPQA
). We assume fast NVMe storage (≥GB/s) and leave hardware details open: e.g. CPU-only vs. GPU-accelerated nodes. All key design choices are compared with cited literature and backed by proposed experiments.

Goals, Constraints, and Metrics
Objective: Achieve “400B-class” inference on a 48 GB machine. Interpreted as either (A) running an open 400–405B model (e.g. Meta’s LLaMA 3.1 405B
) via heavy streaming/offload, or (B) equivalent-capability: using sparse/MoE models plus RAG/distillation to match or exceed 400B quality.
Hard constraints: ≤48 GB resident memory. (Assume ~32–40 GB usable after OS/heap.) Limited GPU VRAM (if any); primary compute may be CPU or unified memory GPU.
Assumptions (open-ended):
NVMe SSD ~1–3 GB/s (tunable as needed) for weight/KV paging.
Multi-core x86_64 or ARM machine; optional GPU (e.g. RTX 5090/RTX 4090 or Apple Silicon).
No exact model is fixed; use publicly available large models for eval.
Success metrics:
Memory: Peak/resident RSS under 48 GB across benchmarks. Report breakdown (weights vs KV).
Latency: TTFT (prefill cost) and inter-token (per-step) latency in interactive setting.
Throughput: tokens/sec in multi-user and shared-prefix workloads (cascade decoding
).
Quality: Measured on LLM benchmarks. Aim for ≤1% accuracy drop (e.g. 75.5%→75.0% on Llama 3.1’s GPQA
) relative to baseline.
Scalability: Ability to run larger models by offload (report effective max model size given hardware).
We will list unspecified factors (SSD speed, inference engine, etc.) and explore sensitivity in benchmarks.

Architecture Overview
NEXUS is designed from the ground up for memory efficiency and streaming. It resembles an OS-level execution engine:

NXF file format: All model data (weights, biases, etc.) is stored in chunks on disk. Each chunk has a codec (quantization scheme, entropy coder) and a checksum. The NXF manifest also encodes graph architecture, layer shapes, and (if present) MoE routing tables. (See section below.)
Engine orchestration: A central scheduler coordinates multiple threads: I/O workers, compute workers, and KV management. It handles policies like lazy loading of layers, device placement, and batching or TTFT optimization.
Memory manager (tiered): Maintains three tiers:
Hot tier: malloc/arena for immediate tensors and small allocations.
Mapped tier: large regions of the model mapped via mmap; pages are faulted on access.
SSD tier: NXF on-disk chunks.
Threads use prefetch advice (POSIX_FADV_WILLNEED, madvise) to pull pages into RAM ahead of compute. When RAM cap is reached, the manager evicts least-needed pages (either highest-layer numbers or cold KV pages) to stay under the limit.
KV-as-DB: Keys/Values are stored in fixed-size pages (like PagedAttention
). Each page is uniquely identified (by model hash, layer, head group, token index). A prefix tree index maps token sequences to page lists (as in RadixAttention
). On new prompts, existing prefix hits reuse KV pages; on cache overflow, pages can be compressed (2-bit quant) or evicted (H2O heavy-hitter keeps). The KV DB can spill to SSD in FIFO or LRU fashion, similar to a swap space, ensuring the working set remains bounded.
Compute kernels: NEXUS integrates high-performance kernels. For attention and matmuls, we call into FlashAttention/FlashInfer engines
 that support our paged memory layouts. These kernels are IO-aware: they schedule computation on small blocks to maximize reuse. Where possible, we fuse decompression into the multiply (e.g. INT4→FP16 unpack + GEMM) to reduce bandwidth overhead.
MoE and hybrids: If using a mixture-of-experts model, the scheduler precomputes gating for each token, then only loads the selected expert weights into RAM. This allows, say, a 37B-active MoE model from DeepSeek-V3 (671B total) to run by streaming a few experts per token
. Similarly, SSM/RWKV components (for long sequences) are executed with constant state, reducing KV growth.
Runtime distillation: Optionally, a smaller model can run in parallel (speculative decoding
) or cascade mode to suggest tokens. Full model verification then only requires a few extra steps, boosting throughput.
The architecture contrast: unlike ggml/llama.cpp which assumes the whole quantized model is (almost) memory-resident, NEXUS never requires full residency. It continually streams, compresses, and evicts. This is inspired by PagedAttention/vLLM’s memory paging
, LMCache’s KV offload, and OS virtual memory techniques.

mermaid
Copy
flowchart LR
  Client[Client / API] --> Orchestrator[Engine Orchestrator]
  Orchestrator --> Scheduler[Request Scheduler]
  Scheduler --> MemoryMgr[Memory Manager]
  Scheduler --> KVStore[KV-DB & Prefix Index]
  MemoryMgr --> NXFLoader[NXF Loader (mmap/io_uring)]
  NXFLoader --> SSD[Disk (NXF & KV Pages)]
  MemoryMgr --> Compute[Compute Kernels<br/>(FlashInfer/FlashAttention)]
  Compute --> MemoryMgr
  Compute --> KVStore
  Orchestrator --> Telemetry[Telemetry/Logging]
Component Designs
NXF File Format
Purpose: NXF (Nexus Format) is a binary container designed for chunked streaming. It contains:

Header/Manifest: Model graph description (layers, shapes, types), quantization profiles, and (if MoE) expert routing metadata.
Tensor Index: For each tensor (e.g. Layer2/Attn/Wq), a list of ChunkDesc entries: each ChunkDesc has (file_offset, compressed_size, uncompressed_size, codec_id, checksum). Chunks are chosen (e.g. 16MB each) to align with storage pages.
Chunk Data: Compressed data blocks concatenated. Each block is aligned to 4K or larger.
Optional KV Store: We reserve a section or sidecar file where evicted KV pages and prefix indices can persist (so inter-request reuse is possible).
NXF is not a load-and-go format: on import we reorganize data for streaming, not as a flat array. For example, mixed-precision means some chunks of a weight matrix might be 4-bit quant, others 8-bit, etc. Entropy coding (e.g. ANS/Huffman) is allowed per chunk to squeeze out extra bits (DeepCompression style
).

C++ API Sketch
cpp
Copy
namespace nexus::format {
  enum class Codec : uint8_t { FP16, INT8, INT4, NVFP4, AQ_Multi, ENTROPY };
  struct ChunkDesc { uint64_t file_offset; uint32_t comp_bytes, decomp_bytes; uint32_t checksum; Codec codec; };
  struct TensorInfo { std::string name; std::vector<int64_t> shape; std::vector<ChunkDesc> chunks; };
  class NXFModel {
  public:
    // Load manifest and index
    static std::unique_ptr<NXFModel> load(const std::string& path);
    // Map (but don't load) a tensor chunk into memory
    // Returns pointer (void*) to mapped region of comp_bytes.
    void* map_chunk(const ChunkDesc& desc);

    // Unmap and free all pages
    void unload();
    const TensorInfo* get_tensor(const std::string& name) const;
  };
}
Memory mapping: On Linux, we use mmap(file_fd, ...) for the chunk region and rely on page faults to trigger actual IO. We tag pages with madvise(MADV_WILLNEED) when prefetching. On Windows, we use CreateFileMapping + MapViewOfFile and call PrefetchVirtualMemory on upcoming regions.

Memory Manager & Allocations
Allocation patterns: We use an arena/slab allocator. All weight chunks and KV pages are a multiple of a fixed page size (e.g. 64 KB or 1 MB for weights, 4 KB or 1 MB for KV). This minimizes fragmentation. The Memory Manager offers:

cpp
Copy
class MemoryManager {
public:
  MemoryManager(size_t ram_limit);
  // Reserve `bytes` of RAM (pages). Returns a handle or pointer.
  void* alloc_pages(size_t bytes);
  void  free_pages(void* ptr);
  // Async prefetch from file to a target address range
  void async_read(void* dst, uint64_t file_offset, size_t bytes);
  // Advise OS to prefetch future pages
  void prefetch(uint64_t file_offset, size_t length);
};
The eviction policy is important: we maintain LRU lists. For weights, we evict highest layers first (least likely to be re-used). For KV, we track “attention importance” (via H2O strategy
) so that tokens which are attended to later get retained. When eviction is needed, we compress or drop lowest-priority KV pages.

Example: Using posix_fadvise(fd, offset, len, POSIX_FADV_SEQUENTIAL) hints, plus explicit preadv via io_uring we pipeline. Unused pages are munmap-ed or madvise(MADV_DONTNEED) to free RAM.

KV-as-DB and Prefix Cache
Instead of a flat KV map, we store KV in pages (e.g. one page holds several key or value vectors). Each page key is (modelID, sessionID, layer, head_group, token_range). A radix-tree index maps token prefixes to pages. This mirrors SGLang’s RadixAttention
.

Insert/Fetch: After computing KV for a new token, split into pages and store in-memory + on-disk with a content hash. The index node for that prefix points to the new pages.
Prefix reuse: Before computing, check if the full input prefix exists in the index. If yes, reuse all its KV pages (skip prefill entirely).
Eviction/Compression: Keep a rolling window or heavy-hitter snapshot (per H2O
). When RAM is full, oldest or least-used KV pages are dumped to SSD (in NXF’s KV area) or quantized from FP16→FP8/2-bit (we use TensorRT-LLM’s NVFP4 idea or KIVI
).
This design generalizes:

Single session: Like llama.cpp’s KV cache but pageable.
Multi-session: Different requests share pages (if identical inputs).
Offline reuse: Entire prefix caches can be reused by future runs or shared services (cache warm-up).
Compute and Kernels
We abstract core operations:

cpp
Copy
namespace nexus::kernels {
  struct GEMMArgs { /* pointers, dims, strides, etc */ };
  struct AttentionArgs { /* Q, K, V pointers, dims, etc */ };

  void gemm(const GEMMArgs&);
  void flash_attention(const AttentionArgs&);
}
Implementations: We interface to FlashInfer for multi-layer autoregressive decoding. FlashInfer’s cascade mode can batch multi-token generation across layers
. It expects KV in blocks, matching our paged scheme. If FlashInfer/FlashAttention is not available (CPU-only), fallback to oneDNN or our own block-optimized matmul.

We also implement fused dequantization: e.g. for INT4 weights, an AVX512/NEON kernel will load 4 bits → unpack to 8-bit then to float for multiply, all in registers. This halved memory access.

Compute threads check with the scheduler if next weights or KV will be needed soon and yield to I/O threads for prefetch (leveraging asynchronous tasks).

Prefetching and Scheduling
Layer streaming: As soon as layer L completes for a token, the scheduler signals prefetcher to load layer L+1 weights (or experts) in the background. We tune prefetch “window” based on observed compute/I-O overlap.

Speculative decoding: Optionally, launch a lightweight model (e.g. 13B MiniGPT) ahead of the big model. The big engine then only verifies the top candidates (like TensorRT-LLM’s speculative decoding). This reduces average ITL without affecting correctness
.

Async I/O threads: Use io_uring submission/completion queues to parallelize disk reads. The Compute engine rarely blocks on I/O.

File Formats and Import
We provide tools to convert common model formats into NXF:

GGUF import: Parse ggml/GGUF (llama.cpp) weights, reshape and chunk.
HuggingFace safetensors import: Load HF weights (safe binary), then quantize/re-encode.
ONNX import: Extract weight tensors from ONNX (using ONNX Runtime APIs) into NXF.
NVFP4 import: If the model is already NVFP4-compressed, ingest it but re-pack into our blocks, possibly re-quantizing to unified scheme.
At runtime, NEXUS only reads NXF; it does not interpret safetensors/GGUF on-the-fly, to ensure the architecture control is maintained. (This addresses questions like the one about supporting popular formats vs. conversion.)

Compression Techniques and Trade-offs
We survey compression schemes with citations:

Technique	Storage Impact	Quality Loss (LLaMA-scale)	Compute Overhead	Key Citations
GPTQ/AWQ (3–4b weights)	~4× reduction vs FP16	<1% typical
Low–medium	[19]
SmoothQuant (W8A8)	~2× (8b→8b, weight-balance)	~0 (negligible)
Low	[21]
NVFP4 (4b FP)	~4×	~0.3% (GPQA gap 75.46→75.71)	Low (GPU-only)	(NVIDIA)
Additive/QV (2–3b, AQLM)	~10×+	1–5% (OK if graded)	High (decode)	[23]
Entropy coding (Huffman/ANS)	+10–20% beyond quant	0 (lossless)	Medium	[11]
Sparse pruning (e.g. 50%)	2× (weights)	~1–2%
High (unless sparse GEMM)	[25]
Structured pruning (2:4,4:8)	modest (needs hardware)	small	Low–medium	[25]
FP8/2b KV cache (KIVI)	~4× (2b vs FP16)	~1–2% (tunable)
Low–med	[19], (vLLM)
KV eviction (H2O/SnapKV)	Variable (e.g. 3× for 2× longer context)	small (task-dependent)
None (select)	[16]
Runtime distillation	N/A (tiny model)	~0 (speculative is exact)	Medium (two models)	[11]
Retrieval (RAG)	Shifts memory off-param	Up to 100% param reduction	Medium (DB lookup)	[15]

GPTQ/AWQ (weight PTQ): Offline 3–4-bit weight-only quant (per-row/block) with minimal loss
. Very popular in llama.cpp, vLLM, etc.
W8A8 (SmoothQuant): Moves Activation outliers into weights so 8-bit GEMMs suffice
. Good for INT8 hardware.
NVFP4: NVIDIA’s new 4-bit float for KV. Effectively ~4× KV compression with negligible loss on GPQA.
Additive Quantization (AQLM): Learn multiple small codebooks per weight to compress into ~2b. State-of-the-art for “extreme compression”
, but runtime decode is heavy.
Entropy coding: Post-process (e.g. Huffman on quant residuals) can squeeze another ~10–20% with zero loss
 (like in DeepCompression) at cost of decode CPU.
Pruning (SparseGPT): 50–60% unstructured sparsity with <0.2% perplexity loss
. Memory saving ~2× but requires sparse kernels.
LoRA/Low-rank: Not a focus here since it’s training-time (fine-tune) tech, but low-rank factorization could compress some matrices if integrated.
In summary, combining weight quant+entropy and KV quant+eviction yields large memory wins. For example, 4-bit weight quant × entropy × 2-bit KV quant can in theory reduce a 400B model + context to ~25–50 GB effective memory (needs verification). AQLM and heavy pruning might push further at higher compute cost.

Benchmark Plan and Experiments
We propose a suite of micro- and macro-benchmarks to validate the system:

Datasets/Workloads
Quality eval: Standard LLM benchmarks (MMLU, GSM8K, HumanEval, etc.) using (for example) LLaMA3-405B or DeepSeek-V3 reference. These define acceptable quality drop (e.g. <1%). [17†L65-L73][28†L1-L4]
Latency tests: Interactive chat prompts (few-shot and long-context QA). Measure TTFT and median ITL.
Throughput tests: Bulk generation under concurrency (e.g. 8 parallel prompts). Also “grouped prefix” scenarios (one prompt with N completions) to test cascade decoding.
Memory profiling: Use Linux tools or custom instrumentation to track peak RSS of weights vs KV vs overhead.
Microbenchmarks
I/O pipeline: Read throughput from NXF (mmap) vs sequential mmap vs io_uring reads; effect of madvise hints.
Dequant GEMM: Test performance of fused INT4→FP16 kernels vs standard FP16.
Attention engine: Prefill and decode speed of FlashAttention vs naive matmul for a given layer size.
KV operations: Latency of KV insert/retrieve, impact of 2-bit quant vs full precision.
Ablation Studies
We will measure end-to-end with various features toggled:

No streaming: Traditional load-all (should fail or thrash), baseline.
Streaming only: Load per-layer, no KV offload or prefix reuse.
+ KV prefix reuse: With persistent prefix caching.
+ KV quantization: FP16 → FP8 or 2-bit on older tokens.
+ KV eviction: Implement H2O/SnapKV eviction to cap KV.
+ Speculative decoding: Optional small-model draft.
+ MoE model: If available, compare dense vs MoE with routing (DeepSeek vs Llama).
Each ablation is measured on memory, latency, and quality. For example, “Configuration X achieves Y tokens/s with Z quality loss and uses M memory” in a chart.

Metrics
Latency (TTFT, ITL), throughput (tokens/sec) across configurations.
Memory (peak and breakdown).
Quality (accuracy/perplexity) vs baseline.
This plan follows MLPerf-type methodology for fairness and reproducibility.

Compatibility, Security, Reproducibility
Format compatibility: NEXUS imports GGUF (ggml) and safetensors (HF) to NXF. We provide tools, not a monkey-patch engine. ONNX import is via ONNX Runtime (for custom layers to standard tensors).
Reproducibility: Publish all code, data splits, and random seeds. Use deterministic kernels where possible. Follow MLCommons guidelines (fixed workloads, hardware logging).
Security: Model files (NXF) will include checksums per chunk (and optional digital signatures). NXF parsers are hardened (bounds checks, no remote code). Fuzzing and CI tests will target the NXF loader and importers (the Hugging Face safetensors avoids pickle risks).
Integrity: We’ll record and verify model hashes and code versions (in manifest).
Roadmap & Resources
Phase 1 (0–3 months): Build basic engine with NXF loader, memory manager, simple attention, and KG API. Demonstrate a small dense model (e.g. 7B) running with streaming.
Phase 2 (3–6 months): Add quant codecs and fused GEMMs (GPTQ, AWQ, SmoothQuant). Integrate FlashInfer. Test on ~70B models with weight streaming.
Phase 3 (6–9 months): Implement KV-as-DB with prefix tree, offload, and quantized KV. Run 175B-class demo.
Phase 4 (9–12 months): MoE and hybrid models (Switch/DeepSeek support). Retrieval interface (RAG). End-to-end “400B-equivalent” demos.
Team: 3–4 C++ systems engineers (memory & I/O, kernel integration, format spec), 1 ML engineer (quant/distill pipelines), 1 tester.
Compute: Multi-core CPU server (64+ GB RAM, NVMe), one or more GPU (A100/H100 or M3 Max) for kernel development and large-model testing.

Each choice is backed by literature and engineered for our target. The combination of streaming architecture and aggressive compression is novel compared to existing runtimes, and should markedly expand what can run on memory-limited hardware.