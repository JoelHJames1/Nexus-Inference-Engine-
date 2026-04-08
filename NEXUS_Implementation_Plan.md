# NEXUS Inference Engine — Final Implementation Plan

## Context

**What:** Build NEXUS, a C++ inference runtime with Metal compute shaders, purpose-built for Apple Silicon Macs. It will run 400B+ parameter LLMs on machines with 48-128 GB unified memory — something no existing engine can do well.

**Why:** Every current inference engine (llama.cpp, MLX, ollama) was designed for NVIDIA GPUs or assumes the model fits in memory. Apple Silicon's Unified Memory Architecture is a fundamentally different paradigm — zero-copy CPU/GPU sharing, 546-819 GB/s bandwidth, fast NVMe — that nobody fully exploits. We will.

**How:** Treat LLM inference as a streaming/caching/compression problem. Never load the full model. Stream weight chunks from SSD, compress KV cache with TurboQuant (Google Research, 2025), route only active MoE experts, and use speculative decoding on the Neural Engine.

**Repo:** https://github.com/JoelHJames1/Nexus-Inference-Engine-.git

---

## Competitive Analysis — Where We Win

| Engine | Peak tok/s (7B, Mac) | 400B Support | Paged KV | Continuous Batch | SSD Streaming |
|--------|---------------------|-------------|----------|-----------------|--------------|
| llama.cpp | ~150 | mmap thrash | No | No | No |
| MLX | ~230 | OOM if >RAM | No | No | No |
| Ollama (MLX) | ~1810 prefill | OOM if >RAM | No | No | No |
| vllm-mlx | ~400 | No | Partial | Yes | No |
| MLC-LLM | ~190 | No | Yes | No | No |
| **NEXUS** | **Target: 250+** | **Yes** | **Yes** | **Yes** | **Yes** |

---

## Core Technology: TurboQuant Integration

**Paper:** "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (Google Research/DeepMind, arXiv:2504.19874, ICLR 2026)

**Why it's our secret weapon for KV cache:**
- **3.5 bits/channel = quality neutral** (0.997 needle-in-haystack score, same as full precision)
- **2.5 bits/channel = marginal degradation** (4.5-6.4x compression)
- **Online / data-oblivious** — no calibration needed, quantize as tokens are generated
- **Outperforms KIVI** (0.997 vs 0.981) and **SnapKV** (0.997 vs 0.858) at 4x compression
- **Accelerator-friendly** — random rotation + scalar quantize = perfect for Metal shaders

**Installed implementations (for reference/prototyping):**
- `turboquant-mlx/` — MLX native (Apple Silicon), has `TurboQuantKVCache` with fused attention
- `turboquant/` — PyTorch reference (`turbokv` package), `TurboQuantizer` class
- `turboquant-pytorch/` — Extended PyTorch with Lloyd-Max codebook generation

**Our tiered KV compression pipeline:**
| Tier | Age | Codec | Bits/channel | Compression |
|------|-----|-------|-------------|-------------|
| Hot | Current token | FP16 | 16 | 1x |
| Warm | Recent (< 1K tokens) | TurboQuant MSE | 3.5 | 4.6x |
| Cool | Older (1K-8K) | TurboQuant Prod | 2.5 | 6.4x |
| Cold | Oldest | H2O eviction | 0 (evicted) | ∞ |

---

## Language & Technology Stack

**Core engine: C++20 + Metal Shading Language**
- All performance-critical paths: weight streaming, dequant, GEMM, attention, KV cache
- Metal compute shaders for GPU acceleration
- Accelerate framework / AMX for CPU-side matmul
- CoreML for Neural Engine (speculative decoding draft model)

**Python (tooling only):**
- Model conversion: GGUF/safetensors → NXF format
- Benchmarking and quality evaluation scripts
- TurboQuant prototyping (reference before C++ port)

**Build system:** CMake with Apple toolchain detection

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Client / API (HTTP + CLI)          │
├─────────────────────────────────────────────────────┤
│                  Engine Orchestrator (C++)            │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────┐  │
│  │ Request   │  │ Memory    │  │ KV-DB            │  │
│  │ Scheduler │  │ Manager   │  │ (TurboQuant +    │  │
│  │ (GCD)     │  │ (UMA)     │  │  Prefix Cache)   │  │
│  └─────┬─────┘  └─────┬─────┘  └────────┬─────────┘  │
│        │              │                  │            │
│  ┌─────▼──────────────▼──────────────────▼─────────┐ │
│  │              Compute Backend                      │ │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────────────┐ │ │
│  │  │ Metal   │  │Accelerate│  │ CoreML (ANE)    │ │ │
│  │  │ Shaders │  │ (AMX)    │  │ Draft Model     │ │ │
│  │  └─────────┘  └──────────┘  └─────────────────┘ │ │
│  └──────────────────────────────────────────────────┘ │
│                        │                              │
│  ┌─────────────────────▼──────────────────────────┐  │
│  │           NXF Loader (mmap + GCD async I/O)     │  │
│  └─────────────────────┬──────────────────────────┘  │
│                        │                              │
│                   NXF on SSD                          │
└─────────────────────────────────────────────────────┘
```

---

## Project Directory Structure

```
nexus/
├── CMakeLists.txt
├── src/
│   ├── core/               # Engine orchestrator, scheduler (C++)
│   │   ├── engine.h/cpp
│   │   ├── scheduler.h/cpp
│   │   └── config.h
│   ├── format/             # NXF format reader/writer (C++)
│   │   ├── nxf.h
│   │   ├── nxf_reader.cpp
│   │   └── nxf_writer.cpp
│   ├── memory/             # UMA-aware memory manager (C++)
│   │   ├── memory_manager.h/cpp
│   │   ├── uma_allocator.h/cpp
│   │   └── prefetcher.h/cpp
│   ├── compute/            # Compute backends (C++ + Metal)
│   │   ├── metal/
│   │   │   ├── metal_backend.h/cpp
│   │   │   └── metal_context.h/cpp
│   │   ├── accelerate/
│   │   │   └── gemm.h/cpp
│   │   ├── coreml/
│   │   │   └── draft_model.h/cpp
│   │   └── cpu/
│   │       └── dequant_neon.h/cpp
│   ├── kv/                 # KV cache with TurboQuant (C++)
│   │   ├── kv_store.h/cpp
│   │   ├── paged_attention.h/cpp
│   │   ├── prefix_cache.h/cpp
│   │   ├── turbo_quant.h/cpp    # C++ port of TurboQuant
│   │   ├── eviction.h/cpp       # H2O / SnapKV
│   │   └── kv_quantize.h/cpp
│   ├── quant/              # Weight quantization codecs (C++)
│   │   ├── quip_sharp.h/cpp
│   │   ├── gptq.h/cpp
│   │   ├── awq.h/cpp
│   │   └── entropy.h/cpp       # ANS/FSE codec
│   ├── model/              # Model graph, layer execution (C++)
│   │   ├── transformer.h/cpp
│   │   ├── moe_router.h/cpp
│   │   ├── moe_layer.h/cpp
│   │   └── speculative.h/cpp
│   ├── api/                # HTTP server + CLI (C++)
│   │   ├── http_server.h/cpp
│   │   └── cli.cpp
│   └── import/             # Format converters (C++)
│       ├── gguf_importer.h/cpp
│       └── safetensors_importer.h/cpp
├── shaders/                # Metal compute shaders (.metal)
│   ├── gemm_dequant.metal
│   ├── flash_attention.metal
│   ├── turbo_quant.metal        # TurboQuant on GPU
│   ├── rmsnorm.metal
│   └── activations.metal
├── tools/                  # Python tooling
│   ├── convert.py          # GGUF/safetensors → NXF
│   ├── quantize.py         # Quantization calibration
│   └── benchmark.py        # Quality/performance benchmarks
├── research/               # Research papers (written as we build)
│   ├── 01_nxf_format.md
│   ├── 02_uma_memory.md
│   ├── 03_metal_kernels.md
│   ├── 04_compression_stack.md
│   ├── 05_kv_turbo_quant.md
│   ├── 06_eagle_ane.md
│   └── 07_nexus_system.md
├── tests/
└── docs/
```

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-4)
**Goal:** Load and run a 7B model with basic layer streaming.
**Research Paper #1:** "NXF: A Streaming-Native Tensor Format for Memory-Constrained LLM Inference"

**Deliverables:**
1. **CMake build system** — Apple toolchain detection, Metal SDK, Accelerate framework linking
2. **NXF format v1** — Binary container with:
   - Header magic (`NXF1`), version, FlatBuffers manifest
   - Tensor index: name → ChunkDesc list (offset, sizes, codec, checksum)
   - 16 KB chunk alignment (Apple Silicon VM page size)
   - Codecs: FP16, INT8, INT4 (block quantized, group=128)
3. **UMA memory manager** — `mmap()` + `madvise(MADV_WILLNEED)` + `fcntl(F_RDADVISE)` + GCD `dispatch_io` for async reads, arena/slab allocator, RSS tracking via `mach_task_info()`
4. **Basic compute** — Accelerate `cblas_sgemm`/`cblas_hgemm` (auto-AMX), NEON INT4→FP16 dequant
5. **Model runner** — Layer-by-layer streaming: load → compute → evict → next layer. Greedy/top-k/top-p sampling. CLI: `nexus run model.nxf --prompt "Hello"`
6. **GGUF importer** — Parse GGUF, re-chunk into NXF. CLI: `nexus convert model.gguf model.nxf --quant q4`

**Milestone:** Run LLaMA 7B streaming on M-series Mac. Verify correctness vs llama.cpp.

---

### Phase 2: Metal GPU + Quantization (Weeks 5-8)
**Goal:** GPU-accelerated inference, outperform llama.cpp on Apple Silicon.
**Research Paper #2:** "Fused Dequantization-GEMM Kernels for Apple Metal"
**Research Paper #3:** "Multi-Layer Compression: QuIP# + AQLM + ANS for Extreme Model Compression"

**Deliverables:**
1. **Metal compute shaders:**
   - Fused dequant+GEMM: INT4 unpack → FP16 multiply, `storageModeShared` MTLBuffers (UMA zero-copy)
   - Flash attention: Br=32, Bc=32 tiles for 16 KB threadgroup memory, SIMD shuffle (32-wide)
   - RMSNorm, SiLU/GELU fused activation shaders
2. **Advanced quantization codecs:**
   - QuIP# 3-bit (E8 lattice — primary weight codec)
   - GPTQ/AWQ 4-bit (compatibility)
   - AQLM 2-bit (non-critical layers)
   - ANS entropy coding (+10-20% free lossless compression)
   - Mixed precision per tensor in NXF metadata
3. **Safetensors importer** — HuggingFace format → NXF with quantization
4. **Hybrid CPU+GPU pipeline** — CPU entropy-decodes while GPU processes previous layer, GCD concurrent queues

**Milestone:** Run LLaMA 70B QuIP# 3-bit on M4 Max 128 GB. Target: 20%+ tok/s improvement over llama.cpp.

---

### Phase 3: KV Cache + Weight Streaming (Weeks 9-13)
**Goal:** 175B+ models via intelligent caching and streaming.
**Research Paper #4:** "TurboQuant-Paged: Tiered KV Cache Compression for Long-Context Inference on Consumer Hardware"

**Deliverables:**
1. **KV-as-DB with paged attention:**
   - Fixed-size pages (256 tokens × head_dim), UMA pool (`MTLBuffer storageModeShared`)
   - Page table: (layer, head, token_range) → page pointer
   - SSD overflow for evicted pages
2. **TurboQuant KV compression (C++ port):**
   - Port TurboQuant algorithms from `turboquant-mlx/` reference
   - Metal shader: `turbo_quant.metal` for GPU-accelerated rotation + quantize
   - Tiered pipeline: FP16 → 3.5-bit (warm) → 2.5-bit (cool) → evict (cold)
3. **Prefix cache (radix tree):**
   - Longest-prefix match → skip prefill for matched tokens
   - Multi-session sharing, persist to disk for warm restarts
4. **H2O + SnapKV eviction:**
   - Track attention scores, keep heavy-hitters + recent window
   - Async eviction on GCD background queue
5. **Advanced weight streaming:**
   - Predictive prefetch: layers L+1, L+2 while computing L
   - Double-buffering with adaptive prefetch window

**Milestone:** Run 175B model on 96 GB Mac. Demonstrate prefix cache speedup and 32K+ context without OOM.

---

### Phase 4: MoE + Speculative Decoding + 400B (Weeks 14-18)
**Goal:** Hit the 400B-class target.
**Research Paper #5:** "EAGLE-ANE: Speculative Decoding on Apple Neural Engine"

**Deliverables:**
1. **MoE support:**
   - Gate → top-k expert selection → load only active experts
   - DeepSeek-V3 671B: 37B active per token (18x reduction)
   - LRU expert caching + predictive prefetch from gate logits
2. **Speculative decoding on ANE:**
   - Draft model (1-3B) → CoreML → Neural Engine
   - `MLComputeUnits.cpuAndNeuralEngine` frees GPU for verification
   - EAGLE-3 style: 3x throughput, zero quality loss
3. **End-to-end 400B demo:**
   - Dense 405B: QuIP# 3-bit + ANS = ~130 GB on SSD, ~20-25 GB active RAM
   - MoE 671B: 37B active × 4-bit = ~18 GB active, easily fits 48 GB
   - Full benchmarks vs llama.cpp, MLX, ollama

**Milestone:** Run 400B-class model on 48-96 GB Mac with published benchmarks.

---

### Phase 5: Production + Ecosystem (Weeks 19-24)
**Goal:** Production-ready with API and model ecosystem.
**Research Paper #6:** "NEXUS: A Compression-Native Inference Runtime for 400B-Class LLMs on Apple Silicon" (comprehensive system paper)

**Deliverables:**
1. **HTTP API server** — OpenAI-compatible REST + SSE streaming + continuous batching
2. **Model hub CLI** — `nexus pull`, `nexus convert`, `nexus quantize`, `nexus bench`
3. **Benchmarking suite** — MMLU, GSM8K, HumanEval, GPQA + memory/latency profiling via `os_signpost`

---

## Memory Budget (48 GB Mac, 405B Dense Model)

| Component | Budget | Notes |
|-----------|--------|-------|
| OS + system | ~6 GB | macOS overhead |
| Weight buffers (double-buffer) | ~6-9 GB | 2-3 layers streamed from SSD |
| KV hot (FP16) | ~2 GB | Current computation window |
| KV warm (TurboQuant 3.5-bit) | ~6 GB | Recent tokens, 4.6x compressed |
| KV cool (TurboQuant 2.5-bit) | ~4 GB | Older tokens, 6.4x compressed |
| Compute scratch | ~4 GB | Activations, intermediates |
| Draft model (ANE) | ~2 GB | EAGLE-3 speculative decoding |
| Headroom | ~8 GB | OS file cache + safety |
| **Total** | **~38-41 GB** | **Within 48 GB** |

Full model on SSD: 405B × QuIP# 3-bit + ANS ≈ 130 GB, streamed at 5-7 GB/s.

---

## macOS-Specific Technical Decisions

| Linux Concept | NEXUS on macOS |
|--------------|---------------|
| io_uring | GCD `dispatch_io` / kqueue |
| CUDA | Metal Compute Shaders |
| cuBLAS | Accelerate `cblas` + AMX coprocessor |
| TensorRT | CoreML + Neural Engine |
| NVFP4 | TurboQuant (better: 0.997 vs 0.981 KIVI score) |
| posix_fadvise | `fcntl(F_RDADVISE)` |
| 4 KB pages | 16 KB pages (Apple Silicon default) |
| Shared memory copies | UMA zero-copy (`storageModeShared`) |

---

## Research Papers Roadmap

| # | Title | Phase | Key Contribution |
|---|-------|-------|-----------------|
| 1 | NXF: A Streaming-Native Tensor Format | 1 | 16KB-aligned chunked format with multi-codec support |
| 2 | Fused Dequant-GEMM Kernels for Metal | 2 | Metal shader design exploiting UMA zero-copy |
| 3 | Multi-Layer Compression Stack | 2 | QuIP# + AQLM + ANS combined, per-layer adaptive precision |
| 4 | TurboQuant-Paged KV Cache | 3 | Tiered TurboQuant compression + H2O eviction + radix prefix |
| 5 | EAGLE-ANE: Speculative Decoding on Neural Engine | 4 | Draft model on ANE, freeing GPU for verification |
| 6 | NEXUS System Paper | 5 | Comprehensive benchmarks, ablations, full system description |

---

## Dependencies

**C++ (core engine):**
- C++20 (concepts, coroutines)
- Metal SDK (`metal-cpp` headers)
- Apple Accelerate (BLAS, vDSP)
- CoreML framework
- FlatBuffers (NXF manifest)
- cpp-httplib or uWebSockets
- simdjson
- xxHash (checksums)
- FiniteStateEntropy (ANS codec)

**Python (tooling):**
- Already installed in `.venv/`: mlx, mlx-lm, torch, scipy, transformers, turbokv
- TurboQuant references: `turboquant-mlx/`, `turboquant/`, `turboquant-pytorch/`

---

## Verification Plan

1. **Correctness:** Token-by-token match vs llama.cpp (same model/prompt/seed)
2. **Memory:** Peak RSS via `mach_task_info` — must stay under target
3. **Performance:** tok/s, TTFT benchmarks on M2 Max, M3 Max, M4 Max, M3 Ultra
4. **Quality:** MMLU, GSM8K, HumanEval, GPQA — <1% drop vs FP16 baseline
5. **Long context:** 32K, 64K, 128K tokens — verify KV paging holds
6. **Stress:** Concurrent requests, repeated prompts (prefix cache hits)

---

## First Implementation Step

**Phase 1.1 + 1.2 + 1.3:** Project scaffolding + NXF format + UMA memory manager + Research Paper #1

This is the foundation that everything builds on. Once approved, we start writing C++ code.
