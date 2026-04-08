# NEXUS Inference Engine

**A compression-native LLM inference runtime built exclusively for Apple Silicon.**

Run 400B+ parameter models on your Mac. No cloud required.

```
  ╔═══════════════════════════════════════════════════╗
  ║  NEXUS Inference Engine v0.1.0                    ║
  ║  Compression-native LLM runtime for Apple Silicon ║
  ╚═══════════════════════════════════════════════════╝
```

## Why NEXUS?

Every existing inference engine (llama.cpp, MLX, ollama) assumes the model fits in memory. **NEXUS doesn't.** It treats LLM inference as a streaming, caching, and compression problem — never loading the full model, instead streaming weight chunks from SSD while aggressively compressing the KV cache.

### NEXUS vs llama.cpp

| Feature | llama.cpp | NEXUS |
|---------|-----------|-------|
| **Max model on 48GB Mac** | ~70B (Q4) | **405B+** |
| **Architecture** | Load-and-go (entire model in RAM) | **Streaming** (2-3 layers in RAM) |
| **KV Cache** | Flat buffer, no compression | **Paged + TurboQuant** (3.5-bit quality-neutral) |
| **Prefix reuse** | No | **Radix tree** (skip prefill for repeated prompts) |
| **KV eviction** | No (OOM on long context) | **H2O + SnapKV** (keeps only important tokens) |
| **Weight format** | GGUF (single codec) | **NXF** (per-tensor codec, mixed precision) |
| **Compression** | Q4_K_M / Q8_0 | **QuIP# 3-bit + ANS entropy** (~6x vs 4x) |
| **MoE support** | Basic | **Expert LRU cache + predictive prefetch** |
| **Speculative decoding** | No | **EAGLE-3 on Neural Engine** (3x throughput) |
| **UMA exploitation** | Partial | **Full** (storageModeShared zero-copy) |
| **API server** | Basic | **OpenAI-compatible** with SSE streaming |
| **Apple Silicon target** | Generic (cross-platform) | **Purpose-built** (Metal, AMX, ANE) |

### How does a 405B model fit in 48GB?

```
405B × FP16 = 810 GB  (impossible)
405B × QuIP# 3-bit + ANS = ~130 GB on SSD
Active in RAM: 2-3 layers (6 GB) + KV cache (8 GB) + scratch (4 GB) = ~28 GB
Stream the rest from NVMe SSD at 5-7 GB/s
```

## Quick Start

### Prerequisites

- macOS 14+ (Sonoma or later)
- Apple Silicon Mac (M1/M2/M3/M4)
- Xcode Command Line Tools
- CMake 3.25+

```bash
# Install build tools
xcode-select --install
brew install cmake

# Optional: Metal shader compilation
xcodebuild -downloadComponent MetalToolchain
```

### Build

```bash
cd nexus
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

### Download a Model

NEXUS supports GGUF (llama.cpp) and safetensors (HuggingFace) formats. Convert them to NXF for optimal streaming performance.

**Option 1: From GGUF (llama.cpp models)**

```bash
# Download a GGUF model (example: LLaMA 3.1 8B Q4)
# From HuggingFace or any GGUF provider
wget https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# Convert to NXF format
./nexus convert Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf llama-8b.nxf --codec int4
```

**Option 2: From safetensors (HuggingFace)**

```bash
# Download a HuggingFace model
pip install huggingface-hub
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir llama-8b-hf

# Convert to NXF (processes all shards + config.json automatically)
./nexus convert llama-8b-hf/ llama-8b.nxf --codec int4
```

### Run Inference

```bash
# Interactive generation
./nexus run llama-8b.nxf --prompt "Explain quantum computing in simple terms"

# With custom parameters
./nexus run llama-8b.nxf \
  --prompt "Write a Python function to sort a list" \
  --max-tokens 500 \
  --temperature 0.3 \
  --top-p 0.9

# Set memory limit (for large models)
./nexus run llama-405b.nxf --prompt "Hello" --ram-limit 48
```

### Start API Server

```bash
# Start OpenAI-compatible server
./nexus serve llama-8b.nxf --port 8080

# Use with any OpenAI client
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nexus",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Inspect Model

```bash
./nexus info llama-8b.nxf
```

### Run Benchmarks

```bash
./nexus bench llama-8b.nxf --prompt "The quick brown fox" --max-tokens 100
```

## Compatible Model Formats

| Format | Source | Support | Notes |
|--------|--------|---------|-------|
| **GGUF** | llama.cpp | Full | All quantization types (Q4_0, Q4_K_M, Q8_0, F16, etc.) |
| **Safetensors** | HuggingFace | Full | Single file and multi-shard, with config.json parsing |
| **NXF** | NEXUS native | Full | Streaming-optimized, per-tensor codec, 16KB page-aligned |

### Supported Architectures

- LLaMA / LLaMA 2 / LLaMA 3 / LLaMA 3.1
- Mistral / Mixtral (MoE)
- DeepSeek / DeepSeek-V3 (MoE)
- Qwen / Qwen2
- Any transformer-based model in GGUF or safetensors format

### Quantization Codecs

| Codec | Bits | Compression | Quality | Use Case |
|-------|------|-------------|---------|----------|
| FP16 | 16 | 1x | Baseline | Reference / small models |
| INT8 | 8 | 2x | Near-lossless | When quality matters most |
| GPTQ (INT4) | 4 | 4x | <1% loss | General purpose |
| AWQ (INT4) | 4 | 4x | <0.5% loss | Activation-aware, higher quality |
| QuIP# | 3 | 5.3x | <1% loss | **Recommended** for large models |
| QuIP# + ANS | ~2.6 | 6.1x | <1% loss | Maximum compression |
| TurboQuant | 2.5-3.5 | 4.6-6.4x | Quality-neutral | KV cache compression |

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

## Key Technologies

- **NXF Format**: Streaming-native tensor container with 16KB page alignment (Apple Silicon VM page size), per-chunk codecs, MoE routing metadata
- **TurboQuant**: Google Research's near-optimal KV cache quantization (ICLR 2026) — 3.5-bit quality-neutral, 2.5-bit marginal loss
- **Paged KV Cache**: Fixed-size pages with tiered compression (Hot → Warm → Cool → Evict)
- **Metal GPU Compute**: Fused dequant+GEMM, flash attention, all using `storageModeShared` for UMA zero-copy
- **Speculative Decoding**: Draft model on Neural Engine via CoreML, main model verifies on GPU
- **MoE Expert Routing**: LRU expert cache with predictive prefetch from gate logits

## Research Papers

The `research/` directory contains technical papers documenting our novel contributions:

1. **NXF: A Streaming-Native Tensor Format for Memory-Constrained LLM Inference on Apple Silicon**
2. **Fused Dequantization-GEMM Kernels for Apple Metal: Exploiting Unified Memory for LLM Inference**
3. **Combining QuIP#, AWQ, ANS, and TurboQuant: A Multi-Layer Compression Stack for 400B-Class LLM Inference on Consumer Hardware**

## Project Structure

```
nexus/
├── src/
│   ├── core/           # Engine orchestrator, scheduler
│   ├── format/         # NXF format reader/writer
│   ├── memory/         # UMA memory manager, prefetcher
│   ├── compute/        # Metal GPU, Accelerate/AMX, CoreML/ANE, NEON
│   ├── kv/             # Paged KV cache, TurboQuant, prefix cache, eviction
│   ├── quant/          # GPTQ, AWQ, QuIP#, ANS entropy codecs
│   ├── model/          # Transformer, MoE routing, speculative decoding
│   ├── api/            # HTTP server, CLI, benchmarks
│   └── import/         # GGUF and safetensors importers
├── shaders/            # Metal compute shaders (5 shaders, 58KB compiled)
├── research/           # Technical research papers
├── tools/              # Python benchmarking tools
└── tests/              # Unit tests
```

## Building from Source

```bash
git clone https://github.com/JoelHJames1/Nexus-Inference-Engine-.git
cd Nexus-Inference-Engine-/nexus
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)

# Run tests
ctest --output-on-failure

# Install (optional)
sudo make install
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `NEXUS_BUILD_TESTS` | ON | Build unit tests |
| `NEXUS_BUILD_TOOLS` | ON | Build CLI tools |
| `NEXUS_ENABLE_COREML` | ON | Enable CoreML/Neural Engine support |

## Memory Requirements

| Model | llama.cpp (Q4) | NEXUS (QuIP# + streaming) |
|-------|---------------|--------------------------|
| 7B | 4 GB | 4 GB |
| 13B | 8 GB | 6 GB |
| 70B | 40 GB | 12 GB |
| 405B | 200 GB (fails) | **28 GB** |
| 671B MoE | 330 GB (fails) | **33 GB** |

## License

MIT

## Contributing

Contributions welcome. See the research papers in `research/` for architecture details and design rationale.

## Acknowledgments

- [TurboQuant](https://arxiv.org/abs/2504.19874) — Google Research / DeepMind (KV cache quantization)
- [QuIP#](https://arxiv.org/abs/2402.04396) — Cornell (lattice-based weight quantization)
- [FlashAttention](https://arxiv.org/abs/2205.14135) — Dao et al. (efficient attention)
- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention inspiration
- [H2O](https://arxiv.org/abs/2306.14048) — Heavy-Hitter Oracle KV eviction
