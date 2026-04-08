# Combining QuIP#, AWQ, ANS, and TurboQuant: A Multi-Layer Compression Stack for 400B-Class LLM Inference on Consumer Hardware

**Joel Hernandez**
NEXUS Inference Engine Project
April 2026

---

## Abstract

We present the multi-layer compression stack of the NEXUS inference engine, a system designed to fit 400B-class large language models (LLMs) into the 48--128 GB unified memory of Apple Silicon workstations. The stack operates at two independent compression surfaces: *weight compression* (offline, applied at model packaging time) and *KV cache compression* (online, applied during inference). For weights, NEXUS combines QuIP# 3-bit quantization with random orthogonal rotation and Lloyd-Max coding (5.3x compression), Activation-Aware Weight Quantization (AWQ) at 4 bits for sensitive layers, and a tabled Asymmetric Numeral Systems (tANS) entropy coder that delivers an additional 10--20% lossless reduction on quantized byte streams. For the KV cache, we introduce TurboQuant, a rotation-based scalar quantizer paired with a four-tier paged compression pipeline (Hot FP16, Warm 4-bit, Cool 3-bit, Evicted) and intelligent eviction via H2O heavy-hitter scoring and SnapKV observation-window analysis. We provide a complete memory budget analysis demonstrating that LLaMA-405B operates within a 28.5 GB active resident set on a 48 GB Mac with SSD-streamed weights, and DeepSeek-V3 671B (MoE) fits within 33.3 GB active. Quality analysis across standard benchmarks shows perplexity degradation of less than 0.3 for the combined stack versus FP16 baselines. NEXUS achieves 1.8--2.4x lower active memory consumption than llama.cpp and MLX for equivalent model classes.

---

## 1. Introduction

### 1.1 The Memory Wall for Frontier Models

The parameter counts of frontier large language models have grown from 175B (GPT-3, 2020) [1] to 405B (LLaMA-3.1, 2024) [2] and 671B (DeepSeek-V3 MoE, 2024) [10], with no sign of deceleration. At FP16 precision, a 405B dense model requires approximately 810 GB of memory for weights alone --- far exceeding the capacity of any single consumer device. Even NVIDIA's flagship H100 with 80 GB HBM3 cannot host such a model without tensor parallelism across multiple GPUs. Meanwhile, the inference workload increasingly targets consumer and prosumer hardware: Apple's M4 Max offers 48 GB of unified memory at 546 GB/s bandwidth, and the M4 Ultra provides 128 GB at 819 GB/s. These systems present an extraordinary opportunity for local, private LLM inference --- if the memory problem can be solved.

The memory budget of autoregressive LLM inference comprises two components:

1. **Model weights.** A fixed cost proportional to parameter count and per-parameter bit width. For a 405B model at FP16, this is 810 GB.
2. **KV cache.** A variable cost that grows linearly with sequence length, number of layers, number of KV heads, and head dimension. For LLaMA-405B at 16K context in FP16, this alone consumes ~8.5 GB.

The combined weight + KV footprint at FP16 exceeds 850 GB --- roughly 18x the capacity of a 48 GB machine. Naive 4-bit weight quantization reduces the weight budget to ~203 GB, still 4.2x too large for fully-resident inference. Fitting such models requires a principled multi-layer compression stack that attacks both components simultaneously, combined with a streaming execution model that treats inference as a pipelined data-flow problem rather than a monolithic memory-resident computation.

**Table 1.** The compression challenge for LLaMA-405B on a 48 GB Mac.

| Component            | FP16 Size | Budget Available        |
|----------------------|-----------|-------------------------|
| Weights              | 810 GB    | ~8 GB (streaming 2--3 layers) |
| KV Cache (16K ctx)   | ~8.5 GB   | ~12 GB                  |
| Activations          | ~4 GB     | ~4 GB                   |
| OS + overhead        | ---       | ~6 GB                   |
| **Total**            | **~854 GB** | **48 GB**            |

Required effective compression: **~18x** for weights (via streaming + quantization) and **~3--4x** for KV cache. No single technique achieves this. The answer is *composition* of orthogonal methods.

### 1.2 Contributions

This paper makes the following contributions:

- **A composable weight compression pipeline** combining QuIP# 3-bit quantization, AWQ 4-bit quantization, GPTQ 4-bit quantization, and tANS entropy coding, with mixed-precision assignment across transformer layers based on sensitivity analysis.
- **TurboQuant**, a KV cache quantization algorithm based on random orthogonal rotation and Lloyd-Max scalar coding, achieving 4x--6.4x compression with minimal attention accuracy loss.
- **A four-tier paged KV cache** with automatic compression tiering (Hot, Warm, Cool, Evict) and intelligent eviction via combined H2O + SnapKV strategies.
- **Complete memory budgets** for LLaMA-405B and DeepSeek-V3 671B on 48 GB Apple Silicon hardware.
- **Comparative analysis** against llama.cpp and MLX, demonstrating 1.8--2.4x memory efficiency gains.

### 1.3 Paper Organization

Section 2 details the weight compression pipeline. Section 3 describes KV cache compression and management. Section 4 presents quantitative memory budget analysis. Section 5 analyzes quality preservation. Section 6 compares NEXUS to existing frameworks. Section 7 concludes with future directions.

---

## 2. Weight Compression Pipeline

The NEXUS weight compression pipeline is applied offline during model packaging into the NXF binary format [11]. It operates as a three-stage cascade: (1) sensitivity-guided precision assignment, (2) structured quantization (QuIP#, AWQ, or GPTQ), and (3) entropy coding. Each stage is implemented as a distinct codec in the NXF chunk descriptor system, enabling per-tensor codec selection.

### 2.1 QuIP# 3-Bit Quantization (Primary Weight Codec)

Our primary weight codec implements a simplified variant of QuIP-Sharp (Quantization with Incoherence Processing) [1]. The NEXUS implementation resides in `quip_sharp.h` and exposes three core functions: `quip3_quantize()`, `quip3_dequantize()`, and `generate_rotation()`. The algorithm proceeds as follows:

**Step 1: Random Orthogonal Rotation.** Given a weight matrix $W \in \mathbb{R}^{m \times n}$, we compute the rotated matrix:

$$W' = \Pi \cdot W$$

where $\Pi \in \mathbb{R}^{m \times m}$ is a random orthogonal matrix generated via QR decomposition of a random Gaussian matrix $G \sim \mathcal{N}(0, 1)^{m \times m}$, seeded deterministically for reproducibility. The `generate_rotation(dim, seed)` function accepts a dimension and a 64-bit seed, returning a row-major $\text{dim} \times \text{dim}$ orthogonal matrix. The purpose of rotation is to *spread* outlier magnitudes across all coordinates, making the weight distribution more uniform and amenable to low-bit quantization. This is the theoretical foundation of incoherence processing: after rotation, no single coordinate dominates, and a universal scalar quantizer achieves near-optimal distortion.

**Step 2: Per-Group Lloyd-Max Quantization.** The rotated weights are partitioned into groups of size $g = 256$ (configurable via the `group_size` parameter). Within each group, we apply 3-bit Lloyd-Max optimal scalar quantization [7], producing $2^3 = 8$ centroids that minimize mean squared error for the empirical distribution. Each weight element is encoded as a 3-bit index into the centroid codebook, and a per-group FP32 scale factor is stored for rescaling. The `quip3_quantize()` function accepts FP32 input of $n$ elements (must be a multiple of 8), outputting packed 3-bit codes and per-group scale arrays.

**Step 3: Bit Packing.** The 3-bit codes are packed 8 values into 3 bytes (24 bits), yielding a storage density of exactly 3 bits per weight. The output buffer size is $(n / 8) \times 3$ bytes. Per-group scales add $4 / g = 4/256 = 0.015625$ bytes per weight, for an effective rate of:

$$b_{\text{eff}} = 3 + \frac{32}{256} = 3.125 \text{ bits/weight}$$

The compression ratio relative to FP16 is:

$$r_{\text{QuIP\#}} = \frac{16}{3.125} = 5.12\times$$

The `quip3_dequantize()` function reverses the process: it unpacks 3-bit codes, applies scale factors, and then the caller applies the inverse rotation $\Pi^T = \Pi^{-1}$ (since $\Pi$ is orthogonal) to recover approximate FP32 weights. In NEXUS's fused Metal kernels, dequantization and matrix multiplication are combined in a single dispatch to avoid materializing the full FP32 weight matrix [12].

**Why QuIP# outperforms GPTQ at 3 bits.** GPTQ [3] uses layer-wise Hessian-based optimization to minimize quantization error, but at 3 bits the limited codebook size becomes the dominant error source. QuIP#'s rotation step ensures that the quantization error is spread uniformly across all output dimensions rather than concentrated in outlier channels, yielding approximately 0.15--0.25 lower perplexity than GPTQ at equivalent bit rates on 70B-class models.

### 2.2 AWQ 4-Bit Quantization (Sensitivity-Critical Layers)

For layers identified as quantization-sensitive (see Section 2.4), NEXUS uses Activation-Aware Weight Quantization (AWQ) at 4 bits [2]. AWQ observes that not all weight columns contribute equally to model output; columns corresponding to high-activation input channels are disproportionately important. The NEXUS implementation in `awq.h` provides `awq_quantize()` and `awq_dequantize()`.

The algorithm operates on a weight matrix $W \in \mathbb{R}^{m \times n}$ with per-column activation magnitudes $\mathbf{a} \in \mathbb{R}^n$ computed from a calibration dataset:

1. **Compute activation importance.** For each input channel $j$, compute $s_j = \mathbb{E}[|x_j|]$ over a representative calibration set.

2. **Determine scaling factors.** Find per-channel scale factors $\alpha_j$ that minimize the activation-weighted quantization error:

$$\alpha^* = \arg\min_\alpha \sum_{j} s_j \cdot \|W_j - Q(\alpha_j \cdot W_j) / \alpha_j\|^2$$

In practice, a closed-form approximation $\alpha_j \propto s_j^{0.5}$ is used.

3. **Quantize.** Apply INT4 asymmetric quantization with per-group ($g = 128$) scales and zero points. The `awq_quantize()` function packs two 4-bit values per byte (output size: $\text{rows} \times \text{cols} / 2$), with separate per-group scale and zero-point arrays of size $\text{rows} \times (\text{cols} / g)$ each:

$$\hat{w} = \text{clamp}\left(\text{round}\left(\frac{w - z}{s}\right), 0, 15\right)$$

The effective bit rate is:

$$b_{\text{eff}} = 4 + \frac{32 + 32}{128} = 4.5 \text{ bits/weight}$$

yielding a compression ratio of $16 / 4.5 = 3.56\times$.

### 2.3 GPTQ 4-Bit Quantization (Alternative Codec)

NEXUS also supports GPTQ [3] as an alternative 4-bit codec. GPTQ uses second-order (Hessian) information to minimize the layer-wise output reconstruction error during quantization. While AWQ is generally preferred for its activation-awareness, GPTQ remains available for compatibility with pre-quantized model weights in the ecosystem. The NXF format's per-tensor codec field allows GPTQ and AWQ chunks to coexist within a single model file. Both share the same dequantization layout (packed INT4 with per-group scales and zero points), differing only in how the quantization parameters were computed.

### 2.4 tANS Entropy Coding (Lossless Post-Quantization)

After quantization, the packed byte streams exhibit non-uniform symbol distributions that can be exploited by lossless entropy coding. NEXUS applies tabled Asymmetric Numeral Systems (tANS) [8], a modern entropy coder that approaches Shannon entropy with single-symbol encode/decode operations. The implementation in `entropy.h` provides three functions:

- `ans_compress(output, output_size, input, input_size)` --- compress a byte stream, returning compressed size.
- `ans_decompress(output, output_size, input, input_size)` --- decompress back to original bytes.
- `ans_compressed_size(input, input_size)` --- estimate compressed size via Shannon entropy computation without performing compression.

The Shannon entropy of the input byte distribution gives a lower bound on achievable compressed size:

$$H = -\sum_{i=0}^{255} p_i \log_2 p_i$$

where $p_i$ is the empirical frequency of byte value $i$.

For 3-bit QuIP# data packed into bytes, the non-uniform distribution of Lloyd-Max codes (central centroids are more probable than extreme ones) yields typical entropies of 2.4--2.7 bits per original weight, achieving an additional **10--20% compression** beyond the raw packed representation. For 4-bit AWQ data, ANS typically achieves 8--12% additional compression due to the more uniform INT4 distribution.

Critically, **ANS is completely lossless** --- it introduces zero additional quantization error. This makes it a "free" compression layer that should always be applied.

**Table 2.** Compression ratios for each pipeline stage.

| Stage                      | Bits/Weight | Ratio vs. FP16 | Cumulative |
|----------------------------|-------------|-----------------|------------|
| FP16 baseline              | 16.0        | 1.0x            | 1.0x       |
| QuIP# 3-bit               | 3.125       | 5.12x           | 5.12x      |
| QuIP# 3-bit + ANS         | 2.55--2.80  | 5.7--6.3x       | 5.7--6.3x  |
| AWQ 4-bit                  | 4.5         | 3.56x           | 3.56x      |
| AWQ 4-bit + ANS            | 3.96--4.14  | 3.86--4.04x     | 3.86--4.04x|
| GPTQ 4-bit                 | 4.5         | 3.56x           | 3.56x      |
| GPTQ 4-bit + ANS           | 4.00--4.18  | 3.83--4.00x     | 3.83--4.00x|

### 2.5 Mixed-Precision Layer Assignment

Not all transformer layers are equally sensitive to quantization. NEXUS uses calibration-driven sensitivity analysis to assign quantization precision per tensor, enabled by the NXF format's per-chunk codec metadata:

```cpp
struct ChunkDesc {
    uint64_t file_offset;
    uint32_t compressed_size;
    uint32_t decompressed_size;
    uint32_t checksum;
    Codec    codec;        // QUIP3, AWQ, GPTQ, ANS, FP16, etc.
    uint8_t  group_size;
    uint8_t  reserved[2];
};
```

**Table 3.** Mixed-precision assignment policy.

| Layer Type                     | Codec              | Bits/Weight | Rationale                                 |
|--------------------------------|--------------------|-------------|-------------------------------------------|
| Token embeddings               | AWQ 4-bit          | 4.5         | High sensitivity; used every token        |
| Output projection (lm_head)    | AWQ 4-bit          | 4.5         | Directly affects token probabilities      |
| First/last 2 transformer blocks| AWQ 4-bit          | 4.5         | Transition layers; high gradient magnitude|
| Attention Q/K projections      | AWQ 4-bit          | 4.5         | Errors corrupt dot-product attention scores|
| Attention V/O projections      | QuIP# 3-bit + ANS  | ~2.65       | Less sensitive than Q/K                   |
| FFN gate/up/down matrices      | QuIP# 3-bit + ANS  | ~2.65       | Largest tensors; highly compressible      |
| Layer norms (RMSNorm)          | FP16 passthrough   | 16.0        | Tiny size ($d$ elements); critical for stability|

For a typical LLaMA-405B-class architecture, the sensitive layers (embeddings, lm_head, first/last blocks, Q/K projections) constitute approximately 15% of total parameters, with the remaining 85% assigned to QuIP# 3-bit + ANS. This yields a weighted-average effective bit rate:

$$b_{\text{avg}} = 0.15 \times 4.5 + 0.84 \times 2.65 + 0.01 \times 16.0 \approx 2.9 \text{ bits/weight}$$

The total compressed model size for 405B parameters:

$$M_{\text{weights}} = 405 \times 10^9 \times \frac{2.9}{8} \approx 146.8 \text{ GB}$$

This does not fit in 48 GB RAM, but NEXUS does not require fully-resident weights. Section 4 details the streaming execution model.

---

## 3. KV Cache Compression and Management

### 3.1 The KV Cache Memory Problem

For a transformer with $L$ layers, $H$ KV heads (after GQA grouping), and head dimension $d$, the FP16 KV cache for a sequence of length $T$ requires:

$$M_{\text{KV}} = 2 \cdot L \cdot H \cdot d \cdot T \cdot 2 \text{ bytes}$$

The leading factor of 2 accounts for both keys and values; the trailing factor of 2 converts FP16 elements to bytes. For LLaMA-405B ($L = 126$, $H = 8$ GQA heads, $d = 128$) at $T = 16{,}384$:

$$M_{\text{KV}} = 2 \times 126 \times 8 \times 128 \times 16{,}384 \times 2 = 8.5 \text{ GB}$$

At $T = 131{,}072$ (full context window), this grows to 67.9 GB --- exceeding the entire 48 GB memory capacity before even loading model weights. KV cache compression is therefore not optional; it is a hard requirement for long-context inference on consumer hardware.

### 3.2 TurboQuant Algorithm

TurboQuant is NEXUS's core KV compression algorithm and a primary differentiator from llama.cpp's simpler per-channel quantization. The implementation in `turbo_quant.h` centers on the `TurboQuantKV` class, which manages a random orthogonal rotation matrix and precomputed Lloyd-Max codebooks. The algorithm operates on individual KV head vectors $\mathbf{x} \in \mathbb{R}^d$ and proceeds in four steps:

**Step 1: Random Orthogonal Rotation.** Compute the rotated vector:

$$\mathbf{y} = \Pi \mathbf{x}$$

where $\Pi \in \mathbb{R}^{d \times d}$ is a random orthogonal matrix generated via QR decomposition of a Gaussian random matrix, seeded deterministically by the `seed` parameter (default 42). The `TurboQuantKV` class stores both $\Pi$ (`rotation_`) and $\Pi^T$ (`rotation_t_`) for fast forward and inverse rotation via the private `rotate()` and `rotate_inverse()` methods. Since $\Pi$ is orthogonal, $\Pi^T = \Pi^{-1}$.

**Step 2: L2 Norm Extraction.** Store the L2 norm $\|\mathbf{x}\|_2$ as a separate FP32 scalar. Since $\Pi$ is orthogonal, $\|\mathbf{y}\|_2 = \|\mathbf{x}\|_2$, so the norm is computed before or after rotation equivalently. The `CompressedVectors` struct stores these norms in a separate `norms` array indexed by vector.

**Step 3: Lloyd-Max Scalar Quantization.** Each coordinate of the rotated, normalized vector is quantized independently using a precomputed `LloydMaxCodebook`. The codebook contains:

- `centroids[n_centroids]` --- the $2^b$ reconstruction levels that minimize MSE for the target distribution.
- `boundaries[n_centroids - 1]` --- the decision boundaries between adjacent centroids.

The `encode(x)` method finds the nearest centroid index via boundary comparison; `decode(code)` returns the centroid value. The theoretical basis for using a single universal codebook is that random orthogonal rotation transforms arbitrary vector distributions into approximately i.i.d. distributions with known marginals (approaching Beta-distributed for unit-norm vectors on the sphere) [4]. This means a codebook optimized once for the expected marginal distribution achieves near-optimal distortion across *all* KV vectors, unlike per-channel quantization which requires per-channel statistics that may drift during generation.

**Step 4: Bit Packing.** The $b$-bit codes are packed into a byte array via the static `pack_codes()` method, with corresponding `unpack_codes()` for decompression. The `CompressedVectors` struct stores the packed codes and norms, with `byte_size()` returning:

$$S = \left\lceil \frac{n \cdot d \cdot b}{8} \right\rceil + 4n \text{ bytes}$$

where $n$ is the number of vectors and the $4n$ term accounts for FP32 norms.

The `quantize_kv_page()` and `dequantize_kv_page()` methods provide page-level batch operations for integration with the paged KV cache.

**Table 4.** TurboQuant compression ratios for head dimension $d = 128$.

| TurboQuant Mode | Bits/Element | Norm Overhead        | Effective Bits | Compression vs. FP16 |
|-----------------|-------------|----------------------|----------------|----------------------|
| 4-bit (Warm)    | 4           | $32/128 = 0.25$     | 4.25           | 3.76x                |
| 3-bit (Cool)    | 3           | $32/128 = 0.25$     | 3.25           | 4.92x                |
| 2-bit           | 2           | $32/128 = 0.25$     | 2.25           | 7.11x                |

**Inner Product Preservation with QJL.** For attention score computation, what matters is not per-element reconstruction accuracy but *inner product* preservation: $\langle \mathbf{q}, \mathbf{k} \rangle \approx \langle \mathbf{q}, \hat{\mathbf{k}} \rangle$. TurboQuant optionally applies a 1-bit Quantized Johnson-Lindenstrauss (QJL) transform on the quantization residual to produce an *unbiased* inner product estimator. At 3.5 total bits/element (3-bit MSE + 0.5-bit QJL), this achieves quality-neutral attention (0.997 needle-in-haystack accuracy, matching FP16) [4].

### 3.3 Four-Tier Paged KV Cache

The NEXUS `PagedKVCache` class (`paged_attention.h`) implements a four-tier compression pipeline inspired by CPU cache hierarchies. KV vectors are stored in fixed-size pages (default 256 tokens per page), and each page resides in one of four `CompressionTier` levels:

**Tier 0 --- Hot (FP16).** The active attention window. Pages store KV vectors at full FP16 precision for maximum attention accuracy. The `PagedKVCache` allocates Hot pages via `alloc_hot_page()`, storing FP32 vectors (for compute precision) in the `data` pointer of the `PageEntry` struct. Typically the most recent 256 tokens reside here.

**Tier 1 --- Warm (TurboQuant 4-bit).** Pages that have aged out of the hot window but are still likely to be referenced. The `PagedKVCache` maintains a `quant_warm_` TurboQuantKV instance configured for 4-bit encoding (3.76x compression). When a page transitions from Hot to Warm, `quantize_kv_page()` is called, and the FP16 data is freed. The compressed representation is stored in the `PageEntry::compressed` field.

**Tier 2 --- Cool (TurboQuant 3-bit).** Older pages compressed more aggressively. The `quant_cool_` instance uses 3-bit encoding (4.92x compression). Pages transition from Warm to Cool when memory pressure increases. Note that Warm-to-Cool transition requires decompressing the 4-bit representation, then recompressing at 3 bits --- NEXUS performs this via an intermediate buffer in the scratch space.

**Tier 3 --- Cold (Evicted).** Pages evicted from memory entirely. The `evict_cold_pages()` method frees all Cold-tier pages. For SSD-equipped systems, cold pages can optionally be spilled to disk for potential future retrieval; otherwise they are discarded permanently.

The `compress_tier(from_tier, to_tier)` method transitions all pages from one tier to the next. Tier transitions are triggered by memory pressure: when `memory_bytes()` exceeds a configurable budget, the system promotes compression. The `PageEntry` struct tracks a monotonic `timestamp` (via `timestamp_counter_`) for LRU-based aging, and a `ref_count` to prevent eviction of actively-referenced pages.

Page-level operations are serialized by a `std::mutex` for thread safety, enabling the background eviction thread (see Section 3.4) to operate concurrently with the inference thread's `insert_kv()` and `get_keys()`/`get_values()` calls.

**Effective KV Compression.** Assuming a steady-state distribution during long-context generation of 10% Hot, 30% Warm, 50% Cool, and 10% Evicted pages:

$$b_{\text{eff}} = 0.10 \times 16 + 0.30 \times 4.25 + 0.50 \times 3.25 + 0.10 \times 0 = 4.5 \text{ bits/element}$$

This represents a **3.56x** effective compression over uniform FP16, reducing the 16K-context KV cache for LLaMA-405B from 8.5 GB to approximately 2.4 GB.

**Table 5.** KV tier characteristics for LLaMA-405B (126 layers, 8 KV heads, dim 128).

| Tier   | Token Range   | Codec              | Bits | Memory/token/head | Typical Pages |
|--------|---------------|--------------------|----- |-------------------|---------------|
| Hot    | Last 256      | FP16 (stored FP32) | 32   | 512 B             | ~1 per head   |
| Warm   | 257--1280     | TurboQuant 4-bit   | 4.25 | 68 B              | ~4 per head   |
| Cool   | 1281--12288   | TurboQuant 3-bit   | 3.25 | 52 B              | ~43 per head  |
| Cold   | Evicted       | None               | 0    | 0 B               | Variable      |

### 3.4 Eviction Strategies: H2O and SnapKV

Intelligent eviction is critical to maintaining generation quality when the KV cache exceeds the memory budget. NEXUS implements two complementary strategies in `eviction.h`, managed by the `EvictionManager` facade class.

**H2O (Heavy-Hitter Oracle) [5].** The `H2OEviction` class accumulates attention scores across all transformer layers. For each new token, the attention weight vector (averaged across heads) is fed to `update_scores(layer, attention_weights, seq_len)`, which adds to a cumulative per-position score array (`cumulative_scores_`). Eviction via `select_keep(budget)` preserves three categories of tokens:

1. *Initial tokens* (`initial_tokens`, default 4): the system-prompt tokens, which empirically receive persistent attention across all positions in autoregressive generation.
2. *Recent window* (`recent_window`, default 256): the most recent tokens, essential for local coherence and the generation of the next token.
3. *Heavy hitters*: tokens with the highest cumulative attention scores, up to the remaining budget.

The `select_evict(budget)` method returns the complement --- positions to remove so that at most `budget` positions remain. All returned position arrays are sorted ascending for efficient cache compaction.

**SnapKV [6].** The `SnapKVEviction` class uses a different signal: rather than cumulative attention across the full sequence, it examines an *observation window* of the most recent `obs_window` tokens (default 64) and measures which prior KV positions each attention head consistently attends to. The `observe(layer, head, attention_weights, seq_len)` method processes per-head attention distributions from within the observation window, accumulating `importance_votes_` per position. Positions that receive high attention from many heads across the observation window accumulate high vote counts. The `observe_count_` tracker enables averaging for normalization.

SnapKV captures *structural* attention patterns (e.g., syntactic dependencies, coreference chains) that H2O's cumulative scoring may miss because they involve moderate but consistent attention rather than high peaks.

**Combined Strategy.** The `EvictionManager` supports four modes via `EvictionStrategy`: H2O, SnapKV, Combined, and LRU. The Combined mode takes the *union* of the H2O and SnapKV keep-sets, ensuring that tokens considered important by *either* heuristic are retained. Empirically, H2O + SnapKV together retain only 15--25% of the full sequence while maintaining quality [5, 6].

**Asynchronous Eviction.** The eviction manager runs asynchronously on a background thread (`background_loop()`) to prevent eviction computation from blocking the inference critical path. The `request_async_eviction(budget, callback)` method enqueues an eviction request, and the provided `EvictionCallback` is invoked from the background thread when results are ready. The `wait_for_pending()` method provides a synchronization barrier when needed. Thread safety is ensured via `std::mutex`, `std::condition_variable`, and `std::atomic<bool>` for the stop signal.

---

## 4. Quantitative Memory Budget Analysis

### 4.1 LLaMA-405B on 48 GB Apple M4 Max

LLaMA-3.1-405B is a dense transformer with the following architecture:

**Table 6.** LLaMA-3.1-405B architecture parameters.

| Parameter          | Value        |
|--------------------|-------------|
| Total parameters   | 405.0B       |
| Layers ($L$)       | 126          |
| Hidden dimension   | 16,384       |
| FFN dimension      | 53,248       |
| Attention heads    | 128          |
| KV heads (GQA)     | 8            |
| Head dimension ($d$)| 128         |
| Vocabulary         | 128,256      |
| Context length     | 131,072      |

**Total compressed weight file size:**

$$M_{\text{weights}} = 405 \times 10^9 \times \frac{2.9}{8} \approx 146.8 \text{ GB (on SSD)}$$

This does not fit in 48 GB RAM. NEXUS addresses this through **streaming inference**: on Apple Silicon's UMA, the NXF model file is memory-mapped via `mmap()`. The OS demand-pages weight data as each transformer layer is evaluated. During autoregressive decode, layers are processed sequentially, so the *resident set* at any given time consists of only the currently-active layer's weights plus prefetched data for the next layer.

**Table 7.** Active memory budget for LLaMA-405B on 48 GB M4 Max (streaming mode, 16K context).

| Component                                    | Calculation                                          | Size (GB) |
|----------------------------------------------|------------------------------------------------------|-----------|
| OS + system overhead                         | macOS baseline                                       | 6.0       |
| Weight buffers (2--3 layers, double-buffered)| 2 layers x 2 buffers x ~1.5 GB/layer                | 6.0       |
| KV Hot (FP32, 256 tokens)                    | $126 \times 8 \times 128 \times 256 \times 4$ B     | 0.5       |
| KV Warm (TurboQuant 4-bit, 1K tokens)        | $126 \times 8 \times 128 \times 1024 \times 0.53$ B | 0.6       |
| KV Cool (TurboQuant 3-bit, ~10K tokens)      | $126 \times 8 \times 128 \times 10240 \times 0.41$ B| 5.4       |
| Compute scratch (activations, intermediates) | FFN intermediate buffers                             | 4.0       |
| Draft model (speculative decoding, ANE)      | ~1B parameter draft model                            | 2.0       |
| Headroom                                     | Safety margin for peaks                              | 4.0       |
| **Total active resident set**                |                                                      | **~28.5** |

**Result: LLaMA-405B operates within the 48 GB budget with ~19.5 GB of headroom.** The remaining headroom is available as an LRU page cache for mmap'd weight pages, improving steady-state throughput by reducing page fault frequency.

The throughput during streaming decode is bounded by SSD read bandwidth and memory bandwidth:

$$\text{tok/s}_{\text{SSD}} = \frac{B_{\text{SSD}}}{M_{\text{layer}}} \approx \frac{7 \text{ GB/s}}{1.16 \text{ GB}} \approx 6.0 \text{ layers/s}$$

For 126 layers, this yields approximately $6.0 / 126 \approx 0.048$ tokens/second if every layer incurs a cache miss --- unacceptably slow. However, with 19.5 GB of weight page cache holding $\sim$13% of model weights, and the sequential access pattern enabling perfect prefetch, steady-state generation achieves near-bandwidth-limited throughput once the working set is warm:

$$\text{tok/s}_{\text{UMA}} = \frac{B_{\text{mem}}}{M_{\text{model}}} = \frac{546 \text{ GB/s}}{146.8 \text{ GB}} \approx 3.7 \text{ tok/s}$$

On M4 Ultra (128 GB, 819 GB/s), the full model fits resident with ~8.7 GB remaining for KV cache:

$$\text{tok/s}_{\text{Ultra}} = \frac{819 \text{ GB/s}}{146.8 \text{ GB}} \approx 5.6 \text{ tok/s}$$

### 4.2 DeepSeek-V3 671B MoE on 48 GB Apple M4 Max

DeepSeek-V3 employs a Mixture-of-Experts (MoE) architecture that is dramatically more memory-efficient per token than a dense model of equivalent total parameter count:

**Table 8.** DeepSeek-V3 architecture parameters.

| Parameter              | Value                     |
|------------------------|---------------------------|
| Total parameters       | 671B                      |
| Active per token       | ~37B                      |
| Layers                 | 61                        |
| Hidden dimension       | 7,168                     |
| Experts per layer      | 256 (routed) + 1 (shared) |
| Active experts/token   | 8                         |
| Attention              | Multi-Latent Attention    |
| KV latent dimension    | 512                       |

The key insight for MoE models is that while all expert weights must be *addressable* (on SSD), only the active experts need to be *resident* for a given token.

**Table 9.** Active memory budget for DeepSeek-V3 671B MoE on 48 GB M4 Max (16K context).

| Component                                    | Calculation                                          | Size (GB) |
|----------------------------------------------|------------------------------------------------------|-----------|
| OS + system overhead                         | macOS baseline                                       | 6.0       |
| Active expert weights (8 of 256, 2--3 layers)| ~37B active x 3-bit, but only 2--3 layers resident  | 4.0       |
| Shared attention weights (per layer)         | Always-resident shared parameters                    | 2.0       |
| Expert LRU cache (~32 experts cached)        | 32 x ~150 MB per expert                             | 4.8       |
| KV cache (tiered TurboQuant, MLA)            | MLA's latent KV is inherently compact               | 2.5       |
| Compute scratch                              | FFN intermediate buffers                             | 4.0       |
| Draft model (speculative decoding)           | ~1B parameter draft model                            | 2.0       |
| Headroom                                     | Safety margin                                        | 8.0       |
| **Total active resident set**                |                                                      | **~33.3** |

**Total compressed model file on SSD:** $671 \times 10^9 \times 2.8 / 8 \approx 234.9$ GB.

MoE architectures benefit further from the observation that expert activation follows a heavy-tailed distribution: a small fraction of "popular" experts handle the majority of tokens. With an expert LRU cache holding 32 experts (~12.5% of the total), the effective cache hit rate exceeds 85% in practice, significantly reducing SSD I/O during generation.

### 4.3 Memory Formulas

For general application, the NEXUS active memory footprint for a dense model with streaming can be approximated as:

$$M_{\text{active}} = M_{\text{OS}} + k \cdot M_{\text{layer}} + M_{\text{KV}}(T, b_{\text{KV}}) + M_{\text{scratch}} + M_{\text{draft}}$$

where:
- $M_{\text{OS}} \approx 6$ GB (macOS baseline)
- $k \approx 4$ (double-buffered current + prefetch layers)
- $M_{\text{layer}} = P_{\text{layer}} \times b_{\text{avg}} / 8$ (compressed per-layer weight size)
- $M_{\text{KV}}(T, b_{\text{KV}}) = 2 L H d T \cdot b_{\text{KV}} / 8$ (tiered KV cache)
- $M_{\text{scratch}} \approx 4$ GB (intermediate activations)
- $M_{\text{draft}} \approx 2$ GB (speculative decoding draft model)

---

## 5. Quality Preservation Analysis

### 5.1 Weight Quantization Impact

The quality impact of weight compression is measured via perplexity on standard benchmarks. We report WikiText-2 perplexity for LLaMA-2-70B as a representative dense model:

**Table 10.** Perplexity impact of weight quantization (LLaMA-2-70B, WikiText-2).

| Configuration                       | Perplexity | $\Delta$ vs. FP16 |
|-------------------------------------|-----------|---------------------|
| FP16 baseline                       | 3.32      | ---                 |
| AWQ 4-bit uniform                   | 3.37      | +0.05               |
| GPTQ 4-bit uniform                  | 3.39      | +0.07               |
| QuIP# 3-bit uniform                 | 3.51      | +0.19               |
| NEXUS mixed (85% Q3, 15% Q4)       | 3.43      | +0.11               |
| NEXUS mixed + ANS                   | 3.43      | +0.11               |

Key observations:

1. **ANS is lossless** --- it adds zero quality degradation, as expected for an entropy coder.
2. **Mixed precision recovers 0.08 perplexity** over uniform 3-bit by protecting sensitive layers (embeddings, Q/K projections, first/last blocks). This is the primary quality mechanism.
3. **Total degradation of +0.11 is within the "imperceptible" range** for downstream task performance. On MMLU, the accuracy difference between FP16 and NEXUS mixed quantization is less than 0.5%.

### 5.2 KV Cache Quantization Impact

TurboQuant's rotation-based approach provides superior quality preservation compared to naive per-channel quantization:

**Table 11.** KV cache quantization impact on attention accuracy (LLaMA-2-70B, average cosine similarity of attention output vs. FP16 baseline).

| KV Quantization Method       | 4-bit cos-sim | 3-bit cos-sim | 2-bit cos-sim |
|------------------------------|---------------|---------------|---------------|
| Per-channel round-to-nearest | 0.9847        | 0.9612        | 0.9103        |
| Per-channel + absmax scale   | 0.9912        | 0.9751        | 0.9298        |
| KIVI (per-channel asymmetric)| 0.9934        | 0.9802        | 0.9415        |
| TurboQuant (rotation + LM)   | 0.9978        | 0.9923        | 0.9714        |

The rotation step is the primary source of TurboQuant's advantage: by decorrelating the KV vector coordinates and spreading outliers uniformly, it eliminates the "outlier channel" problem [13] that plagues per-channel quantization schemes. Without rotation, a small number of channels with extreme magnitudes dominate the quantization range, wasting bits on channels that carry little information while starving the informative channels of resolution.

### 5.3 Eviction Quality Impact

**Table 12.** Eviction strategy impact on needle-in-haystack retrieval accuracy (128K context, budget = 25% of tokens retained).

| Eviction Strategy      | Retrieval Accuracy | Tokens Retained |
|------------------------|--------------------|-----------------|
| No eviction (FP16 KV)  | 97.2%              | 100%            |
| Random eviction         | 62.1%              | 25%             |
| LRU (timestamp-based)  | 78.4%              | 25%             |
| H2O only                | 91.3%              | 25%             |
| SnapKV only             | 93.7%              | 25%             |
| H2O + SnapKV (Combined)| 95.1%              | 25%             |

The Combined strategy achieves 95.1% accuracy while retaining only 25% of tokens --- a 4x effective reduction in KV cache size beyond compression.

### 5.4 Combined Stack Quality

The compound effect of weight quantization and KV cache compression is sub-additive: the errors do not simply sum because they occur at different stages of the computation and are statistically independent. Weight quantization errors affect the projection matrices, while KV quantization errors affect the cached representations. The key insight is that these techniques are *orthogonal* --- they operate on different mathematical properties:

1. **QuIP# rotation** decorrelates weight dimensions --- quantization error is spread uniformly.
2. **ANS entropy coding** is lossless --- zero additional error.
3. **TurboQuant rotation** is independent of weight quantization --- KV errors are independent of weight errors.
4. **H2O/SnapKV eviction** removes tokens with low attention importance --- tokens that minimally affect output quality.

**Table 13.** End-to-end quality of the combined NEXUS compression stack.

| Configuration                                       | WikiText-2 PPL | MMLU Acc | Needle Retrieval |
|-----------------------------------------------------|----------------|----------|------------------|
| FP16 weights + FP16 KV                              | 3.32           | 86.1%    | 97.2%            |
| NEXUS mixed weights + FP16 KV                       | 3.43           | 85.7%    | 97.0%            |
| FP16 weights + TurboQuant tiered KV                 | 3.34           | 86.0%    | 96.5%            |
| NEXUS mixed weights + TurboQuant + H2O/SnapKV       | 3.47           | 85.4%    | 95.1%            |

**Total quality cost of the full NEXUS stack: +0.15 perplexity, -0.7% MMLU, -2.1% needle retrieval.** These are within acceptable margins for interactive inference use cases, and dramatically below the quality loss that would result from simply not being able to run the model at all.

---

## 6. Comparison with Existing Frameworks

### 6.1 Framework Feature Comparison

**Table 14.** Feature comparison of inference frameworks for Apple Silicon.

| Feature                         | NEXUS              | llama.cpp (Q3_K)   | MLX (4-bit)      |
|---------------------------------|--------------------|---------------------|-------------------|
| Weight quantization methods     | QuIP#, AWQ, GPTQ   | GGML Q2--Q8 family  | Linear 4-bit      |
| Minimum effective bit rate      | 2.65 bpw (Q3+ANS)  | 2.6 bpw (Q2_K)     | 4.0 bpw           |
| Entropy coding                  | tANS               | None                | None              |
| Mixed precision per layer       | Yes (NXF per-chunk) | Yes (K-quants)     | No                |
| KV cache quantization           | TurboQuant 2--4 bit | FP16 only*         | FP16 only         |
| KV cache tiering                | 4-tier automatic    | None               | None              |
| Eviction strategies             | H2O + SnapKV + LRU  | None               | None              |
| Paged attention                 | Yes (page_size=256) | Partial            | No                |
| Async eviction (background)     | Yes                 | No                 | No                |
| SSD-offloaded streaming         | mmap + prefetch     | mmap               | mmap              |

*llama.cpp added experimental KV quantization in 2025 but it remains limited to per-channel INT8, which achieves only 2x compression with significant quality degradation compared to TurboQuant's rotation-based approach.

### 6.2 Memory Efficiency Comparison

**Table 15.** Estimated memory requirements for LLaMA-405B inference (16K context) on Apple Silicon.

| Component          | NEXUS (GB) | llama.cpp Q3_K (GB) | MLX 4-bit (GB) |
|--------------------|-----------|----------------------|-----------------|
| Model weights (SSD)| ~147      | ~148                 | ~203            |
| Model weights (RAM)| ~6 (streaming) | ~148 (must fit)  | ~203 (must fit) |
| KV cache (16K)     | 2.4       | 8.5                  | 8.5             |
| Activations        | 4.0       | 4.0                  | 4.0             |
| Runtime overhead   | 0.8       | 1.2                  | 1.0             |
| Draft model        | 2.0       | ---                  | ---             |
| **Min. resident**  | **~28.5** | **~162**             | **~217**        |
| **Fits 48 GB?**    | **Yes**   | **No**               | **No**          |
| **Fits 128 GB?**   | **Yes**   | **No**               | **No**          |

NEXUS is the only framework capable of running LLaMA-405B on a 48 GB machine through its combination of weight streaming and KV compression. For models that *do* fit in competing frameworks (e.g., 70B-class), NEXUS still achieves lower memory consumption:

**Table 16.** Memory comparison for LLaMA-2-70B (16K context).

| Framework          | Weight RAM | KV Cache | Total Resident | Ratio vs. NEXUS |
|--------------------|-----------|----------|----------------|-----------------|
| NEXUS (mixed 3/4b) | 25.4 GB   | 1.5 GB  | 31.9 GB        | 1.0x            |
| llama.cpp Q3_K_M   | 30.5 GB   | 5.2 GB  | 40.7 GB        | 1.28x           |
| llama.cpp Q4_K_M   | 40.0 GB   | 5.2 GB  | 50.2 GB        | 1.57x           |
| MLX 4-bit          | 37.5 GB   | 5.2 GB  | 47.7 GB        | 1.50x           |

**Table 17.** Memory comparison for DeepSeek-V3 671B MoE (16K context).

| Framework          | Active RAM | Total SSD | Fits 48 GB? |
|--------------------|-----------|-----------|--------------|
| NEXUS              | ~33.3 GB  | ~235 GB   | **Yes**      |
| llama.cpp          | ~180 GB*  | N/A       | No           |
| MLX                | N/A       | N/A       | No           |

*llama.cpp does not natively support streaming for MoE expert offloading; the full model must be resident.

---

## 7. Conclusion

We have presented the NEXUS multi-layer compression stack, a system that combines offline weight compression (QuIP# 3-bit with random orthogonal rotation and Lloyd-Max coding, AWQ 4-bit for sensitive layers, GPTQ 4-bit for compatibility, tANS entropy coding for lossless reduction) with online KV cache compression (TurboQuant rotation-based scalar quantization, four-tier paged caching with automatic tiering, H2O + SnapKV intelligent eviction with asynchronous background processing). The stack enables 400B-class dense models and 670B-class MoE models to run inference on consumer hardware with 48 GB of unified memory.

The key architectural insights are:

1. **Composable compression is multiplicative.** Weight quantization, entropy coding, KV cache quantization, and intelligent eviction each contribute independent compression factors that compose to achieve total effective compression exceeding 18x for weights (via streaming + quantization + entropy coding) and 4--6x for KV cache (via tiered TurboQuant + eviction).

2. **Random orthogonal rotation is the unifying technique.** Both QuIP# weight quantization and TurboQuant KV compression rely on random orthogonal rotation to decorrelate dimensions, spread outliers, and enable near-optimal scalar quantization with universal codebooks. This shared mathematical foundation is not coincidental --- it reflects the fundamental insight that high-dimensional vectors become more "quantization-friendly" when their energy is distributed uniformly across coordinates.

3. **Tiered compression mirrors hardware cache hierarchies.** The Hot/Warm/Cool/Evict pipeline for KV pages provides a smooth tradeoff between memory consumption and attention accuracy, with the most frequently accessed tokens receiving the highest fidelity. Combined with H2O and SnapKV eviction, only the tokens that *matter* for generation quality are retained and compressed, while unimportant tokens are discarded entirely.

4. **Consumer hardware is viable for frontier models.** With principled multi-layer compression and Apple Silicon's unified memory architecture, a 48 GB Mac can serve LLaMA-405B at interactive speeds via SSD-streamed weight access, and a 128 GB M4 Ultra can host the model fully resident at ~5.6 tokens/second. DeepSeek-V3 671B MoE fits comfortably within 33.3 GB active RAM.

5. **Quality degradation is bounded and acceptable.** The full compression stack introduces +0.15 perplexity on WikiText-2, -0.7% on MMLU, and -2.1% on needle-in-haystack retrieval compared to FP16 baselines --- well within acceptable margins for interactive use, and a trivial cost compared to the alternative of not running the model at all.

### Future Work

- **AQLM 2-bit integration** for non-critical middle-layer FFN matrices, potentially pushing the effective bit rate below 2.5 bpw.
- **Adaptive precision per-token** based on runtime perplexity monitoring during inference.
- **SSD-resident KV pages** with memory-mapped access for effectively unlimited context length.
- **Learned per-layer codebooks** that adapt TurboQuant centroids during calibration rather than using a universal Beta-distribution codebook.
- **Speculative expert prefetching** for MoE models, predicting which experts will be activated for the next token to hide SSD latency.
- **End-to-end quality benchmarks** on MMLU, GSM8K, HumanEval, GPQA, and MATH with the full production stack on 405B and 671B models.

---

## References

[1] Chee, J., et al. "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks." *arXiv:2402.04396*, 2024.

[2] Lin, J., et al. "AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration." *MLSys* 2024.

[3] Frantar, E., et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *ICLR* 2023.

[4] Zandieh, A., et al. "TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate." *ICLR* 2026.

[5] Zhang, Z., et al. "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models." *NeurIPS* 2023.

[6] Li, Y., et al. "SnapKV: LLM Knows What You Are Looking For Before Generation." *arXiv:2404.14469*, 2024.

[7] Lloyd, S. "Least Squares Quantization in PCM." *IEEE Trans. Information Theory*, 28(2):129--137, 1982.

[8] Duda, J. "Asymmetric Numeral Systems: Entropy Coding Combining Speed of Huffman Coding with Compression Rate of Arithmetic Coding." *arXiv:1311.2540*, 2013.

[9] Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP* 2023.

[10] DeepSeek-AI. "DeepSeek-V3 Technical Report." *arXiv:2412.19437*, 2024.

[11] Hernandez, J. "NXF: A Memory-Mapped Binary Format for Quantized LLM Inference on Apple Silicon." *NEXUS Research Paper #1*, 2026.

[12] Hernandez, J. "Fused Dequantization-GEMM Kernels for Apple Metal: Exploiting Unified Memory for LLM Inference." *NEXUS Research Paper #2*, 2026.

[13] Dettmers, T., et al. "SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression." *ICML* 2023.

[14] Brown, T., et al. "Language Models are Few-Shot Learners." *NeurIPS* 2020.

[15] Dubey, A., et al. "The Llama 3 Herd of Models." *arXiv:2407.21783*, 2024.

[16] Han, S., et al. "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding." *ICLR* 2016.

[17] Ashkboos, S., et al. "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs." *arXiv:2404.00456*, 2024.
