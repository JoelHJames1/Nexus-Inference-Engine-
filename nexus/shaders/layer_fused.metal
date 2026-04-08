/// NEXUS Metal Shader — Fused Layer Kernel
///
/// Executes the ENTIRE dense FFN portion of a transformer layer in a
/// sequence of dispatches within a single command buffer (no commit
/// between them). The caller encodes all dispatches, commits once.
///
/// This eliminates per-GEMM command buffer overhead:
///   Before: 7 dispatches × 1 commit each = 7 commits/layer
///   After:  7 dispatches × 0 commits + 1 final commit = 1 commit/layer
///
/// Operations fused (encoded sequentially with memory barriers):
///   1. RMSNorm (pre-attention)
///   2. Q/K/V GEMV (3 parallel dispatches)
///   3. Output projection GEMV
///   4. Residual add
///   5. RMSNorm (pre-FFN)
///   6. W1 GEMV + W3 GEMV (parallel)
///   7. SwiGLU fused
///   8. W2 GEMV
///   9. Residual add
///
/// The key insight: Apple Silicon's Metal memory barriers are lightweight
/// (~microseconds) compared to command buffer commit+wait (~500μs).
/// So encoding 7+ dispatches with barriers costs ~10μs total vs ~3.5ms
/// for 7 separate command buffers.

// This file just documents the approach.
// The actual implementation is in the MetalBackend which encodes
// multiple existing shader dispatches (rmsnorm, gemv_int4_uniform,
// swiglu_fused, residual_add) into one command buffer using
// persistent encoder mode (batch mode).
//
// No new shader needed — we reuse existing shaders but encode them
// differently (one command buffer for the entire layer).

#include <metal_stdlib>
using namespace metal;

// Placeholder — the real fusion happens at the encoding level in
// metal_backend.mm's batch mode, not in a single monolithic shader.
// Each operation is still a separate dispatch but they share one
// command buffer with memory barriers between them.
