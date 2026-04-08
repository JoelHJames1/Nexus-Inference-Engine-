/// NEXUS Inference Engine — NEON dequantization kernels for arm64.

#include "dequant_neon.h"

#include <arm_neon.h>
#include <cassert>

namespace nexus::compute {

// ─── Helpers ────────────────────────────────────────────────────────────────

namespace {

/// Unpack 8 packed INT4 bytes (16 nibbles) into two uint8x8 vectors
/// containing the low and high nibble values (0-15) respectively.
inline void unpack_nibbles(uint8x16_t packed,
                           uint8x8_t& lo_nibbles,
                           uint8x8_t& hi_nibbles)
{
    // packed contains 16 bytes. Each byte = (high_nibble << 4) | low_nibble.
    // We want to extract the first 8 bytes and produce 16 uint8 values.
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);

    uint8x16_t lo = vandq_u8(packed, mask_lo);         // low nibbles
    uint8x16_t hi = vshrq_n_u8(packed, 4);             // high nibbles

    lo_nibbles = vget_low_u8(lo);
    hi_nibbles = vget_low_u8(hi);
}

}  // anonymous namespace

// ─── dequant_int4_to_f32 ────────────────────────────────────────────────────

void dequant_int4_to_f32(float* out, const uint8_t* data,
                         const float* scales, const float* zeros,
                         int n, int group_size)
{
    assert(n % 2 == 0 && "n must be even for INT4 packing");
    assert(group_size % 2 == 0 && "group_size must be even");

    int i = 0;  // index into output elements

    const int num_groups = n / group_size;

    for (int g = 0; g < num_groups; ++g) {
        const float scale = scales[g];
        const float zero  = zeros[g];

        const float32x4_t v_scale = vdupq_n_f32(scale);
        const float32x4_t v_zero  = vdupq_n_f32(zero);

        const int group_start = g * group_size;
        const int byte_start  = group_start / 2;

        // Process 16 output elements per iteration (8 packed bytes).
        int j = 0;
        for (; j + 16 <= group_size; j += 16) {
            // Load 8 bytes = 16 nibbles.
            uint8x8_t packed = vld1_u8(&data[byte_start + j / 2]);

            // Unpack to low/high nibbles.
            const uint8x8_t mask_lo = vdup_n_u8(0x0F);
            uint8x8_t lo = vand_u8(packed, mask_lo);    // even indices
            uint8x8_t hi = vshr_n_u8(packed, 4);        // odd indices

            // Interleave: out[0]=lo[0], out[1]=hi[0], out[2]=lo[1], ...
            uint8x8x2_t interleaved = vzip_u8(lo, hi);  // two 8-element vectors

            // Convert first 8 nibbles to float and dequantize.
            {
                uint16x8_t u16 = vmovl_u8(interleaved.val[0]);
                uint32x4_t u32_lo = vmovl_u16(vget_low_u16(u16));
                uint32x4_t u32_hi = vmovl_u16(vget_high_u16(u16));

                float32x4_t f_lo = vcvtq_f32_u32(u32_lo);
                float32x4_t f_hi = vcvtq_f32_u32(u32_hi);

                // out = (nibble - zero) * scale
                f_lo = vmulq_f32(vsubq_f32(f_lo, v_zero), v_scale);
                f_hi = vmulq_f32(vsubq_f32(f_hi, v_zero), v_scale);

                vst1q_f32(&out[group_start + j + 0], f_lo);
                vst1q_f32(&out[group_start + j + 4], f_hi);
            }

            // Convert second 8 nibbles.
            {
                uint16x8_t u16 = vmovl_u8(interleaved.val[1]);
                uint32x4_t u32_lo = vmovl_u16(vget_low_u16(u16));
                uint32x4_t u32_hi = vmovl_u16(vget_high_u16(u16));

                float32x4_t f_lo = vcvtq_f32_u32(u32_lo);
                float32x4_t f_hi = vcvtq_f32_u32(u32_hi);

                f_lo = vmulq_f32(vsubq_f32(f_lo, v_zero), v_scale);
                f_hi = vmulq_f32(vsubq_f32(f_hi, v_zero), v_scale);

                vst1q_f32(&out[group_start + j +  8], f_lo);
                vst1q_f32(&out[group_start + j + 12], f_hi);
            }
        }

        // Scalar tail for remaining elements within the group.
        for (; j < group_size; j += 2) {
            uint8_t byte = data[byte_start + j / 2];
            float lo_val = static_cast<float>(byte & 0x0F);
            float hi_val = static_cast<float>(byte >> 4);

            out[group_start + j + 0] = (lo_val - zero) * scale;
            out[group_start + j + 1] = (hi_val - zero) * scale;
        }
    }
}

// ─── dequant_int8_to_f32 ────────────────────────────────────────────────────

void dequant_int8_to_f32(float* out, const int8_t* data,
                         const float* scales,
                         int n, int group_size)
{
    const int num_groups = n / group_size;

    for (int g = 0; g < num_groups; ++g) {
        const float scale = scales[g];
        const float32x4_t v_scale = vdupq_n_f32(scale);

        const int base = g * group_size;

        // Process 16 elements per iteration.
        int j = 0;
        for (; j + 16 <= group_size; j += 16) {
            // Load 16 signed bytes.
            int8x16_t s8 = vld1q_s8(&data[base + j]);

            // Widen to 16-bit.
            int16x8_t s16_lo = vmovl_s8(vget_low_s8(s8));
            int16x8_t s16_hi = vmovl_s8(vget_high_s8(s8));

            // Widen to 32-bit and convert to float.
            int32x4_t s32_0 = vmovl_s16(vget_low_s16(s16_lo));
            int32x4_t s32_1 = vmovl_s16(vget_high_s16(s16_lo));
            int32x4_t s32_2 = vmovl_s16(vget_low_s16(s16_hi));
            int32x4_t s32_3 = vmovl_s16(vget_high_s16(s16_hi));

            float32x4_t f0 = vcvtq_f32_s32(s32_0);
            float32x4_t f1 = vcvtq_f32_s32(s32_1);
            float32x4_t f2 = vcvtq_f32_s32(s32_2);
            float32x4_t f3 = vcvtq_f32_s32(s32_3);

            // Multiply by scale.
            f0 = vmulq_f32(f0, v_scale);
            f1 = vmulq_f32(f1, v_scale);
            f2 = vmulq_f32(f2, v_scale);
            f3 = vmulq_f32(f3, v_scale);

            vst1q_f32(&out[base + j +  0], f0);
            vst1q_f32(&out[base + j +  4], f1);
            vst1q_f32(&out[base + j +  8], f2);
            vst1q_f32(&out[base + j + 12], f3);
        }

        // Scalar tail.
        for (; j < group_size; ++j) {
            out[base + j] = static_cast<float>(data[base + j]) * scale;
        }
    }
}

// ─── dequant_int4_gemv ──────────────────────────────────────────────────────

void dequant_int4_gemv(float* out, const uint8_t* weight,
                       const float* scales, const float* zeros,
                       const float* x,
                       int rows, int cols, int group_size)
{
    assert(cols % 2 == 0);

    const int groups_per_row = cols / group_size;

    for (int row = 0; row < rows; ++row) {
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);

        const uint8_t* row_data   = &weight[row * (cols / 2)];
        const float*   row_scales = &scales[row * groups_per_row];
        const float*   row_zeros  = &zeros[row * groups_per_row];

        for (int g = 0; g < groups_per_row; ++g) {
            const float scale = row_scales[g];
            const float zero  = row_zeros[g];

            const float32x4_t v_scale = vdupq_n_f32(scale);
            const float32x4_t v_zero  = vdupq_n_f32(zero);

            const int col_base  = g * group_size;
            const int byte_base = col_base / 2;

            // Process 16 columns per iteration (8 packed bytes).
            int j = 0;
            for (; j + 16 <= group_size; j += 16) {
                // Load 8 packed bytes = 16 INT4 values.
                uint8x8_t packed = vld1_u8(&row_data[byte_base + j / 2]);

                const uint8x8_t mask = vdup_n_u8(0x0F);
                uint8x8_t lo = vand_u8(packed, mask);
                uint8x8_t hi = vshr_n_u8(packed, 4);

                // Interleave low and high nibbles.
                uint8x8x2_t zipped = vzip_u8(lo, hi);

                // First 8 elements.
                {
                    uint16x8_t u16 = vmovl_u8(zipped.val[0]);
                    float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16)));
                    float32x4_t f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16)));

                    f0 = vmulq_f32(vsubq_f32(f0, v_zero), v_scale);
                    f1 = vmulq_f32(vsubq_f32(f1, v_zero), v_scale);

                    // Load corresponding x values.
                    float32x4_t x0 = vld1q_f32(&x[col_base + j + 0]);
                    float32x4_t x1 = vld1q_f32(&x[col_base + j + 4]);

                    acc0 = vfmaq_f32(acc0, f0, x0);
                    acc1 = vfmaq_f32(acc1, f1, x1);
                }

                // Second 8 elements.
                {
                    uint16x8_t u16 = vmovl_u8(zipped.val[1]);
                    float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16)));
                    float32x4_t f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16)));

                    f0 = vmulq_f32(vsubq_f32(f0, v_zero), v_scale);
                    f1 = vmulq_f32(vsubq_f32(f1, v_zero), v_scale);

                    float32x4_t x0 = vld1q_f32(&x[col_base + j +  8]);
                    float32x4_t x1 = vld1q_f32(&x[col_base + j + 12]);

                    acc2 = vfmaq_f32(acc2, f0, x0);
                    acc3 = vfmaq_f32(acc3, f1, x1);
                }
            }

            // Scalar tail for remaining columns in this group.
            float scalar_acc = 0.0f;
            for (; j < group_size; j += 2) {
                uint8_t byte = row_data[byte_base + j / 2];
                float v_lo = (static_cast<float>(byte & 0x0F) - zero) * scale;
                float v_hi = (static_cast<float>(byte >> 4)   - zero) * scale;

                scalar_acc += v_lo * x[col_base + j + 0];
                if (j + 1 < group_size) {
                    scalar_acc += v_hi * x[col_base + j + 1];
                }
            }

            // Fold scalar tail into accumulator lane.
            acc0 = vsetq_lane_f32(vgetq_lane_f32(acc0, 0) + scalar_acc, acc0, 0);
        }

        // Horizontal reduction of 4 accumulators.
        float32x4_t sum01 = vaddq_f32(acc0, acc1);
        float32x4_t sum23 = vaddq_f32(acc2, acc3);
        float32x4_t total = vaddq_f32(sum01, sum23);

        out[row] = vaddvq_f32(total);
    }
}

}  // namespace nexus::compute
