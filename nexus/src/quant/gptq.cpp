/// NEXUS GPTQ Codec — INT4/INT8 dequantization.

#include "quant/gptq.h"
#include <cmath>
#include <algorithm>

namespace nexus::quant {

void dequant_int4(float* out, const uint8_t* data, const float* scales,
                  const float* zeros, int n, int group_size) {
    for (int i = 0; i < n; i += 2) {
        uint8_t packed = data[i / 2];
        int group = i / group_size;
        float scale = scales[group];
        float zero = zeros ? zeros[group] : 0.0f;

        // Low nibble
        int val_lo = packed & 0x0F;
        out[i] = (static_cast<float>(val_lo) - zero) * scale;

        // High nibble
        if (i + 1 < n) {
            int val_hi = (packed >> 4) & 0x0F;
            out[i + 1] = (static_cast<float>(val_hi) - zero) * scale;
        }
    }
}

void dequant_int8(float* out, const int8_t* data, const float* scales,
                  int n, int group_size) {
    for (int i = 0; i < n; i++) {
        int group = i / group_size;
        out[i] = static_cast<float>(data[i]) * scales[group];
    }
}

void quant_int4(uint8_t* out, float* scales_out, float* zeros_out,
                const float* data, int n, int group_size) {
    int num_groups = (n + group_size - 1) / group_size;

    for (int g = 0; g < num_groups; g++) {
        int start = g * group_size;
        int end = std::min(start + group_size, n);

        // Find min/max in group
        float vmin = data[start], vmax = data[start];
        for (int i = start + 1; i < end; i++) {
            vmin = std::min(vmin, data[i]);
            vmax = std::max(vmax, data[i]);
        }

        // Compute scale and zero point for 4-bit range [0, 15]
        float scale = (vmax - vmin) / 15.0f;
        if (scale == 0.0f) scale = 1.0f;
        float zero = -vmin / scale;

        scales_out[g] = scale;
        zeros_out[g] = zero;

        // Quantize
        for (int i = start; i < end; i += 2) {
            int q0 = std::clamp(static_cast<int>(roundf(data[i] / scale + zero)), 0, 15);
            int q1 = 0;
            if (i + 1 < end) {
                q1 = std::clamp(static_cast<int>(roundf(data[i + 1] / scale + zero)), 0, 15);
            }
            out[i / 2] = static_cast<uint8_t>(q0 | (q1 << 4));
        }
    }
}

}  // namespace nexus::quant
