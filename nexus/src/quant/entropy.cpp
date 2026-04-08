/// NEXUS ANS Entropy Codec — tabled Asymmetric Numeral Systems.

#include "quant/entropy.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>
#include <numeric>

namespace nexus::quant {

// ─── Constants ──────────────────────────────────────────────────────────────

static constexpr int kTableBits   = 11;                         // log2(table size)
static constexpr int kTableSize   = 1 << kTableBits;           // 2048
static constexpr int kNumSymbols  = 256;                        // byte alphabet
static constexpr uint32_t kStateLower = kTableSize;             // minimum state
static constexpr uint32_t kStateUpper = kStateLower * 2 - 1;   // maximum normalized state

// Header: 4 bytes magic + 4 bytes original size + 256*2 bytes freq table
static constexpr size_t kHeaderSize = 4 + 4 + kNumSymbols * 2;
static constexpr uint32_t kANSMagic = 0x31534E41; // "ANS1"

// ─── Frequency table ────────────────────────────────────────────────────────

struct FreqTable {
    uint16_t freq[kNumSymbols];     // normalized frequencies summing to kTableSize
    uint16_t cumfreq[kNumSymbols];  // cumulative frequencies (exclusive prefix sum)
};

/// Build and normalize frequency table from input data.
static FreqTable build_freq_table(const uint8_t* input, size_t input_size) {
    FreqTable ft{};

    // Count raw frequencies
    uint32_t counts[kNumSymbols] = {};
    for (size_t i = 0; i < input_size; i++) {
        counts[input[i]]++;
    }

    // Count number of present symbols
    int num_present = 0;
    for (int s = 0; s < kNumSymbols; s++) {
        if (counts[s] > 0) num_present++;
    }
    if (num_present == 0) return ft;

    // Normalize to sum = kTableSize, ensuring each present symbol gets at least 1.
    // First pass: assign proportional frequencies.
    int remaining = kTableSize;
    int assigned = 0;

    for (int s = 0; s < kNumSymbols; s++) {
        if (counts[s] == 0) {
            ft.freq[s] = 0;
        } else {
            uint32_t f = static_cast<uint32_t>(
                (static_cast<uint64_t>(counts[s]) * kTableSize) / input_size);
            if (f < 1) f = 1;
            ft.freq[s] = static_cast<uint16_t>(f);
            assigned += f;
        }
    }

    // Adjust to make sum exactly kTableSize.
    // Distribute the difference among the highest-frequency symbols.
    int diff = kTableSize - assigned;
    while (diff != 0) {
        // Find symbol with highest frequency to adjust
        int best = -1;
        for (int s = 0; s < kNumSymbols; s++) {
            if (ft.freq[s] == 0) continue;
            if (diff > 0) {
                if (best < 0 || ft.freq[s] > ft.freq[best]) best = s;
            } else {
                // Need to subtract; only from symbols with freq > 1
                if (ft.freq[s] > 1) {
                    if (best < 0 || ft.freq[s] > ft.freq[best]) best = s;
                }
            }
        }
        if (best < 0) break;
        if (diff > 0) {
            ft.freq[best]++;
            diff--;
        } else {
            ft.freq[best]--;
            diff++;
        }
    }

    // Build cumulative frequency table
    ft.cumfreq[0] = 0;
    for (int s = 1; s < kNumSymbols; s++) {
        ft.cumfreq[s] = ft.cumfreq[s - 1] + ft.freq[s - 1];
    }

    return ft;
}

// ─── tANS encoding table ───────────────────────────────────────────────────

struct ANSEncSymbol {
    uint16_t freq;
    uint16_t cumfreq;
};

struct ANSDecSymbol {
    uint8_t  symbol;
    uint16_t freq;
    uint16_t cumfreq;
};

/// Build decode table: for each slot in [0, kTableSize), store the symbol info.
static void build_decode_table(ANSDecSymbol* dec_table, const FreqTable& ft) {
    int pos = 0;
    for (int s = 0; s < kNumSymbols; s++) {
        for (int j = 0; j < ft.freq[s]; j++) {
            dec_table[pos].symbol  = static_cast<uint8_t>(s);
            dec_table[pos].freq    = ft.freq[s];
            dec_table[pos].cumfreq = ft.cumfreq[s];
            pos++;
        }
    }
}

// ─── Bit stream (write LSB first) ──────────────────────────────────────────

struct BitWriter {
    uint8_t* buf;
    size_t   capacity;
    size_t   pos;       // byte position
    int      bit_pos;   // bits used in current byte
    bool     overflow;

    BitWriter(uint8_t* b, size_t cap)
        : buf(b), capacity(cap), pos(0), bit_pos(0), overflow(false) {
        if (cap > 0) buf[0] = 0;
    }

    void write_bits(uint32_t val, int nbits) {
        for (int i = 0; i < nbits; i++) {
            if (pos >= capacity) { overflow = true; return; }
            buf[pos] |= ((val >> i) & 1) << bit_pos;
            bit_pos++;
            if (bit_pos == 8) {
                bit_pos = 0;
                pos++;
                if (pos < capacity) buf[pos] = 0;
            }
        }
    }

    size_t bytes_written() const {
        return bit_pos > 0 ? pos + 1 : pos;
    }
};

struct BitReader {
    const uint8_t* buf;
    size_t   size;
    size_t   pos;
    int      bit_pos;

    BitReader(const uint8_t* b, size_t s)
        : buf(b), size(s), pos(0), bit_pos(0) {}

    uint32_t read_bits(int nbits) {
        uint32_t val = 0;
        for (int i = 0; i < nbits; i++) {
            if (pos >= size) return val;
            val |= ((buf[pos] >> bit_pos) & 1) << i;
            bit_pos++;
            if (bit_pos == 8) {
                bit_pos = 0;
                pos++;
            }
        }
        return val;
    }
};

// ─── Public API ─────────────────────────────────────────────────────────────

size_t ans_compress(uint8_t* output, size_t output_size,
                    const uint8_t* input, size_t input_size) {
    if (input_size == 0) return 0;
    if (output_size < kHeaderSize + 4) return 0;

    FreqTable ft = build_freq_table(input, input_size);

    // Write header
    std::memcpy(output, &kANSMagic, 4);
    uint32_t orig_size = static_cast<uint32_t>(input_size);
    std::memcpy(output + 4, &orig_size, 4);
    for (int s = 0; s < kNumSymbols; s++) {
        std::memcpy(output + 8 + s * 2, &ft.freq[s], 2);
    }

    // ANS encoding: process input in REVERSE order.
    // We encode into a temporary bit buffer, then copy.
    // State machine: state in [kTableSize, 2*kTableSize)
    //
    // Encoding step for symbol s with freq[s] and cumfreq[s]:
    //   While state >= freq[s] * (kStateUpper+1)/kTableSize, output low bits
    //   state = ((state / freq[s]) * kTableSize) + (state % freq[s]) + cumfreq[s]

    // Use a vector for the bitstream since we write in reverse
    // We'll collect output bits, then reverse them.
    // Simpler approach: encode forward, store renormalization bits, reverse at end.

    // Actually, standard rANS approach:
    // Encode in reverse order of input. Emit bits to bring state into range.
    // At the end, flush the final state.

    size_t max_compressed = output_size - kHeaderSize;
    BitWriter bw(output + kHeaderSize, max_compressed);

    uint32_t state = kTableSize; // initial state

    // Encode in reverse
    for (int i = static_cast<int>(input_size) - 1; i >= 0; i--) {
        uint8_t sym = input[i];
        uint16_t freq = ft.freq[sym];
        uint16_t cf   = ft.cumfreq[sym];

        if (freq == 0) {
            // Symbol not in table (shouldn't happen if table built from same data)
            return 0;
        }

        // Renormalize: bring state down so that after encoding, state stays in range.
        // We need: state < freq * ((kStateUpper + 1) / kTableSize) after renorm
        // i.e., state < freq * 2  (since kStateUpper+1 = 2*kTableSize, /kTableSize = 2)
        // Wait, kStateUpper = 2*kTableSize - 1, so (kStateUpper+1)/kTableSize = 2
        // So we need state < freq * 2 after renorm.
        // No — let me use a cleaner formulation.
        //
        // rANS: state' = (state / freq) * M + (state % freq) + cumfreq
        // where M = kTableSize.
        // For state' to stay <= kStateUpper = 2M-1, we need state/freq <= 1 + (M-1-cumfreq)/M
        // Simpler: we renormalize by outputting bits until state < freq << kTableBits
        // But actually for a streaming rANS with bit-level output:
        //
        // max_state = (freq << kTableBits) - 1  (this ensures output state < 2*M)
        // While state > max_state, output one bit and shift.

        uint32_t max_state_for_sym = (static_cast<uint32_t>(freq) << kTableBits) - 1;
        // Actually for proper rANS: we want after encoding,
        // new_state = (state/freq)*M + state%freq + cf
        // and new_state should be in [M, 2M).
        // This means state/freq in [1, 2) after renorm, i.e. state in [freq, 2*freq).
        // So renormalize until state < 2*freq by emitting bits.

        while (state >= static_cast<uint32_t>(freq) * 2) {
            bw.write_bits(state & 1, 1);
            state >>= 1;
        }

        // Encode: state' = (state / freq) * kTableSize + (state % freq) + cf
        state = (state / freq) * kTableSize + (state % freq) + cf;
    }

    // Flush final state (kTableBits + 1 bits)
    bw.write_bits(state, kTableBits + 1);

    if (bw.overflow) return 0;

    size_t compressed = kHeaderSize + bw.bytes_written();

    // If compressed is larger than original, store uncompressed (return 0 = failure)
    if (compressed >= input_size + kHeaderSize) return 0;

    return compressed;
}

size_t ans_decompress(uint8_t* output, size_t output_size,
                      const uint8_t* input, size_t input_size) {
    if (input_size < kHeaderSize) return 0;

    // Read header
    uint32_t magic;
    std::memcpy(&magic, input, 4);
    if (magic != kANSMagic) return 0;

    uint32_t orig_size;
    std::memcpy(&orig_size, input + 4, 4);
    if (orig_size > output_size) return 0;

    // Rebuild frequency table
    FreqTable ft{};
    for (int s = 0; s < kNumSymbols; s++) {
        std::memcpy(&ft.freq[s], input + 8 + s * 2, 2);
    }
    ft.cumfreq[0] = 0;
    for (int s = 1; s < kNumSymbols; s++) {
        ft.cumfreq[s] = ft.cumfreq[s - 1] + ft.freq[s - 1];
    }

    // Build decode table
    std::vector<ANSDecSymbol> dec_table(kTableSize);
    build_decode_table(dec_table.data(), ft);

    // Read compressed bitstream
    // We need to read bits in reverse order of how they were written.
    // The bitstream is: [renorm bits for last symbol ... renorm bits for first symbol] [final state]
    // We read from the END: first read the final state, then decode forward.

    // Read all bits into a buffer for reverse reading
    const uint8_t* bitdata = input + kHeaderSize;
    size_t bitdata_size = input_size - kHeaderSize;

    // Count total bits
    size_t total_bits = bitdata_size * 8;

    // We need to read from the end. Build a reverse bit reader.
    // The last thing written was the final state (kTableBits+1 bits).
    // Before that were the renormalization bits in reverse encode order.

    // Read all bits into a flat array for easy reverse access
    std::vector<uint8_t> bits(total_bits);
    for (size_t i = 0; i < total_bits; i++) {
        bits[i] = (bitdata[i / 8] >> (i % 8)) & 1;
    }

    // Read final state from the END of the bitstream
    // The writer wrote: renorm bits... then state bits at the end.
    // We need to find the end. The writer used bw.bytes_written() bytes.
    // We have exactly that many bytes in bitdata_size.

    // Read state from the last kTableBits+1 bits
    int state_bits = kTableBits + 1;
    size_t bit_cursor = total_bits; // points past the end

    // Scan backwards to find actual end (skip padding zeros in last byte)
    // Actually, the bitwriter may have padding. Let's just use total_bits
    // and read state from the end.

    // The state was written at the end, LSB first. Read it.
    bit_cursor -= state_bits;
    uint32_t state = 0;
    for (int i = 0; i < state_bits; i++) {
        if (bit_cursor + i < total_bits) {
            state |= static_cast<uint32_t>(bits[bit_cursor + i]) << i;
        }
    }

    // Now read renormalization bits. They were written in reverse-encode order
    // (last input symbol first). We decode in forward order (first symbol first),
    // so we read renorm bits from the bit_cursor backwards.
    size_t renorm_cursor = bit_cursor; // next bit to read is at renorm_cursor - 1

    // Decode forward
    for (size_t i = 0; i < orig_size; i++) {
        // Decode symbol from state
        // state is in [kTableSize, 2*kTableSize)
        // slot = state % kTableSize (but actually state IS in that range, so
        // we need to reverse the encoding step)
        //
        // Encoding was: state' = (state/freq)*M + (state%freq) + cf
        // So: slot = state' % M = (state%freq) + cf
        // freq_idx = state' / M = state/freq (approximately)
        //
        // To decode: slot = state % M
        //            find symbol s such that cumfreq[s] <= slot < cumfreq[s]+freq[s]
        //            state = (state / M) * freq[s] + slot - cumfreq[s]
        //            Then renormalize by reading bits

        uint32_t slot = state % kTableSize;
        const ANSDecSymbol& ds = dec_table[slot];
        output[i] = ds.symbol;

        // Update state
        state = (state / kTableSize) * ds.freq + (slot - ds.cumfreq);

        // Renormalize: read bits until state >= kTableSize
        while (state < kTableSize && renorm_cursor > 0) {
            renorm_cursor--;
            state = (state << 1) | bits[renorm_cursor];
        }
    }

    return orig_size;
}

size_t ans_compressed_size(const uint8_t* input, size_t input_size) {
    if (input_size == 0) return 0;

    // Count symbol frequencies
    uint32_t counts[kNumSymbols] = {};
    for (size_t i = 0; i < input_size; i++) {
        counts[input[i]]++;
    }

    // Compute Shannon entropy
    double entropy_bits = 0.0;
    double n = static_cast<double>(input_size);
    for (int s = 0; s < kNumSymbols; s++) {
        if (counts[s] == 0) continue;
        double p = counts[s] / n;
        entropy_bits -= counts[s] * std::log2(p);
    }

    // Convert to bytes and add header overhead
    size_t data_bytes = static_cast<size_t>(std::ceil(entropy_bits / 8.0));
    return kHeaderSize + data_bytes;
}

}  // namespace nexus::quant
