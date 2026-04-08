/// NEXUS KV Store — Phase 1 flat buffer implementation.

#include "kv/kv_store.h"
#include <cstdlib>
#include <cstring>
#include <vector>

namespace nexus::kv {

KVStore::KVStore(uint32_t num_layers, uint32_t num_kv_heads,
                 uint32_t head_dim, uint32_t max_seq_len)
    : num_layers_(num_layers)
    , num_kv_heads_(num_kv_heads)
    , head_dim_(head_dim)
    , max_seq_len_(max_seq_len)
    , kv_dim_(num_kv_heads * head_dim) {

    layers_.resize(num_layers);
    size_t buf_size = max_seq_len * kv_dim_ * sizeof(float);

    for (auto& lc : layers_) {
        lc.keys = static_cast<float*>(aligned_alloc(kPageSize, (buf_size + kPageSize - 1) & ~(kPageSize - 1)));
        lc.values = static_cast<float*>(aligned_alloc(kPageSize, (buf_size + kPageSize - 1) & ~(kPageSize - 1)));
        lc.seq_len = 0;
        if (lc.keys) memset(lc.keys, 0, buf_size);
        if (lc.values) memset(lc.values, 0, buf_size);
    }
}

KVStore::~KVStore() {
    for (auto& lc : layers_) {
        free(lc.keys);
        free(lc.values);
    }
}

float* KVStore::key_at(uint32_t layer, uint32_t seq_pos) {
    if (layer >= num_layers_ || seq_pos >= max_seq_len_) return nullptr;
    return layers_[layer].keys + seq_pos * kv_dim_;
}

float* KVStore::value_at(uint32_t layer, uint32_t seq_pos) {
    if (layer >= num_layers_ || seq_pos >= max_seq_len_) return nullptr;
    return layers_[layer].values + seq_pos * kv_dim_;
}

uint32_t KVStore::seq_len(uint32_t layer) const {
    if (layer >= num_layers_) return 0;
    return layers_[layer].seq_len;
}

void KVStore::advance(uint32_t layer) {
    if (layer < num_layers_ && layers_[layer].seq_len < max_seq_len_) {
        layers_[layer].seq_len++;
    }
}

void KVStore::reset() {
    for (auto& lc : layers_) {
        lc.seq_len = 0;
    }
}

size_t KVStore::memory_bytes() const {
    size_t per_layer = max_seq_len_ * kv_dim_ * sizeof(float) * 2;  // keys + values
    return per_layer * num_layers_;
}

}  // namespace nexus::kv
