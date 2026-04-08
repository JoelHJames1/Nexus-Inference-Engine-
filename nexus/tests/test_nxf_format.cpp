/// NEXUS Tests — NXF format read/write round-trip.

#include "format/nxf.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdlib>

using namespace nexus;
using namespace nexus::format;

static void test_write_read_roundtrip() {
    const char* test_path = "/tmp/nexus_test.nxf";

    // Write a test NXF file
    {
        auto writer = NXFWriter::create(test_path);
        assert(writer && "Failed to create writer");

        ModelManifest manifest{};
        manifest.architecture = "llama";
        manifest.name = "test-model-7b";
        manifest.num_layers = 2;
        manifest.hidden_dim = 128;
        manifest.num_heads = 4;
        manifest.num_kv_heads = 4;
        manifest.head_dim = 32;
        manifest.vocab_size = 256;
        manifest.max_seq_len = 512;
        manifest.rope_theta = 10000.0f;
        manifest.rms_norm_eps = 1e-5f;
        manifest.num_experts = 0;
        manifest.num_active_experts = 0;
        manifest.default_codec = Codec::FP16;
        manifest.default_group_size = 128;
        writer->set_manifest(manifest);

        // Write a small test tensor
        std::vector<float> data(128 * 128, 1.0f);
        for (int i = 0; i < 128 * 128; i++) data[i] = static_cast<float>(i) / 1000.0f;

        writer->begin_tensor("layers.0.attention.wq.weight", {128, 128}, DType::F32);
        writer->add_chunk(data.data(), data.size() * sizeof(float), Codec::FP32);
        writer->end_tensor();

        writer->begin_tensor("layers.0.attention_norm.weight", {128}, DType::F32);
        std::vector<float> norm_data(128, 1.0f);
        writer->add_chunk(norm_data.data(), norm_data.size() * sizeof(float), Codec::FP32);
        writer->end_tensor();

        writer->finalize();
    }

    // Read it back
    {
        auto reader = NXFReader::open(test_path);
        assert(reader && "Failed to open reader");

        const auto& m = reader->manifest();
        assert(m.architecture == "llama");
        assert(m.name == "test-model-7b");
        assert(m.num_layers == 2);
        assert(m.hidden_dim == 128);
        assert(m.vocab_size == 256);

        auto names = reader->tensor_names();
        assert(names.size() == 2);

        const auto* wq = reader->get_tensor("layers.0.attention.wq.weight");
        assert(wq != nullptr);
        assert(wq->shape.size() == 2);
        assert(wq->shape[0] == 128 && wq->shape[1] == 128);
        assert(wq->chunks.size() == 1);

        // Map and verify data
        const void* mapped = reader->map_chunk(wq->chunks[0]);
        assert(mapped != nullptr);
        const float* fdata = static_cast<const float*>(mapped);
        assert(fdata[0] == 0.0f);
        assert(fdata[1000] == 1.0f);

        reader->close();
    }

    // Cleanup
    remove(test_path);
    printf("[PASS] NXF write/read round-trip\n");
}

static void test_chunk_alignment() {
    // Verify chunks are 16KB aligned
    assert(kChunkAlignment == 16384);
    assert(kPageSize == 16384);

    ChunkDesc desc{};
    desc.file_offset = 0;
    desc.compressed_size = 100;
    desc.decompressed_size = 200;
    desc.codec = Codec::INT4;
    desc.group_size = 128;
    assert(sizeof(ChunkDesc) == 24);

    printf("[PASS] Chunk alignment constants\n");
}

static void test_header_size() {
    assert(sizeof(NXFHeader) == 64);
    printf("[PASS] NXF header size\n");
}

int main() {
    printf("NEXUS NXF Format Tests\n");
    printf("======================\n");

    test_header_size();
    test_chunk_alignment();
    test_write_read_roundtrip();

    printf("\nAll NXF format tests passed!\n");
    return 0;
}
