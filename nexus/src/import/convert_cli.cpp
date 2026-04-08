/// NEXUS CLI — convert command.
///
/// Usage:
///   nexus convert <input.gguf> <output.nxf> [--codec int4|int8|fp16|fp32]
///
/// Converts a llama.cpp GGUF model file to NEXUS NXF format.

#include "import/gguf_importer.h"
#include "core/config.h"

#include <cstdio>
#include <cstring>
#include <string>

using nexus::Codec;
using nexus::import::GGUFImporter;

static void print_usage(const char* argv0) {
    fprintf(stderr,
        "Usage: %s <input.gguf> <output.nxf> [options]\n"
        "\n"
        "Convert a GGUF model file to NEXUS NXF format.\n"
        "\n"
        "Options:\n"
        "  --codec <type>   Target quantization codec (default: int4)\n"
        "                   Supported: int4, int8, fp16, fp32\n"
        "  --help           Show this help message\n"
        "\n"
        "Examples:\n"
        "  %s llama-7b.gguf llama-7b.nxf\n"
        "  %s llama-7b-f16.gguf llama-7b-q4.nxf --codec int4\n"
        "  %s llama-7b.gguf llama-7b-fp16.nxf --codec fp16\n",
        argv0, argv0, argv0, argv0);
}

static Codec parse_codec(const char* s) {
    if (strcmp(s, "int4") == 0 || strcmp(s, "INT4") == 0 || strcmp(s, "q4") == 0)
        return Codec::INT4;
    if (strcmp(s, "int8") == 0 || strcmp(s, "INT8") == 0 || strcmp(s, "q8") == 0)
        return Codec::INT8;
    if (strcmp(s, "fp16") == 0 || strcmp(s, "FP16") == 0 || strcmp(s, "f16") == 0)
        return Codec::FP16;
    if (strcmp(s, "fp32") == 0 || strcmp(s, "FP32") == 0 || strcmp(s, "f32") == 0)
        return Codec::FP32;
    if (strcmp(s, "gptq") == 0 || strcmp(s, "GPTQ") == 0)
        return Codec::GPTQ;
    if (strcmp(s, "awq") == 0 || strcmp(s, "AWQ") == 0)
        return Codec::AWQ;
    // Default to INT4 on unrecognized
    fprintf(stderr, "warning: unrecognized codec '%s', defaulting to int4\n", s);
    return Codec::INT4;
}

/// Entry point for the "convert" CLI subcommand.
///
/// @param argc  Argument count (including the subcommand name itself).
/// @param argv  Argument vector. argv[0] is the subcommand name ("convert").
/// @return 0 on success, 1 on failure.
int cmd_convert(int argc, char** argv) {
    // We expect: convert <input> <output> [--codec <type>]
    // argv[0] = "convert" (or program name, depending on how caller passes it)

    std::string input_path;
    std::string output_path;
    Codec target_codec = Codec::INT4;

    // Parse arguments (skip argv[0] which is the command name)
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "--codec") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --codec requires an argument\n");
                return 1;
            }
            target_codec = parse_codec(argv[++i]);
            continue;
        }
        if (argv[i][0] == '-') {
            fprintf(stderr, "error: unknown option '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
        // Positional arguments
        if (positional == 0) {
            input_path = argv[i];
        } else if (positional == 1) {
            output_path = argv[i];
        } else {
            fprintf(stderr, "error: too many positional arguments\n");
            print_usage(argv[0]);
            return 1;
        }
        ++positional;
    }

    if (input_path.empty() || output_path.empty()) {
        fprintf(stderr, "error: input and output paths are required\n");
        print_usage(argv[0]);
        return 1;
    }

    // Run conversion
    bool ok = GGUFImporter::convert(input_path, output_path, target_codec);
    return ok ? 0 : 1;
}
