cmake_minimum_required(VERSION 3.25)
project(nexus VERSION 0.1.0 LANGUAGES C CXX OBJCXX)

# ─── C++20 ───────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Platform check ──────────────────────────────────────────────────────────
if(NOT APPLE)
    message(FATAL_ERROR "NEXUS is designed exclusively for Apple Silicon (macOS arm64)")
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    message(WARNING "NEXUS targets Apple Silicon (arm64). Building on ${CMAKE_SYSTEM_PROCESSOR} may not produce optimal code.")
endif()

# ─── Build type defaults ─────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -mcpu=apple-m1")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DNEXUS_DEBUG")

# ─── Apple frameworks ────────────────────────────────────────────────────────
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(COREML_FRAMEWORK CoreML)

# Metal shader compiler
find_program(METAL_COMPILER xcrun REQUIRED)

# ─── Options ─────────────────────────────────────────────────────────────────
option(NEXUS_BUILD_TESTS "Build unit tests" ON)
option(NEXUS_BUILD_TOOLS "Build CLI tools" ON)
option(NEXUS_ENABLE_COREML "Enable CoreML/Neural Engine support" ON)

# ─── Third-party dependencies ────────────────────────────────────────────────
# metal-cpp (Apple's C++ wrapper for Metal)
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/metal-cpp")
    message(STATUS "metal-cpp not found, will use Objective-C++ Metal API directly")
    set(NEXUS_USE_METAL_CPP OFF)
else()
    set(NEXUS_USE_METAL_CPP ON)
    add_library(metal-cpp INTERFACE)
    target_include_directories(metal-cpp INTERFACE third_party/metal-cpp)
endif()

# xxHash (header-only for checksums)
if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/xxHash")
    add_library(xxhash INTERFACE)
    target_include_directories(xxhash INTERFACE third_party/xxHash)
    target_compile_definitions(xxhash INTERFACE XXH_INLINE_ALL)
else()
    message(STATUS "xxHash not found in third_party/, checksum support disabled")
endif()

# ─── Core library ────────────────────────────────────────────────────────────
add_library(nexus_core STATIC
    # Format
    src/format/nxf_reader.cpp
    src/format/nxf_writer.cpp

    # Memory
    src/memory/memory_manager.cpp
    src/memory/uma_allocator.cpp
    src/memory/prefetcher.cpp

    # Compute - Accelerate
    src/compute/accelerate/gemm.cpp

    # Compute - CPU (NEON)
    src/compute/cpu/dequant_neon.cpp

    # Compute - Metal
    src/compute/metal/metal_backend.mm
    src/compute/metal/metal_context.mm

    # KV Cache
    src/kv/kv_store.cpp
    src/kv/turbo_quant.cpp
    src/kv/paged_attention.cpp
    src/kv/kv_quantize.cpp
    src/kv/prefix_cache.cpp
    src/kv/eviction.cpp

    # Quantization
    src/quant/gptq.cpp
    src/quant/quip_sharp.cpp
    src/quant/awq.cpp
    src/quant/entropy.cpp

    # Model
    src/model/transformer.cpp
    src/model/hybrid_model.cpp
    src/model/moe_router.cpp
    src/model/moe_layer.cpp
    src/model/speculative.cpp

    # Compute - CoreML/ANE
    src/compute/coreml/draft_model.mm

    # Import
    src/import/gguf_importer.cpp
    src/import/safetensors_importer.cpp
    src/import/convert_cli.cpp

    # API
    src/api/http_server.cpp
    src/api/benchmark.cpp
    src/api/bench_cli.cpp
    src/api/serve_cli.cpp

    # Core
    src/core/engine.cpp
    src/core/scheduler.cpp
)

target_include_directories(nexus_core PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(nexus_core PUBLIC
    ${ACCELERATE_FRAMEWORK}
    ${METAL_FRAMEWORK}
    ${METALKIT_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

if(NEXUS_ENABLE_COREML AND COREML_FRAMEWORK)
    target_link_libraries(nexus_core PUBLIC ${COREML_FRAMEWORK})
    target_compile_definitions(nexus_core PUBLIC NEXUS_HAS_COREML=1)
endif()

if(NEXUS_USE_METAL_CPP)
    target_link_libraries(nexus_core PUBLIC metal-cpp)
    target_compile_definitions(nexus_core PUBLIC NEXUS_USE_METAL_CPP=1)
endif()

# NEON intrinsics are always available on arm64
target_compile_options(nexus_core PRIVATE -march=armv8.2-a+fp16)

# ─── Metal shader compilation ────────────────────────────────────────────────
set(METAL_SHADERS
    shaders/gemm_dequant.metal
    shaders/flash_attention.metal
    shaders/turbo_quant.metal
    shaders/rmsnorm.metal
    shaders/activations.metal
)

set(METAL_AIR_FILES "")
foreach(SHADER ${METAL_SHADERS})
    if(EXISTS "${CMAKE_SOURCE_DIR}/${SHADER}")
        get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
        set(AIR_FILE "${CMAKE_BINARY_DIR}/${SHADER_NAME}.air")
        add_custom_command(
            OUTPUT ${AIR_FILE}
            COMMAND xcrun -sdk macosx metal -c ${CMAKE_SOURCE_DIR}/${SHADER} -o ${AIR_FILE}
            DEPENDS ${CMAKE_SOURCE_DIR}/${SHADER}
            COMMENT "Compiling Metal shader: ${SHADER}"
        )
        list(APPEND METAL_AIR_FILES ${AIR_FILE})
    endif()
endforeach()

# Check if Metal compiler is actually available
execute_process(
    COMMAND xcrun -sdk macosx metal --version
    RESULT_VARIABLE METAL_COMPILER_RESULT
    OUTPUT_QUIET ERROR_QUIET
)

if(METAL_AIR_FILES AND METAL_COMPILER_RESULT EQUAL 0)
    add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/nexus_shaders.metallib"
        COMMAND xcrun -sdk macosx metallib ${METAL_AIR_FILES} -o "${CMAKE_BINARY_DIR}/nexus_shaders.metallib"
        DEPENDS ${METAL_AIR_FILES}
        COMMENT "Linking Metal shader library"
    )
    add_custom_target(nexus_shaders ALL DEPENDS "${CMAKE_BINARY_DIR}/nexus_shaders.metallib")
    add_dependencies(nexus_core nexus_shaders)
else()
    message(STATUS "Metal shader compiler not available — shaders will not be compiled.")
    message(STATUS "Run: xcodebuild -downloadComponent MetalToolchain")
endif()

# ─── CLI executable ──────────────────────────────────────────────────────────
if(NEXUS_BUILD_TOOLS)
    add_executable(nexus
        src/api/cli.cpp
    )
    target_link_libraries(nexus PRIVATE nexus_core)

    # Install
    install(TARGETS nexus RUNTIME DESTINATION bin)
    if(EXISTS "${CMAKE_BINARY_DIR}/nexus_shaders.metallib")
        install(FILES "${CMAKE_BINARY_DIR}/nexus_shaders.metallib" DESTINATION share/nexus)
    endif()
endif()

# ─── Tests ───────────────────────────────────────────────────────────────────
if(NEXUS_BUILD_TESTS)
    enable_testing()

    add_executable(nexus_test_format tests/test_nxf_format.cpp)
    target_link_libraries(nexus_test_format PRIVATE nexus_core)
    add_test(NAME nxf_format COMMAND nexus_test_format)

    add_executable(nexus_test_memory tests/test_memory_manager.cpp)
    target_link_libraries(nexus_test_memory PRIVATE nexus_core)
    add_test(NAME memory_manager COMMAND nexus_test_memory)
endif()

# ─── Summary ─────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "╔══════════════════════════════════════════════╗")
message(STATUS "║       NEXUS Inference Engine v${PROJECT_VERSION}          ║")
message(STATUS "╠══════════════════════════════════════════════╣")
message(STATUS "║ Platform:    macOS arm64 (Apple Silicon)     ║")
message(STATUS "║ C++ Standard: C++20                          ║")
message(STATUS "║ Build Type:  ${CMAKE_BUILD_TYPE}                          ║")
message(STATUS "║ Metal:       ${METAL_FRAMEWORK}              ║")
message(STATUS "║ Accelerate:  ${ACCELERATE_FRAMEWORK}         ║")
message(STATUS "║ CoreML:      ${COREML_FRAMEWORK}             ║")
message(STATUS "║ Tests:       ${NEXUS_BUILD_TESTS}            ║")
message(STATUS "╚══════════════════════════════════════════════╝")
message(STATUS "")
