/// NEXUS Tests — Memory manager allocation and RSS tracking.

#include "memory/memory_manager.h"
#include <cassert>
#include <cstdio>
#include <cstring>

using namespace nexus;

static void test_alloc_free() {
    MemoryConfig config;
    config.ram_limit = 1ULL * 1024 * 1024 * 1024;  // 1 GB for test
    MemoryManager mm(config);

    // Allocate a 1 MB page-aligned buffer
    size_t alloc_size = 1 * 1024 * 1024;
    void* ptr = mm.alloc_pages(alloc_size);
    assert(ptr != nullptr);

    // Verify it's page-aligned (16 KB)
    assert(reinterpret_cast<uintptr_t>(ptr) % kPageSize == 0);

    // Write to it
    memset(ptr, 0xAB, alloc_size);

    // Free it
    mm.free_pages(ptr, alloc_size);

    printf("[PASS] alloc/free pages\n");
}

static void test_rss_tracking() {
    MemoryConfig config;
    config.ram_limit = 1ULL * 1024 * 1024 * 1024;
    MemoryManager mm(config);

    size_t rss_before = mm.current_rss();
    assert(rss_before > 0);

    printf("[PASS] RSS tracking (current: %.1f MB)\n", rss_before / (1024.0 * 1024.0));
}

int main() {
    printf("NEXUS Memory Manager Tests\n");
    printf("==========================\n");

    test_alloc_free();
    test_rss_tracking();

    printf("\nAll memory manager tests passed!\n");
    return 0;
}
