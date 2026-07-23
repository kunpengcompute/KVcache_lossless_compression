#ifndef KVFOLD_TEST_COMMON_H
#define KVFOLD_TEST_COMMON_H

#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
static double now_seconds(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
#else
#include <time.h>
static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

#pragma pack(push, 1)
typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t layers;
    uint32_t heads;
    uint32_t blocks;
    uint32_t block_size;
    uint32_t head_dim;
    uint32_t dtype_code;
    uint64_t data_size;
} raw_header_t;
#pragma pack(pop)

typedef struct {
    size_t offset;
    size_t size;
} payload_ref_t;

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "malloc(%zu) failed\n", n);
        exit(2);
    }
    return p;
}

static void *load_raw_bytes(const char *path, raw_header_t *h, uint32_t dtype_code) {
    FILE *f = fopen(path, "rb");
    void *data;
    size_t got;
    if (!f) {
        fprintf(stderr, "open %s failed: %s\n", path, strerror(errno));
        exit(2);
    }
    if (fread(h, sizeof(*h), 1, f) != 1) {
        fprintf(stderr, "read header failed\n");
        exit(2);
    }
    if (memcmp(h->magic, "KVFRAW1", 7) != 0 || h->version != 1 || h->dtype_code != dtype_code) {
        fprintf(stderr, "invalid raw file header\n");
        exit(2);
    }
    data = xmalloc((size_t)h->data_size);
    got = fread(data, 1, (size_t)h->data_size, f);
    fclose(f);
    if (got != (size_t)h->data_size) {
        fprintf(stderr, "read data failed: got %zu expected %" PRIu64 "\n", got, h->data_size);
        exit(2);
    }
    return data;
}

static const void *block_ptr_bytes(const void *raw, const raw_header_t *h,
                                   size_t elem_size, size_t kv,
                                   size_t layer, size_t head, size_t block) {
    size_t off = (((((kv * h->layers + layer) * h->heads + head) * h->blocks + block)
                  * h->block_size) * h->head_dim);
    return (const uint8_t *)raw + off * elem_size;
}

#endif
