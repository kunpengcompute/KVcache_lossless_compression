#include "kvfold_huffman_bf16.h"
#include "kvfold_test_common.h"

static uint16_t *load_raw(const char *path, raw_header_t *h) {
    return (uint16_t *)load_raw_bytes(path, h, 2);
}

static const uint16_t *block_ptr(const uint16_t *raw, const raw_header_t *h,
                                  size_t kv, size_t layer, size_t head, size_t block) {
    return (const uint16_t *)block_ptr_bytes(raw, h, sizeof(*raw), kv, layer, head, block);
}

int main(int argc, char **argv) {
    const char *path;
    int repeat = 5;
    raw_header_t h;
    uint16_t *raw;
    uint8_t *scratch_comp;
    uint16_t *scratch_dec;
    uint8_t *pool;
    size_t pool_cap, pool_size = 0;
    payload_ref_t *refs;
    size_t ref_count, ref_i = 0;
    size_t count, block_bytes, bound;
    size_t l, hd, b, kv, r;
    double t0, t1;
    size_t total_comp_once = 0;
    size_t total_comp_perf = 0;
    uint64_t logical_bytes;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <tensor.kvraw> [repeat]\n", argv[0]);
        return 2;
    }
    path = argv[1];
    if (argc >= 3) repeat = atoi(argv[2]);
    if (repeat < 1) repeat = 1;

    raw = load_raw(path, &h);

    count       = (size_t)h.block_size * h.head_dim;
    block_bytes = count * sizeof(uint16_t);
    logical_bytes = h.data_size;
    bound       = kvfold_huffman_bf16_compress_bound(count);
    scratch_comp = (uint8_t *)xmalloc(bound);
    scratch_dec  = (uint16_t *)xmalloc(block_bytes);
    ref_count    = (size_t)2 * h.layers * h.heads * h.blocks;
    refs         = (payload_ref_t *)xmalloc(ref_count * sizeof(*refs));
    pool_cap     = (size_t)((double)h.data_size * 1.05) + 1024 * 1024;
    pool         = (uint8_t *)xmalloc(pool_cap);

    /* 1. 压缩所有 block，收集到 pool 里 -------------------------------------------------- */
    for (l = 0; l < h.layers; l++) {
        for (hd = 0; hd < h.heads; hd++) {
            for (b = 0; b < h.blocks; b++) {
                for (kv = 0; kv < 2; kv++) {
                    const uint16_t *src = block_ptr(raw, &h, kv, l, hd, b);
                    size_t csz = kvfold_huffman_bf16_compress_block(scratch_comp, bound, src, count);
                    if (csz == 0) {
                        fprintf(stderr, "compress failed L=%zu H=%zu B=%zu KV=%zu\n", l, hd, b, kv);
                        return 1;
                    }
                    if (pool_size + csz > pool_cap) {
                        size_t new_cap = pool_cap * 2 + csz;
                        uint8_t *np = (uint8_t *)realloc(pool, new_cap);
                        if (!np) {
                            fprintf(stderr, "realloc compressed pool failed\n");
                            return 2;
                        }
                        pool = np;
                        pool_cap = new_cap;
                    }
                    refs[ref_i].offset = pool_size;
                    refs[ref_i].size = csz;
                    memcpy(pool + pool_size, scratch_comp, csz);
                    pool_size += csz;
                    ref_i++;
                    total_comp_once += csz;
                }
            }
        }
    }

    /* 2. 验证 round-trip 正确性 ----------------------------------------------------------- */
    ref_i = 0;
    for (l = 0; l < h.layers; l++) {
        for (hd = 0; hd < h.heads; hd++) {
            for (b = 0; b < h.blocks; b++) {
                for (kv = 0; kv < 2; kv++) {
                    const uint16_t *src = block_ptr(raw, &h, kv, l, hd, b);
                    size_t dsz = kvfold_huffman_bf16_decompress_block(
                        pool + refs[ref_i].offset, refs[ref_i].size,
                        scratch_dec, count);
                    if (dsz != block_bytes || memcmp(src, scratch_dec, block_bytes) != 0) {
                        fprintf(stderr, "round-trip mismatch L=%zu H=%zu B=%zu KV=%zu\n",
                                l, hd, b, kv);
                        return 1;
                    }
                    ref_i++;
                }
            }
        }
    }

    printf("shape=(2,%u,%u,%u,%u,%u) raw=%" PRIu64 " bytes\n",
           h.layers, h.heads, h.blocks, h.block_size, h.head_dim, logical_bytes);
    printf("round-trip: OK, payloads=%zu\n", ref_i);
    printf("compressed=%zu bytes, ratio=%.6f\n",
           total_comp_once, (double)total_comp_once / (double)logical_bytes);

    /* 3. 压缩性能测试 -------------------------------------------------------------------- */
    total_comp_perf = 0;
    t0 = now_seconds();
    for (r = 0; r < (size_t)repeat; r++) {
        for (l = 0; l < h.layers; l++) {
            for (hd = 0; hd < h.heads; hd++) {
                for (b = 0; b < h.blocks; b++) {
                    for (kv = 0; kv < 2; kv++) {
                        const uint16_t *src = block_ptr(raw, &h, kv, l, hd, b);
                        size_t csz = kvfold_huffman_bf16_compress_block(scratch_comp, bound, src, count);
                        if (csz == 0) return 1;
                        total_comp_perf += csz;
                    }
                }
            }
        }
    }
    t1 = now_seconds();
    {
        double sec = t1 - t0;
        double mbps = ((double)logical_bytes * (double)repeat / 1000000.0) / sec;
        printf("compress:   repeat=%d time=%.6f s bandwidth=%.2f MBps output=%zu bytes\n",
               repeat, sec, mbps, total_comp_perf);
    }

    /* 4. 解压性能测试 -------------------------------------------------------------------- */
    t0 = now_seconds();
    for (r = 0; r < (size_t)repeat; r++) {
        ref_i = 0;
        for (l = 0; l < h.layers; l++) {
            for (hd = 0; hd < h.heads; hd++) {
                for (b = 0; b < h.blocks; b++) {
                    for (kv = 0; kv < 2; kv++) {
                        size_t dsz = kvfold_huffman_bf16_decompress_block(
                            pool + refs[ref_i].offset, refs[ref_i].size,
                            scratch_dec, count);
                        if (dsz != block_bytes) return 1;
                        ref_i++;
                    }
                }
            }
        }
    }
    t1 = now_seconds();
    {
        double sec = t1 - t0;
        double mbps = ((double)logical_bytes * (double)repeat / 1000000.0) / sec;
        printf("decompress: repeat=%d time=%.6f s bandwidth=%.2f MBps\n",
               repeat, sec, mbps);
    }

    free(pool);
    free(refs);
    free(scratch_dec);
    free(scratch_comp);
    free(raw);
    return 0;
}
