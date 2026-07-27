#ifndef KVFOLD_HUFFMAN_BF16_H
#define KVFOLD_HUFFMAN_BF16_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define KVFOLD_HUFFMAN_BF16_API __declspec(dllexport)
#elif defined(KVFOLD_HUFFMAN_BF16_BUILD_SO)
#define KVFOLD_HUFFMAN_BF16_API __attribute__((visibility("default")))
#else
#define KVFOLD_HUFFMAN_BF16_API
#endif

#if defined(_MSC_VER)
#define KVFOLD_HUFFMAN_BF16_RESTRICT __restrict
#elif defined(__cplusplus)
/* restrict is a C99 keyword, not available in C++; omit for test wrapper */
#define KVFOLD_HUFFMAN_BF16_RESTRICT
#else
#define KVFOLD_HUFFMAN_BF16_RESTRICT restrict
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * kvfold_huffman_bf16_compress_bound:
 *   返回 count 个 BF16 值的最大压缩后大小（字节）。
 *   count = block_size * head_dim
 */
KVFOLD_HUFFMAN_BF16_API size_t kvfold_huffman_bf16_compress_bound(size_t count);

/**
 * kvfold_huffman_bf16_compress_block:
 *   对 src 中 count 个 BF16 值做 Huffman 压缩，写入 dst。
 *   返回压缩后的字节数；返回 0 表示压缩失败（dst 太小或数据不可压缩）。
 */
KVFOLD_HUFFMAN_BF16_API size_t kvfold_huffman_bf16_compress_block(
    void *KVFOLD_HUFFMAN_BF16_RESTRICT dst, size_t dst_capacity,
    const uint16_t *KVFOLD_HUFFMAN_BF16_RESTRICT src,
    size_t count);

/**
 * kvfold_huffman_bf16_decompress_block:
 *   解压 compressed_size 字节的 Huffman 数据到 dst（count 个 BF16 值）。
 *   返回解压后的字节数（count * 2）；失败返回 0。
 */
KVFOLD_HUFFMAN_BF16_API size_t kvfold_huffman_bf16_decompress_block(
    const void *src, size_t compressed_size,
    uint16_t *KVFOLD_HUFFMAN_BF16_RESTRICT dst,
    size_t count);

#ifdef __cplusplus
}
#endif

#endif /* KVFOLD_HUFFMAN_BF16_H */
