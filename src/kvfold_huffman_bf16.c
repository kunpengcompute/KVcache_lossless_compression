#include "kvfold_huffman_bf16.h"
#include "huf.h"

size_t kvfold_huffman_bf16_compress_bound(size_t count) {
    /* 格式: 16B元数据 + Huffman编码的指数 + count字节全精度尾数
     * HUF_compressBound 只覆盖 Huffman 部分，尾数是额外存储的
     * +64 为元数据(16) + bitstream末尾可能的越界写入(24) + 安全余量 */
    return HUF_compressBound(count * sizeof(uint16_t)) + count + 64;
}

size_t kvfold_huffman_bf16_compress_block(
    void *KVFOLD_HUFFMAN_BF16_RESTRICT dst, size_t dst_capacity,
    const uint16_t *KVFOLD_HUFFMAN_BF16_RESTRICT src,
    size_t count)
{
    return HUF_compress_float(dst, dst_capacity,
                              src, count * sizeof(uint16_t),
                              DT_BF16);
}

size_t kvfold_huffman_bf16_decompress_block(
    const void *src, size_t compressed_size,
    uint16_t *KVFOLD_HUFFMAN_BF16_RESTRICT dst,
    size_t count)
{
    DataType dt = DT_INVALID;
    size_t ret = HUF_decompress_float((void *)dst, count * sizeof(uint16_t),
                                      src, compressed_size, &dt);
    if (ret != count * sizeof(uint16_t)) return 0;
    return ret;
}
