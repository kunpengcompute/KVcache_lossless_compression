# CompressTest.py


import io
import json
import struct
import time
from typing import Any, Dict, List, Optional

import torch
import zstandard as zstd
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
import kvfold_huffman_bf16
# =========================
# 配置区：只改这里
# =========================

#
# EX: DATA_PATH = str(PROJECT_DIR / "qwen3_8b_bf16_prompt3_token256.pt")
# EX: DATA_PATH = str(PROJECT_DIR / "qwen3_8b_bf16_prompt3_token256_contiguous.pt")
DATA_PATH = str(PROJECT_DIR / "qwen3_8b_bf16_prompt3_token256_contiguous.pt")

# 如果 .pt 里面直接是 Tensor，保持 None
# 如果 .pt 是 dict，例如 {"kv_cache": tensor}，这里写 "kv_cache"
TENSOR_KEY: Optional[str] = None

# 如果你的数据已经是 5D：
#   [layer, 2, seq, head, dim]
# 或
#   [2, layer, seq, head, dim]
# 这里保持 None 即可
#
# 如果你的数据是 4D：
#   [2, layer, seq, hidden]
# 需要填 num_heads 和 head_dim
NUM_HEADS: Optional[int] = None
HEAD_DIM: Optional[int] = None

# Qwen3-8B 如果 hidden = 1024，通常可以这样：
# NUM_HEADS = 8
# HEAD_DIM = 128

REPEATS = 5
ZSTD_LEVELS = [3, 5]


# =========================
# 工具函数
# =========================

def load_pt_tensor(path: str, tensor_key: Optional[str] = None) -> torch.Tensor:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, torch.Tensor):
        return obj.contiguous()

    if isinstance(obj, dict):
        if tensor_key is not None:
            if tensor_key not in obj:
                raise KeyError(
                    f"tensor_key={tensor_key!r} not found. Available keys: {list(obj.keys())}"
                )
            value = obj[tensor_key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"obj[{tensor_key!r}] is not a torch.Tensor, got {type(value)}"
                )
            return value.contiguous()

        tensor_items = {
            k: v for k, v in obj.items()
            if isinstance(v, torch.Tensor)
        }

        if len(tensor_items) == 1:
            key, value = next(iter(tensor_items.items()))
            print(f"Auto selected tensor key: {key}")
            return value.contiguous()

        raise ValueError(
            "Loaded object is a dict. Please set TENSOR_KEY.\n"
            f"Available keys: {list(obj.keys())}"
        )

    raise TypeError(f"Unsupported .pt object type: {type(obj)}")


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def normalize_kv_tensor(
    tensor: torch.Tensor,
    num_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
) -> torch.Tensor:
    """
    统一成 BitSniper encoder 使用的格式：

        [layer, 2, seq, head, dim]
    """
    tensor = tensor.detach().contiguous()

    if tensor.ndim == 5:
        # 已经是 [layer, 2, seq, head, dim]
        if tensor.shape[1] == 2:
            return tensor.contiguous()

        # 如果是 [2, layer, seq, head, dim]
        if tensor.shape[0] == 2:
            return tensor.permute(1, 0, 2, 3, 4).contiguous()

        raise ValueError(
            f"Unexpected 5D tensor shape: {tuple(tensor.shape)}. "
            "Expected [layer, 2, seq, head, dim] or [2, layer, seq, head, dim]."
        )

    if tensor.ndim == 4:
        # LMCache / vLLM 常见格式: [2, layer, seq, hidden]
        if tensor.shape[0] != 2:
            raise ValueError(
                f"Unexpected 4D tensor shape: {tuple(tensor.shape)}. "
                "Expected [2, layer, seq, hidden]."
            )

        if num_heads is None or head_dim is None:
            raise ValueError(
                "For 4D input [2, layer, seq, hidden], "
                "please set NUM_HEADS and HEAD_DIM."
            )

        kv_num, layer_num, seq_num, hidden_size = tensor.shape

        if hidden_size != num_heads * head_dim:
            raise ValueError(
                f"hidden_size={hidden_size} != num_heads*head_dim={num_heads * head_dim}"
            )

        tensor = tensor.view(
            kv_num,
            layer_num,
            seq_num,
            num_heads,
            head_dim,
        )

        # [2, layer, seq, head, dim] -> [layer, 2, seq, head, dim]
        tensor = tensor.permute(1, 0, 2, 3, 4).contiguous()
        return tensor

    raise ValueError(
        f"Unsupported tensor ndim={tensor.ndim}, shape={tuple(tensor.shape)}. "
        "Expected 4D or 5D KV tensor."
    )


# =========================
# Pure ZSTD
# =========================

def pure_zstd_pack_tensor(tensor: torch.Tensor, level: int) -> bytes:
    """
    纯 ZSTD baseline：
    直接压缩原始 Tensor bytes。
    """
    tensor_cpu = tensor.detach().contiguous().cpu()

    meta = {
        "version": 1,
        "format": "pure_zstd",
        "source_dtype": str(tensor_cpu.dtype)[6:],
        "shape": list(tensor_cpu.shape),
        "zstd_level": level,
    }

    meta_json = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    raw = tensor_cpu.view(torch.uint8).numpy().tobytes(order="C")

    compressed = zstd.ZstdCompressor(level=level).compress(raw)

    with io.BytesIO() as f:
        f.write(struct.pack("<I", len(meta_json)))
        f.write(meta_json)
        f.write(struct.pack("<Q", len(compressed)))
        f.write(compressed)
        return f.getvalue()


def pure_zstd_unpack_tensor(blob: bytes) -> torch.Tensor:
    pos = 0

    meta_len = struct.unpack("<I", blob[pos:pos + 4])[0]
    pos += 4

    meta = json.loads(blob[pos:pos + meta_len].decode("utf-8"))
    pos += meta_len

    compressed_len = struct.unpack("<Q", blob[pos:pos + 8])[0]
    pos += 8

    compressed = blob[pos:pos + compressed_len]
    raw = zstd.ZstdDecompressor().decompress(compressed)

    dtype = getattr(torch, meta["source_dtype"])
    shape = tuple(meta["shape"])

    u8_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    result = u8_tensor.view(dtype).reshape(shape).clone()

    return result

# =========================
# kvfold Huffman BF16
# =========================

def huffman_pack_tensor(tensor: torch.Tensor, level: int) -> bytes:
    """对 tensor 做 Huffman 压缩。level 参数被忽略，仅为适配统一接口。"""
    _ = level
    tensor_cpu = tensor.detach().contiguous().cpu()
    if tensor_cpu.dtype != torch.bfloat16:
        raise TypeError(f"Huffman expects bfloat16 tensor, got {tensor_cpu.dtype}")

    src = tensor_cpu.view(torch.uint16).numpy()
    payload = kvfold_huffman_bf16.compress(src)

    meta = {
        "version": 1,
        "format": "kvfold_huffman_bf16",
        "source_dtype": str(tensor_cpu.dtype)[6:],
        "shape": list(tensor_cpu.shape),
    }
    meta_json = json.dumps(meta, separators=(",", ":")).encode("utf-8")

    with io.BytesIO() as f:
        f.write(struct.pack("<I", len(meta_json)))
        f.write(meta_json)
        f.write(struct.pack("<Q", len(payload)))
        f.write(payload)
        return f.getvalue()


def huffman_unpack_tensor(blob: bytes) -> torch.Tensor:
    pos = 0
    meta_len = struct.unpack("<I", blob[pos:pos + 4])[0]
    pos += 4
    meta = json.loads(blob[pos:pos + meta_len].decode("utf-8"))
    pos += meta_len
    payload_len = struct.unpack("<Q", blob[pos:pos + 8])[0]
    pos += 8
    payload = blob[pos:pos + payload_len]

    result_uint16 = kvfold_huffman_bf16.decompress(payload)
    dtype = getattr(torch, meta["source_dtype"])
    shape = tuple(meta["shape"])
    result = torch.from_numpy(result_uint16).view(dtype).reshape(shape).clone()
    return result


# =========================
# Benchmark
# =========================

def check_correctness(
    name: str,
    level: int,
    original: torch.Tensor,
    decoded: torch.Tensor,
) -> None:
    if original.shape != decoded.shape:
        raise RuntimeError(
            f"{name} L{level} correctness failed: "
            f"shape mismatch, original={tuple(original.shape)}, decoded={tuple(decoded.shape)}"
        )

    if original.dtype != decoded.dtype:
        raise RuntimeError(
            f"{name} L{level} correctness failed: "
            f"dtype mismatch, original={original.dtype}, decoded={decoded.dtype}"
        )

    if not torch.equal(original.cpu(), decoded.cpu()):
        diff_count = (original.cpu() != decoded.cpu()).sum().item()
        raise RuntimeError(
            f"{name} L{level} correctness failed: diff_count={diff_count}"
        )


def benchmark_one_method(
    method_name: str,
    pack_fn,
    unpack_fn,
    tensor: torch.Tensor,
    level: int,
    repeats: int,
) -> Dict[str, Any]:
    original_bytes = tensor_nbytes(tensor)

    # warmup + 正确性校验
    warmup_blob = pack_fn(tensor, level)
    warmup_decoded = unpack_fn(warmup_blob)
    check_correctness(method_name, level, tensor, warmup_decoded)

    compressed_bytes = len(warmup_blob)

    compress_times: List[float] = []
    decompress_times: List[float] = []

    blobs: List[bytes] = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        blob = pack_fn(tensor, level)
        t1 = time.perf_counter()

        compress_times.append(t1 - t0)
        blobs.append(blob)

    last_decoded = None

    for blob in blobs:
        t0 = time.perf_counter()
        last_decoded = unpack_fn(blob)
        t1 = time.perf_counter()

        decompress_times.append(t1 - t0)

    check_correctness(method_name, level, tensor, last_decoded)

    avg_compress_time = sum(compress_times) / len(compress_times)
    avg_decompress_time = sum(decompress_times) / len(decompress_times)

    compression_ratio = original_bytes / compressed_bytes
    compress_bw_gbps = original_bytes / avg_compress_time / 1e9
    decompress_bw_gbps = original_bytes / avg_decompress_time / 1e9

    return {
        "method": method_name,
        "zstd_level": level,
        "original_MB": original_bytes / 1024 / 1024,
        "compressed_MB": compressed_bytes / 1024 / 1024,
        "compression_ratio": compression_ratio,
        "compress_time_ms": avg_compress_time * 1000,
        "decompress_time_ms": avg_decompress_time * 1000,
        "compress_BW_GBps": compress_bw_gbps,
        "decompress_BW_GBps": decompress_bw_gbps,
    }


def print_results(results: List[Dict[str, Any]]) -> None:
    header = (
        f"{'Method':<18}"
        f"{'Level':>8}"
        f"{'Orig(MB)':>12}"
        f"{'Comp(MB)':>12}"
        f"{'Ratio':>10}"
        f"{'Enc(ms)':>12}"
        f"{'Dec(ms)':>12}"
        f"{'Enc(GB/s)':>12}"
        f"{'Dec(GB/s)':>12}"
    )

    print()
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['method']:<18}"
            f"{r['zstd_level']:>8}"
            f"{r['original_MB']:>12.2f}"
            f"{r['compressed_MB']:>12.2f}"
            f"{r['compression_ratio']:>10.4f}"
            f"{r['compress_time_ms']:>12.3f}"
            f"{r['decompress_time_ms']:>12.3f}"
            f"{r['compress_BW_GBps']:>12.3f}"
            f"{r['decompress_BW_GBps']:>12.3f}"
        )


def main():
    tensor = load_pt_tensor(DATA_PATH, TENSOR_KEY)

    print("Loaded raw tensor:")
    print(f"  path  : {DATA_PATH}")
    print(f"  shape : {tuple(tensor.shape)}")
    print(f"  dtype : {tensor.dtype}")
    print(f"  device: {tensor.device}")
    print(f"  size  : {tensor_nbytes(tensor) / 1024 / 1024:.2f} MB")

    tensor = normalize_kv_tensor(
        tensor,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
    )

    print()
    print("Normalized KV tensor:")
    print(f"  shape : {tuple(tensor.shape)}")
    print(f"  dtype : {tensor.dtype}")
    print(f"  size  : {tensor_nbytes(tensor) / 1024 / 1024:.2f} MB")

    if tensor.dtype != torch.bfloat16:
        raise TypeError(
            f"This script expects a bf16 KV tensor, got {tensor.dtype}"
        )

    results: List[Dict[str, Any]] = []

    for level in ZSTD_LEVELS:
        results.append(
            benchmark_one_method(
                method_name="Pure-ZSTD",
                pack_fn=pure_zstd_pack_tensor,
                unpack_fn=pure_zstd_unpack_tensor,
                tensor=tensor,
                level=level,
                repeats=REPEATS,
            )
        )


    # kvfold Huffman BF16 (no ZSTD level, run once)
    results.append(
        benchmark_one_method(
            method_name="Huffman-BF16",
            pack_fn=huffman_pack_tensor,
            unpack_fn=huffman_unpack_tensor,
            tensor=tensor,
            level=0,
            repeats=REPEATS,
        )
    )

    print_results(results)


if __name__ == "__main__":
    main()
