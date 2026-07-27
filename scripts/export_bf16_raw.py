#!/usr/bin/env python3
"""Export a BF16 KV cache .pt tensor to the simple .kvraw format used by C tests."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch


BF16_DTYPE = "torch.bfloat16"
RAW_MAGIC = b"KVFRAW1\0"
RAW_VERSION = 1


def load_bf16_tensor(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or "kv_cache" not in obj:
        raise ValueError(f"{path}: expected a dict containing key 'kv_cache'")
    tensor = obj["kv_cache"].contiguous()
    if str(tensor.dtype) != BF16_DTYPE:
        raise ValueError(f"{path}: expected BF16 tensor, got {tensor.dtype}")
    arr = tensor.view(torch.uint16).numpy().copy()
    if arr.ndim != 6 or arr.shape[0] != 2:
        raise ValueError(f"{path}: expected shape (2,L,H,B,T,C), got {arr.shape}")
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a BF16 KV cache .pt tensor to .kvraw.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = load_bf16_tensor(args.input)
    _, layers, heads, blocks, block_size, head_dim = raw.shape
    data = np.ascontiguousarray(raw).astype("<u2", copy=False)
    header = struct.pack("<8sIIIIIIIQ", RAW_MAGIC, RAW_VERSION, layers, heads, blocks, block_size, head_dim, 2, int(raw.nbytes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        f.write(header)
        f.write(data.tobytes())
    print(f"Wrote {args.output} ({raw.nbytes} data bytes)")


if __name__ == "__main__":
    main()
