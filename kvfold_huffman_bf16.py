"""kvfold_huffman_bf16 — Python ctypes wrapper for libkvfold_huffman_bf16.so

Usage:
    import kvfold_huffman_bf16

    payload = kvfold_huffman_bf16.compress(arr)       # arr is np.uint16
    result  = kvfold_huffman_bf16.decompress(payload)  # returns np.uint16
"""

import ctypes as _ctypes
import os as _os
import struct as _struct
from pathlib import Path as _Path
from typing import Optional as _Optional

import numpy as _np

_BUILD_DIR = _Path(__file__).resolve().parent / "build"
if _os.name == "nt":
    _LIB_PATH = _BUILD_DIR / "kvfold_huffman_bf16.dll"
else:
    _LIB_PATH = _BUILD_DIR / "libkvfold_huffman_bf16.so"

_lib = _ctypes.CDLL(str(_LIB_PATH))

_lib.kvfold_huffman_bf16_compress_bound.argtypes = [_ctypes.c_size_t]
_lib.kvfold_huffman_bf16_compress_bound.restype = _ctypes.c_size_t

_lib.kvfold_huffman_bf16_compress_block.argtypes = [
    _ctypes.c_void_p, _ctypes.c_size_t,
    _ctypes.c_void_p, _ctypes.c_size_t,
]
_lib.kvfold_huffman_bf16_compress_block.restype = _ctypes.c_size_t

_lib.kvfold_huffman_bf16_decompress_block.argtypes = [
    _ctypes.c_void_p, _ctypes.c_size_t,
    _ctypes.c_void_p, _ctypes.c_size_t,
]
_lib.kvfold_huffman_bf16_decompress_block.restype = _ctypes.c_size_t


def compress(src: _np.ndarray) -> bytes:
    """Compress a uint16 numpy array. Returns compressed bytes.

    Format: 8-byte total element count (little-endian uint64), then Huffman payload.
    """
    if src.dtype != _np.uint16:
        raise TypeError(f"Expected uint16 array, got {src.dtype}")
    src = _np.ascontiguousarray(src)
    count = src.size
    bound = _lib.kvfold_huffman_bf16_compress_bound(count)
    dst = _np.zeros(bound, dtype=_np.uint8)
    csize = _lib.kvfold_huffman_bf16_compress_block(
        dst.ctypes.data_as(_ctypes.c_void_p), bound,
        src.ctypes.data_as(_ctypes.c_void_p), count,
    )
    # csize == 0 means incompressible; csize > bound means error code (e.g. ~0UL)
    if csize == 0 or csize > bound:
        # Fall back to storing raw bytes (no compression)
        raw = src.tobytes()
        header = _struct.pack("<IQ", count, 0)  # count | (flag=0 → raw)
        return header + raw
    # Prepend total element count so decompress knows the output size
    header = _struct.pack("<Q", count)
    return header + dst[:csize].tobytes()


def decompress(payload: bytes, count: _Optional[int] = None) -> _np.ndarray:
    """Decompress bytes produced by compress(). Returns uint16 numpy array.

    The payload format is: 8-byte total element count (uint64 LE), then either
    raw fallback data or Huffman-compressed blocks.
    """
    # Read total element count from 8-byte header
    total_count = _struct.unpack("<Q", payload[:8])[0]
    data = payload[8:]

    # Check if this is a raw fallback (next 4 bytes are flag=0)
    if len(data) >= 4:
        raw_flag = _struct.unpack("<I", data[:4])[0]
        if raw_flag == 0:
            # Raw uncompressed fallback
            return _np.frombuffer(data[4:], dtype=_np.uint16, count=total_count).copy()

    count_val = count if count is not None else total_count
    dst = _np.zeros(count_val, dtype=_np.uint16)
    dec = _lib.kvfold_huffman_bf16_decompress_block(
        data, len(data),
        dst.ctypes.data_as(_ctypes.c_void_p), count_val,
    )
    if dec != count_val * 2:
        raise RuntimeError(
            f"kvfold_huffman_bf16 decompress failed: "
            f"expected {count_val * 2} bytes, got {dec}"
        )
    return dst
