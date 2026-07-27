"""Smoke-test the Linux libkvfold_huffman_bf16.so with ctypes only."""

import argparse
import ctypes
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=65_536)
    parser.add_argument(
        "--library",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "build"
        / "libkvfold_huffman_bf16.so",
    )
    args = parser.parse_args()

    library_path = args.library.resolve()
    if not library_path.is_file():
        raise FileNotFoundError(f"Shared library not found: {library_path}")

    lib = ctypes.CDLL(str(library_path))
    lib.kvfold_huffman_bf16_compress_bound.argtypes = [ctypes.c_size_t]
    lib.kvfold_huffman_bf16_compress_bound.restype = ctypes.c_size_t
    lib.kvfold_huffman_bf16_compress_block.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.kvfold_huffman_bf16_compress_block.restype = ctypes.c_size_t
    lib.kvfold_huffman_bf16_decompress_block.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.kvfold_huffman_bf16_decompress_block.restype = ctypes.c_size_t

    rng = random.Random(0)
    source_type = ctypes.c_uint16 * args.count
    source = source_type(
        *(
            (rng.choice((0x3E, 0x3F, 0x40, 0xBF)) << 8) | rng.randrange(256)
            for _ in range(args.count)
        )
    )

    capacity = lib.kvfold_huffman_bf16_compress_bound(args.count)
    compressed = ctypes.create_string_buffer(capacity)
    compressed_size = lib.kvfold_huffman_bf16_compress_block(
        compressed,
        capacity,
        source,
        args.count,
    )
    if compressed_size == 0:
        raise RuntimeError("Compression returned 0")

    decoded = source_type()
    decoded_size = lib.kvfold_huffman_bf16_decompress_block(
        compressed,
        compressed_size,
        decoded,
        args.count,
    )
    raw_size = args.count * ctypes.sizeof(ctypes.c_uint16)
    if decoded_size != raw_size:
        raise RuntimeError(
            f"Unexpected decoded size: expected {raw_size}, got {decoded_size}"
        )

    source_bytes = ctypes.string_at(ctypes.addressof(source), raw_size)
    decoded_bytes = ctypes.string_at(ctypes.addressof(decoded), raw_size)
    if decoded_bytes != source_bytes:
        raise AssertionError("Decoded BF16 bit patterns differ from the source")

    print("SO round-trip: OK")
    print(f"library          : {library_path}")
    print(f"elements         : {args.count}")
    print(f"raw bytes        : {raw_size}")
    print(f"compressed bytes : {compressed_size}")
    print(f"compression ratio: {raw_size / compressed_size:.4f}")


if __name__ == "__main__":
    main()
