CXX      := g++
CXXFLAGS := -std=c++11 -O3 -march=native -I src -I compress_lib

SRC := src/test_kvfold_huffman_bf16.c src/kvfold_huffman_bf16.c \
       compress_lib/huf_compress.cc compress_lib/huf_decompress.cc \
       compress_lib/hist.cc compress_lib/entropy_common.cc \
       compress_lib/fse_compress.cc compress_lib/fse_decompress.cc

SO_SRC := src/kvfold_huffman_bf16.c \
          compress_lib/huf_compress.cc compress_lib/huf_decompress.cc \
          compress_lib/hist.cc compress_lib/entropy_common.cc \
          compress_lib/fse_compress.cc compress_lib/fse_decompress.cc

.PHONY: all so asan wall clean

all: build/test_kvfold_huffman_bf16

build/test_kvfold_huffman_bf16: $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@

so: build/libkvfold_huffman_bf16.so

build/libkvfold_huffman_bf16.so: $(SO_SRC)
	$(CXX) $(CXXFLAGS) -fPIC -shared -DKVFOLD_HUFFMAN_BF16_BUILD_SO $^ -o $@

asan: CXXFLAGS := -std=c++11 -O1 -g -fsanitize=address -fno-omit-frame-pointer -I src -I compress_lib
asan: build/test_kvfold_huffman_bf16_asan

build/test_kvfold_huffman_bf16_asan: $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@

wall: CXXFLAGS := -std=c++11 -O2 -Wall -Wextra -I src -I compress_lib
wall: build/test_kvfold_huffman_bf16_wall

build/test_kvfold_huffman_bf16_wall: $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f build/test_kvfold_huffman_bf16 build/test_kvfold_huffman_bf16_asan build/test_kvfold_huffman_bf16_wall build/libkvfold_huffman_bf16.so
