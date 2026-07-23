# KV Cache Lossless Compression

## 项目介绍

KV Cache Lossless Compression 是面向大语言模型 BF16 KV Cache 数据的无损压缩项目。项目当前提供 Huffman-BF16 压缩算法、Linux 动态库接口、Python 调用封装以及与 Pure-ZSTD 的对比测试。

本项目用于指导用户编译、部署、验证和使用 Huffman-BF16 压缩算法，并通过测试脚本统计压缩大小、压缩比、压缩与解压耗时以及处理带宽。

## 简要介绍

随着大语言模型上下文长度和并发请求数量增加，KV Cache 占用的内存空间和数据传输带宽不断增长。如何在不损失模型数据精度的前提下降低 KV Cache 的存储和传输开销，是大模型推理系统中的重要问题。

本项目针对 BF16 格式的 KV Cache 数据实现无损 Huffman 压缩。每个 BF16 数值由 16 位组成，算法将其拆分为：

- `E8`：高 8 位，包含符号位和指数相关信息。
- `S1M7`：低 8 位，包含尾数相关信息。

其中，`E8` 部分采用 Huffman 熵编码，并使用 8 路交错数据流进行处理；`S1M7` 部分按原始精度保存。解压时重新组合两个部分，恢复完整的 BF16 位模式，因此解压结果与原始数据逐位一致。

项目同时提供 Pure-ZSTD 基线测试，用于对比以下指标：

- 压缩后大小
- 压缩比
- 压缩与解压耗时
- 压缩与解压带宽
- 解压正确性

## 目录结构

```text
├── compress_lib                           # Huffman、FSE 等底层压缩源码
│   ├── huf_compress.cc                    # Huffman 压缩实现
│   ├── huf_decompress.cc                  # Huffman 解压实现
│   ├── hist.cc                            # BF16 数据直方图统计
│   ├── entropy_common.cc                  # Huffman/FSE 公共逻辑
│   ├── fse_compress.cc                    # FSE 压缩实现
│   ├── fse_decompress.cc                  # FSE 解压实现
│   └── *.h                                # 底层算法头文件
├── scripts                                # 辅助和验证脚本
│   ├── export_bf16_raw.py                 # 将 PyTorch BF16 数据导出为原始数据
│   └── test_so_smoke.py                   # Linux .so 无损往返测试
├── src                                    # Huffman-BF16 接口与 C 测试代码
│   ├── kvfold_huffman_bf16.c              # Huffman-BF16 C 接口实现
│   ├── kvfold_huffman_bf16.h              # 对外公开头文件
│   ├── test_kvfold_huffman_bf16.c         # C 正确性与性能测试
│   └── kvfold_test_common.h               # C 测试公共函数
├── CompressTest.py                        # Pure-ZSTD 与 Huffman-BF16 对比测试
├── kvfold_huffman_bf16.py                 # Python ctypes 调用封装
├── Makefile                               # Linux 编译入口
├── qwen3_8b_bf16_prompt3_token256.pt      # BF16 测试数据
├── qwen3_8b_bf16_prompt3_token256_contiguous.pt
│                                            # 连续内存布局的 BF16 测试数据
├── .gitignore                             # Git 忽略规则
├── LICENSE                                # Apache License 2.0
└── README.md                              # 项目介绍和使用文档
```

编译后会在 `build/` 目录生成测试程序和动态库。`build/` 属于构建产物目录，不需要提交到代码仓。

## 版本说明

当前版本为初始开发版本，主要包含以下能力：

- BF16 KV Cache 无损 Huffman 压缩与解压
- C 语言压缩接口
- Linux `.so` 动态库
- Python ctypes 调用接口
- Pure-ZSTD 与 Huffman-BF16 对比测试
- 压缩与解压正确性校验
- 压缩比、耗时和带宽统计
- AddressSanitizer 和编译告警检查目标

正式发布时建议使用明确的版本号，例如 `v1.0.0`，并补充对应版本的变更记录。

## 兼容性信息

当前项目存在以下约束：

- 当前主要支持 Linux x86-64 运行环境。
- 编译需要 GNU Make 和支持 C++11 的 g++ 编译器。
- 默认编译参数包含 `-march=native`，建议在最终运行算法的目标服务器上编译。
- Python 测试需要 Python 3、PyTorch、NumPy 和 zstandard。
- `CompressTest.py` 的输入张量必须为 `torch.bfloat16`。
- C 接口使用 `uint16_t` 保存 BF16 原始位模式。
- 测试脚本支持 `[layer, 2, seq, head, dim]` 和 `[2, layer, seq, head, dim]` 两种 5 维布局。
- 对于 `[2, layer, seq, hidden]` 四维输入，需要配置 `NUM_HEADS` 和 `HEAD_DIM`。
- 单个 Huffman 原始数据块上限为 128 KiB，即 64K 个 BF16 元素；更大的输入由内部实现处理。
- Linux 生成的 `.so` 文件不能直接由 Windows Python 加载。

说明：

Huffman-BF16 是无损位模式压缩。测试脚本会检查解压结果的形状、数据类型以及每一个 BF16 元素是否与原始数据完全一致。

## 环境部署

### 安装系统依赖

Ubuntu 或 Debian 环境执行：

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3 python3-venv
```

### 获取项目

```bash
git clone https://gitcode.com/boostkit/KVcache_lossless_compression.git
cd KVcache_lossless_compression
```

### 编译 Linux 动态库

首次编译前创建构建目录：

```bash
mkdir -p build
make so
```

编译完成后生成：

```text
build/libkvfold_huffman_bf16.so
```

检查动态库类型：

```bash
file build/libkvfold_huffman_bf16.so
```

预期输出包含：

```text
ELF 64-bit LSB shared object
```

### 其他编译目标

```bash
make          # 编译 C 正确性与性能测试
make asan     # 编译 AddressSanitizer 调试版本
make wall     # 开启额外编译告警
make clean    # 清理编译产物
```

### 安装 Python 依赖

建议使用独立虚拟环境：

```bash
python3 -m venv ~/.venvs/huffman-bf16
source ~/.venvs/huffman-bf16/bin/activate

python -m pip install --upgrade pip
python -m pip install numpy zstandard
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 快速入门

### 验证 `.so` 动态库

`scripts/test_so_smoke.py` 只依赖 Python 标准库，可以直接验证动态库导出的 C 接口：

```bash
python3 scripts/test_so_smoke.py
```

验证成功时输出包含：

```text
SO round-trip: OK
```

该测试会生成确定性的 BF16 位模式，通过 `.so` 完成压缩和解压，并检查解压结果是否与原始数据逐字节一致。

### 配置测试数据

仓库提供以下两个测试数据：

```text
qwen3_8b_bf16_prompt3_token256.pt
qwen3_8b_bf16_prompt3_token256_contiguous.pt
```

运行 `CompressTest.py` 前，需要在脚本配置区设置 `DATA_PATH`。例如：

```python
DATA_PATH = str(
    PROJECT_DIR / "qwen3_8b_bf16_prompt3_token256_contiguous.pt"
)
```

如果 `.pt` 文件中保存的是字典，需要通过 `TENSOR_KEY` 指定张量字段：

```python
TENSOR_KEY = "kv_cache"
```

如果输入是 `[2, layer, seq, hidden]` 四维张量，还需要设置：

```python
NUM_HEADS = 8
HEAD_DIM = 128
```

### 运行对比测试

```bash
python CompressTest.py
```

当前测试只比较：

- Pure-ZSTD Level 3
- Pure-ZSTD Level 5
- Huffman-BF16

输出指标包括：

| 指标 | 说明 |
|---|---|
| `Method` | 压缩算法名称 |
| `Level` | ZSTD 压缩等级，Huffman-BF16 显示为 0 |
| `Orig(MB)` | 原始数据大小 |
| `Comp(MB)` | 压缩后大小 |
| `Ratio` | 压缩比 |
| `Enc(ms)` | 平均压缩耗时 |
| `Dec(ms)` | 平均解压耗时 |
| `Enc GB/s` | 压缩带宽 |
| `Dec GB/s` | 解压带宽 |

## 学习文档

| 学习资源类别 | 学习资源名称 | 学习资源简介 |
|---|---|---|
| 文档 | README | 提供项目介绍、环境部署和快速入门说明 |
| 头文件 | `src/kvfold_huffman_bf16.h` | 提供 C API 定义及参数说明 |
| Python 封装 | `kvfold_huffman_bf16.py` | 提供 Python 压缩和解压接口 |
| 示例 | `scripts/test_so_smoke.py` | 提供 `.so` 动态库正确性验证示例 |
| 测试 | `CompressTest.py` | 提供 Pure-ZSTD 与 Huffman-BF16 对比测试 |

## 免责声明

### 致本项目使用者

本项目仅供调试和开发之用，使用者需自行承担使用风险，并理解以下内容：

- **数据处理及删除：**用户在使用本工具过程中产生的数据属于用户责任范畴。建议用户在使用完毕后及时删除相关数据，以防信息泄露。

- **数据保密与传播：**使用者了解并同意不得将通过本工具产生的数据随意外发或传播。对于由此产生的信息泄露、数据泄露或其他不良后果，本工具及其开发者概不负责。

- **用户输入安全性：**用户需自行保证输入的命令行的安全性，并承担因输入不当而导致的任何安全风险或损失。对于输入命令行不当所导致的问题，本工具及其开发者概不负责。

- **免责声明范围：**本免责声明适用于所有使用本工具的个人或实体。使用本工具即表示您同意并接受本声明的内容，并愿意承担因使用该功能而产生的风险和责任，如有异议请停止使用本工具。

在使用本工具之前，请谨慎阅读并理解以上免责声明的内容。对于使用本工具所产生的任何问题或疑问，请及时联系开发者。

### 致数据所有者

如果您不希望您的模型或数据集等信息在本项目中被提及，或希望更新本项目有关的描述，请在GitCode提交issue，我们将根据您的issue要求删除或更新您相关描述。衷心感谢您对本项目的理解和贡献。

## Licesen

## 贡献声明

欢迎大家为社区做贡献，如果使用过程中有任何问题/建议，或者需要反馈特性需求和bug报告，可以提交Issues联系我们，具体贡献方法可参考这里。同时也欢迎大家在讨论专区展开讨论交流。感谢您的支持。
