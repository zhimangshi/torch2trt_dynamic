## ScatterND（VDSP 向量化）初版说明

这个目录提供一个 **ScatterND 的最小可运行骨架**，用于你在 Synopsys VDSP 仿真器里快速起步：

- 文件：`scatternd_vdsp_demo.cpp`
- 包含：
  - **标量参考实现**（便于对照正确性）
  - **VDSP 向量实现**（核心用 `vscatter` + predicate tail）
  - **简单 demo main**（初始化数据、计时、对比 ref）

### 当前支持范围（v1）

- **数据类型**：`updates/output = int8`（可视作你说的 FIXPOINT 数据流），`indices = int32`
- **语义**：`reduction="none"`（覆盖写入；若索引重复，表现为 **last write wins**）
- **形状**：
  - `indices` 形状 `[N, K]`
  - `output` 形状 `[D0..D(R-1)]`
  - `updates` 形状 `[N, slice_size]`，其中 `slice_size = prod(output_shape[K..R-1])`
    - 若 `K == R`，则 `slice_size == 1`（每个 index 写一个元素）
    - 若 `K < R`，则每个 index 写一个连续 slice（例如写一整行/一整块）

### 为什么 ScatterND 要这样向量化

ScatterND 的难点是 **写地址不规则（随机写）**：

- SIMD 最擅长的是“连续 load/store”。但 ScatterND 的每个更新对应的输出位置由 `indices` 决定，通常是离散的。
- 在 VDSP 上要并行做这种随机写，最直接可行的方式是：
  - **把多个更新打包成一个向量**（每个 lane 一个更新）
  - **计算每个 lane 的目标 offset**
  - 使用 **`vscatter`** 一次性把整向量写到各自地址

同时，为了覆盖 `K < R`（每次写一个 slice）这种常见场景：

- 先计算单个更新的 `base_offset`
- 再在 slice 内做向量化（slice 通常是连续的），这样吞吐更稳定

尾部处理（当 N 或 slice_size 不是向量宽度整数倍）：

- 用 predicate（`vvci_b() < n`）保护 tail lanes，避免越界。

### 如何迁移到你的工程

你需要把这个 demo 的两个部分“拷贝式迁移”：

1. `scatternd_vdsp_i8(...)`：核心向量内核
2. 你工程的内存分配/张量布局/计时框架（类似你 RMSNorm 那套）

然后根据你真实的 ScatterND 规格补齐：

- bounds check（index 越界处理）
- reduction（add/mul/…）
- 支持更多 dtype（int16/int32/f16 等）

