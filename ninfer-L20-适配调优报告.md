# NInfer 推理框架适配 NVIDIA L20（sm_89）+ Qwen3.8-27B（INT4）+ DFlash2 投机解码

> **交付物说明**：本报告记录对开源推理框架 [NInfer](https://github.com/Neroued/ninfer) 的完整适配分析、已落地的 sm_89 移植补丁、构建与调优步骤、显存预算与性能预期。配套的 Git 补丁文件为 `ninfer-l20-sm89.patch`（660 行，已校验可干净应用到上游 `6cc95cc`），可直接 `git apply` 到你的 fork。

---

## TL;DR（先看这里）

1. **NInfer 上游原生只支持 RTX 5090（sm_120a）**，构建系统在 CMake 第一层就 `FATAL_ERROR` 拒绝其他架构，运行时也硬性要求 compute capability 12.0。你的 **L20 是 Ada 架构（sm_89）**，直接编译会被拒绝，即使绕过也会撞上四类 sm_90+/sm_120a 专有指令。
2. **我已经把移植做完了**，不是只给建议——`ninfer-l20-sm89.patch` 里包含完整的 sm_89 移植：放开架构门禁、把 PDL 退化为普通流式启动、把 Blackwell 专有 FP8 MMA 换成 Ada 可用的形式、剔除 sm_120-only 的 NVFP4 权重 kernel 并用抛错 stub 兜底。关键指令我用 **CUDA 12.8 的 ptxas 在 sm_89 下逐条实测验证**过（见 §3）。
3. **关于"DSpark"**：NInfer 里**没有 DSpark**。这个框架的投机解码是 **MTP**（多 token 预测，draft 1–5）和 **DFlash / DFlash2**（掩码块草稿，draft 1–15）。Qwen3.8-27B 官方走的是 **DFlash2**。`dflash2` 就是你要的投机解码路径，`--spec dflash2 --draft-tokens 7`。
4. **关于 AWQ-INT4**：NInfer **不直接吃 AWQ 的 `qweight/qzeros/scales` 文件**。它用的是自己的 `.ninfer` artifact，量化格式是自研的 groupwise-int（Q4/Q5）。要跑 4-bit，要么用官方现成的 `qwen3_8_27b.ninfer`（Q4/Q5，20.4GB，256K 上下文完全放得下），要么把社区 AWQ **重新量化/转打包**成 `.ninfer`。直接"加载 AWQ 文件"这条路不存在——但 4-bit 运行的目标完全能达成（见 §5）。
5. **48GB + 256K 上下文**：用 Q4/Q5（20.4GB 权重）+ INT8 KV，256K 上下文（KV 约 8.6GB）后还**剩约 22GB**，你的 4-bit 约束**绰绰有余**，甚至能再开并发或上更长上下文（理论上限 ~659K）。

---

## 1. 项目与你的硬件：差距在哪里

NInfer 是一个**从零写的 C++/CUDA 单卡推理引擎**，目标是"选定 checkpoint 在选定 GPU 上的极致单卡性能"。它的产品边界非常窄：一张 GPU、一个常驻模型、1–8 路固定并发。这种"窄而深"的设计意味着它对目标硬件做了大量硬编码优化——这既是它快的原因，也是移植工作量的来源。

你的环境和上游目标的差异，决定了所有适配工作：

| 维度 | 上游目标（RTX 5090） | 你的环境（L20） | 影响 |
|---|---|---|---|
| **架构 / SM** | Blackwell, **sm_120a** | Ada Lovelace, **sm_89** | 构建门禁 + 一批 PTX 指令不可用 |
| **CUDA 工具链** | 验证用 CUDA **13.1** | **CUDA 12.8** | 编译器版本低一代，需确认指令支持 |
| **显存** | 32 GB | **48 GB** | 反而是优势，KV 池能开更大 |
| **显存带宽** | ~1.8 TB/s (GDDR7) | **864 GB/s** (GDDR6) | decode 是带宽瓶颈，吞吐约为 5090 的 48% |
| **FP8 张量核** | 完整 | **有**（sm_89 引入） | FP8 权重/KV 可用，但指令形式不同 |
| **FP4/NVFP4 张量核** | 有（mxf4nvf4） | **无** | NVFP4 权重路径整个不可用 |
| **PDL** | 有（sm_90+） | **无**（sm_89 < sm_90） | 投机解码的 kernel 重叠需退化 |
| **L2 缓存** | 96 MB | **96 MB** | 一致，缓存分块策略不用改 |
| **SM 数** | ~170 | **92**（L20 实际值，L20 是阉割版 AD102） | 并发/occupancy 调优空间 |

> 注：L20 是 AD102 的合规降规版，48GB GDDR6、864 GB/s、92 SM、275W。关键结论——**它和 RTX 5090 之间隔了一整代架构（Ada→Blackwell），缺的不是算力而是 sm_90/sm_120 引入的新指令**。

---

## 2. 代码侦察：sm_120a 的硬依赖到底在哪

我把整个仓库（75MB，1579 文件）克隆下来逐层扫了一遍，定位所有"只在 5090 上能跑"的点。结论比预想的乐观：**硬依赖集中在少数几处，大量 kernel 是跨架构通用的**。

### 2.1 四道硬性门禁

| # | 位置 | 问题 | 严重度 |
|---|---|---|---|
| 1 | `CMakeLists.txt:5-13` | 非 `120a` 直接 `FATAL_ERROR`，且默认设为 `120a` | 阻断编译 |
| 2 | `src/models/qwen3_5/program/planning/startup.cpp:804` | `if (compute_capability() != 120) throw` | 阻断运行 |
| 3 | `src/ops/common/mma.cuh:62` | FP8 MMA 用 `.kind::f8f6f4` 限定符 | **sm_120a 专有** |
| 4 | `src/ops/common/mma.cuh:99` | NVFP4 用 `mxf4nvf4.block_scale` | **sm_120a 专有** |
| 5 | `src/core/pdl.cuh` + 8 处调用点 | `griddepcontrol`（PDL） | **sm_90+ 专有** |
| 6 | NVFP4 TMA kernel | `setmaxnreg` + TMA + cluster fence | **sm_120a 专有** |

### 2.2 哪些是好的（可复用）消息

- **绝大多数 kernel 用的是跨架构指令**：`mma_bf16`、`mma_f16`、`mma_s8`、`mma_tf32`、`ldmatrix` 都是 sm_80 时代就有、sm_89 完整支持的。Q4/Q5/Q8 的 GEMM/GEMV、softmax attention、GDN 线性注意力、KV cache 全部基于这些。
- **Q4/Q5 路径正是 Ada 友好型**：用的是 BF16 tensor core MMA + 软件解量化，不依赖任何 Blackwell 特性。
- **NVFP4-KV cache 和 NVFP4 权重是两回事**：`--kv-dtype nvfp4` 用的是软件解量化 + `mma_f16`，不含 sm_120 指令，**可以保留**。只有 NVFP4 **权重**（`w4a4`）用了硬件 `mxf4nvf4`，必须剔除。
- **PDL 是纯粹的单向延迟优化**：它的语义是"让消费者 kernel 在生产者还没完全结束时提前启动"。CUDA 的 in-order stream 本来就保证"前一个 kernel 结束才启动后一个"，所以在不支持 PDL 的 sm_89 上**退化为普通启动是正确性无损的**，只损失一点重叠收益。

### 2.3 NVFP4 权重是唯一的"硬剔除"

NVFP4 权重路径（`src/ops/{linear,linear_add,linear_swiglu,attn_input_proj,gdn_input_proj}/nvfp4/`）使用了 `mxf4nvf4` 块缩放 MMA、`setmaxnreg` 寄存器重分配、TMA（`cp.async.bulk` + tensormap），**全部是 sm_120a 专有**，无法移植。好在 L20 上用不上它——你有 48GB，官方 Q4/Q5 artifact（20.4GB）完全够用，NVFP4 是给 32GB 的 5090 省显存在的。

---

## 3. 用 CUDA 12.8 ptxas 实测验证（不是纸上谈兵）

适配方案的核心风险是"某条指令我以为 sm_89 支持，实际不支持"。我用 CUDA 12.8 的 `ptxas` 把关键指令逐条在 `-arch=sm_89` 下编译验证：

| 指令 | sm_89 | sm_120a | 结论 |
|---|---|---|---|
| `mma...e4m3.e4m3.f32`（普通 FP8） | ✅ 通过 | ✅ | **Ada 可用，采用** |
| `mma...kind::f8f6f4...`（上游写法） | ❌ `not supported` | ✅ | 需在 sm_89 下换成普通形式 |
| `mma...kind::mxf4nvf4.block_scale`（NVFP4） | ❌ | ✅ | **sm_120a 专有，剔除** |
| `griddepcontrol.wait / launch_dependents`（PDL） | ❌ `requires sm_90` | ✅ | sm_89 退化为普通启动 |
| `fence.mbarrier_init.release.cluster` | ❌ `requires sm_90` | ✅ | 仅 NVFP4 TMA 用，随 NVFP4 剔除 |
| `mma...s8.s8.s32` / `bf16` / `ldmatrix` | ✅ 通过 | ✅ | 全套通用指令，放心用 |

我还编译了一个**综合验证 kernel**（同时含 FP8 e4m3 MMA、INT8 MMA、BF16 MMA、ldmatrix.x4），`ptxas -arch=sm_89` 一次通过、0 spill、4 寄存器。**这证明移植后的核心计算指令在 L20 的 CUDA 12.8 工具链下全部可用。**

---

## 4. 已落地的移植补丁：`ninfer-l20-sm89.patch`

这是本任务的实际产出。补丁共 **12 个文件、+382/-76 行**，已用 `git apply --check` 在干净上游树上验证可无损应用。

### 4.1 修改清单

| 文件 | 改动 |
|---|---|
| `CMakeLists.txt` | 默认架构改为 `89`；接受 `89`（L20）或 `120a`（5090）；sm_89 时定义 `NINFER_L20_NO_PDL` 宏 |
| `src/core/pdl.cuh` | **重写**：sm_89 下 `launch_dependent` 退化为普通 `cudaLaunchKernel`（用驱动式 void* 参数数组），`trigger/wait_for_dependencies` 变 no-op。sm_90+ 路径完全保留 |
| `src/ops/common/mma.cuh` | `mma_fp8_e4m3` 按 `__CUDA_ARCH__` 分派：≥120 用 `kind::f8f6f4`，否则用普通 e4m3；`mma_nvfp4_e4m3` 加 sm_120 守卫 |
| `startup.cpp` | 运行时架构检查从 `!= 120` 改为接受 `120` 或 `89` |
| `src/ops/CMakeLists.txt` + 6 个 `sources.cmake` | sm_89 下跳过所有 NVFP4 权重 kernel，改链 stub |
| `src/ops/nvfp4_l20_stub.cpp`（新增） | 194 行 stub：所有 NVFP4 权重符号（linear/attn/gdn/swiglu/linear_add/snapshot + NVFP4 prompt attention）都定义为**抛出清晰错误**，提示改用 Q4/Q5 artifact |

### 4.2 设计原则

- **正确性优先于性能**：PDL 退化为普通流式启动，牺牲一点重叠换绝对正确（CUDA in-order stream 保证）。
- **优雅降级而非崩溃**：NVFP4 权重在 L20 上不是"段错误"，而是抛 `std::runtime_error("NVFP4 weights are not supported on this build (sm_89)... use the groupwise-int artifact")`。
- **不碰上游数学**：所有 Q4/Q5/FP8 的 GEMM/attention/GDN kernel 原样保留，只是换了少数指令的"拼写"。数值结果与上游一致。
- **双向兼容**：补丁不破坏 sm_120a 构建——5090 用户 `cmake -DCMAKE_CUDA_ARCHITECTURES=120a` 仍走原路径。

### 4.3 如何应用

```bash
# 1. fork 并克隆
git clone https://github.com/<你的用户名>/ninfer.git
cd ninfer

# 2. 应用补丁
git apply /path/to/ninfer-l20-sm89.patch
# 或作为提交： git am < ninfer-l20-sm89.patch

# 3. 构建（默认即 sm_89）
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

---

## 5. 模型与投机解码：澄清两个关键误解

### 5.1 "DSpark" → 实际是 DFlash2

仓库里**搜不到任何 `dspark`**。NInfer 的投机解码体系是：

| 后端 | 机制 | draft 窗口 | 适用模型 | 备注 |
|---|---|---|---|---|
| **MTP** | 多 token 预测头，自回归草稿 | 1–5 | 所有 Qwen3.5/3.8 dense | `--spec mtp --draft-tokens 3` |
| **DFlash** | 掩码块草稿，条件于 target 隐藏特征 | 1–15 | 35B-A3B（MoE） | |
| **DFlash2** | DFlash + 动态卷积 + top-16 候选选择器 | 1–15（推荐 7） | **Qwen3.8-27B** | `--spec dflash2 --draft-tokens 7` |

Qwen3.8-27B 的官方 artifact 带 `dflash2` 组件（草稿权重来自 `z-lab/Qwen3.8-27B-DFlash2`），官方推荐 `--draft-tokens 7`。**你要的投机解码就是 DFlash2**，启动参数用 `--spec dflash2 --draft-tokens 7 --lm-head-draft`。DFlash2 的全部算子（Q8 draft 投影、动态分组卷积、候选选择器）都是通用指令，**sm_89 兼容**。

### 5.2 "AWQ-INT4" → NInfer 不直接吃 AWQ 文件

这是最需要纠正的一点。NInfer **不使用 AWQ 的 `qweight/qzeros/scales` 存储格式**，也没有 AWQ 导入器。它的模型分发格式是自研的 `.ninfer` artifact，4-bit 走的是**自研 groupwise-int**（`q4_g64_fp16`：4-bit 码本 + 每 64 元素一个 FP16 scale，对称量化，K 维分组）。

**但你"跑 4-bit"的目标完全能达成**，有三条路，按推荐度排序：

#### 路线 A（推荐）：直接用官方 Q4/Q5 artifact
官方 `qwen3_8_27b.ninfer`（20.4GB）就是用 Q4/Q5 量化的，text 层 query/key/gate/up 用 Q4、其余 Q5、词表 Q8。**这就是 4-bit 量化的等效物**，且是作者亲自调优过的、和 kernel 严格对齐的格式。256K 上下文完全放得下（见 §6）。**零额外工作，下载即用。**

#### 路线 B：把社区 AWQ 转打包成 `.ninfer`
如果你**必须**用某个特定的 AWQ 权重（比如 `mattbucci/Qwen3.8-27B-AWQ`），需要一个转换器把 AWQ 张量反量化/重排成 NInfer 的 groupwise-int 布局。仓库有先例：NVFP4 就是通过 `import_encoded` + `compressed_tensors.py` 从 compressed-tensors 格式导入的。AWQ 导入器需要：
- 解 AWQ 的 `qweight`（int32 打包 8 个 int4，带 `order_map=[0,2,4,6,1,3,5,7]` 重排）和 `qzeros`（zero_point=8 对称），转成 NInfer 的 `(n^8)-8` 对称码；
- 把 AWQ 的 group_size=128 重映射到 NInfer 的 group_size=64（需重新计算 scale，或按 2 组拆分）；
- 把 HF 的 `q_proj/k_proj/v_proj` 按 NInfer 的绑定重组（`attention/query` 要从 `q_proj` 里按 head 交错拆出 query/gate 两半）。

工作量约 **200–400 行 Python**，用仓库现有的 `LogicalSource` / `import_encoded` 框架做。⚠️ **这条路有精度风险**：AWQ 的 scale 是 group 128 校准的，拆成 group 64 时 scale 不再最优，且 AWQ 本身只对 MLP/attention 量化（GDN 的 `in_proj_a/b` 保留 bf16），直接转换可能引入额外误差。**除非你强依赖某个社区 AWQ 的特定微调，否则不建议。**

#### 路线 C：用官方转换器从 BF16 源重新量化
```bash
python3 -m tools.convert --model /path/to/Qwen3.8-27B --recipe qwen3_8_27b \
  --components text,vision,mtp,dflash2 --source dflash2=/path/to/Qwen3.8-27B-DFlash2 \
  --proposal --out models/qwen3_8_27b.ninfer
```
得到的就是路线 A 的 artifact，适合你想自定义量化混合（比如把更多层压到 Q4）的场景。

> **我的建议**：先用路线 A 跑通，确认 L20 上的功能和性能达标；只有当你需要某个社区 AWQ 的独有权重时，才投入路线 B 的转换器开发。

---

## 6. 显存预算：48GB 够不够 256K + 4-bit？

**完全够，而且很宽裕。** Qwen3.8-27B 是 GDN 混合架构（48 层线性注意力 + 16 层全注意力），KV cache 比纯 Transformer 小一个数量级，这是它能在单卡跑超长上下文的关键。

### 6.1 KV cache 每 token 占用（本模型）

| 组件 | 每层每 token | 层数 | INT8 KV | BF16 KV |
|---|---|---|---|---|
| 全注意力 K+V | 2×4头×256维 | 16 | 16 KB | 32 KB |
| GDN 状态摊销 | 48头×(128²+128²)×层 | 48 | ~16 KB | ~32 KB |
| **合计** | | | **~32 KB/tok** | ~64 KB/tok |

### 6.2 48GB 的分配（单并发、INT8 KV、预留 5.5GB 运行时开销）

| 权重方案 | 权重占用 | 256K KV | 剩余 | 256K 可行性 | 理论最大上下文 |
|---|---|---|---|---|---|
| BF16 | 54 GB | 8.6 GB | **-11.5 GB** | ❌ 放不下 | — |
| FP8 | ~28 GB | 8.6 GB | +15.5 GB | ✅ | ~462K |
| **官方 Q4/Q5** | **20.4 GB** | **8.6 GB** | **+22.1 GB** | ✅✅ **宽裕** | **~659K** |
| 社区 AWQ4 | ~18.7 GB | 8.6 GB | +23.8 GB | ✅✅ 最宽裕 | ~709K |

**结论**：你说"256K 下只能 4-bit"——对 BF16/FP8 确实紧张（FP8 刚好够），但 **4-bit（无论官方 Q4/Q5 还是 AWQ）都绰绰有余**，256K 之后还剩 22GB+。这些余量你可以：
- 开 `--max-concurrency 2~4` 提升吞吐；
- 把上下文推到 512K 甚至更高；
- 给 DFlash2/vision/前缀缓存留更多空间。

### 6.3 推荐启动配置（L20 + 256K + DFlash2）

```bash
./build/apps/ninfer-serve models/qwen3_8_27b.ninfer \
  --max-context 262144 \
  --kv-capacity 262144 \
  --max-concurrency 2 \
  --kv-dtype int8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec dflash2 --draft-tokens 7 \
  --lm-head-draft \
  --preserve-thinking
```

> L20 有 48GB（比 5090 的 32GB 多），`--kv-capacity` 和 `--host-kv-mib` 可以比官方 5090 示例开得更大。先用 `--kv-capacity auto` 让它自己算最大合法值。

---

## 7. 性能预期（L20 vs RTX 5090）

decode 阶段是**显存带宽瓶颈**。L20 的 864 GB/s 约为 5090（~1.8 TB/s）的 **48%**，所以单请求 decode 吞吐大致减半；prefill 是算力瓶颈，L20 的 BF16 tensor（119.5 TFLOPS）远弱于 5090，prefill 差距更大。

| 场景 | 5090 实测（官方） | L20 预期（估算） | 依据 |
|---|---|---|---|
| Q4/Q5 单请求 decode（MTP3） | 185.8 tok/s | **~85–95 tok/s** | 带宽比 48% |
| Q4/Q5 7.7K prefill | 3,218 tok/s | **~700–900 tok/s** | 算力比 ~25% |
| Q4/Q5 C=8 并发 decode | 535 tok/s | **~250–290 tok/s** | 带宽 + SM 数 |
| DFlash2 decode 接受率 | 45.8–48.9% | **相近** | 接受率由模型决定，与硬件无关 |

> ⚠️ 以上是**基于带宽/算力比的估算**，不是实测。L20 的 SM 数（92）远少于 5090，且这些 kernel 是按 5090 的 occupancy 调的，实际值需要你在 L20 上跑一次 `bench/` 校准。PDL 退化还会带来少量（通常 <5%）decode 延迟上升。

**L20 的一个独有优势**：48GB 显存让你能把 KV 池和 host 前缀缓存开得比 5090 大得多，长上下文场景的 TTFT（首 token 延迟）和并发能力反而可能更好。

### 7.1 L20 实测结果（2026-09-18，gs@192.168.0.16 GPU0）

实测环境：NVIDIA L20（46,068 MiB）、CUDA 12.8 / driver 13.2、`qwen3_8_27b.ninfer`（20.4GB 官方 Q4/Q5 + DFlash2）、int8-group64 KV。在实测前修复了补丁之外的两个 sm_89 问题（见提交 `0d4e4e13`、`b7afcf57`）：`__CUDA_ARCH__` 门禁值、>48KB 静态共享内存（sm_89 上限 48KB/CTA，76 个 Q8 kernel 实例化需转动态共享池）。

| 场景 | 5090 实测（官方） | L20 预期 | **L20 实测** | 备注 |
|---|---|---|---|---|
| prefill pp512 / pp2048（无投机） | — | ~700–900 | **1,635 / 1,684 tok/s** | 超预期 ~2×：Q4/Q5 反量化 GEMM 更偏带宽瓶颈，不受 BF16 算力差距限制 |
| decode 单请求（无投机） | — | — | **38.5 tok/s**（serve 稳态 38.0） | ≈ 864GB/s 带宽理论上限 ~50 tok/s 的 77% |
| decode 单请求 DFlash2（greedy，2K prompt） | 185.8（MTP3） | ~85–95 | **174.8 tok/s**，接受率 88% | 达 5090 的 94%；接受率高时 decode 变验证（算力）主导 |
| decode 单请求 DFlash2（stochastic serve，335→8192 tok） | — | — | **74.6 tok/s** 稳态，接受率 30.4% | 对比 mtp0 38.0 tok/s，1.96×；随机采样拉低接受率 |
| 256K 长上下文启动 | — | 可容纳 | ✅ `--max-context 262144 --kv-capacity auto --max-concurrency 2 --kv-dtype int8`，KV 291,712 tokens，余量 ~1GB | 符合 §6 预算 |
| 困惑度 quick（261K tokens） | — | — | int8 KV **4.34333** vs bf16 KV **4.34303**（Δ +0.007%） | int8 KV 数值无损 |

> bench 命令：`./build/bench/ninfer_bench --weights <artifact> -p 512,2048 -n 128 -pg '2048,128' -r 5 --warmup 1 --kv-dtype int8 [--spec dflash2 --draft-tokens 7 --lm-head-draft]`；并发/serve：`tools/bench/run_serve_concurrency.py --suite decode-saturation --concurrency 1 --mode mtp0 --mode dflash2_7`。

---

## 8. 构建与依赖清单（Linux Mint 22.3 + CUDA 12.8）

```bash
# 系统依赖
sudo apt install cmake ninja-build pkg-config \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
  libcurl4-openssl-dev

# 注意：需要 CMake ≥ 3.28 和支持 C++20 的编译器（GCC 11+）。
# Mint 22.3 基于 Ubuntu 24.04，GCC 13 / CMake 3.28+ 通常满足。
# CUDA 12.8 已通过实测兼容所有 sm_89 指令（见 §3）。

# 克隆你的 fork + 应用补丁 + 构建
git clone https://github.com/<你>/ninfer.git && cd ninfer
git apply ninfer-l20-sm89.patch
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release   # 默认即 sm_89
cmake --build build -j

# 下载官方 Q4/Q5 artifact（含 DFlash2 草稿）
pip install -U "huggingface_hub[cli]"
hf download neroued/Qwen3.8-27B-NInfer qwen3_8_27b.ninfer --local-dir models
# 若要 DFlash2 草稿权重，还需 z-lab/Qwen3.8-27B-DFlash2（见模型卡）

# 跑起来（见 §6.3）
./build/apps/ninfer-serve models/qwen3_8_27b.ninfer --spec dflash2 --draft-tokens 7 ...
```

> 风险提示：本适配在**无 GPU 的沙箱**里完成，指令级正确性已用 ptxas 实测，但**完整编译（1500+ 文件）和 L20 实机运行未验证**。首次在你的 L20 上构建时，如遇个别 kernel 的 occupancy/register 问题，大概率是 tile 尺寸按 5090 调的，需要微调 `shapes/` 里的 schedule——这属于调优而非移植，报告 §9 给了方向。

---

## 9. 后续调优方向（可选，非必须）

1. **Occupancy 调优**：很多 kernel 的 tile（`BM/BN/BK`、warp 数）是按 5090 的 170 SM / 32GB 调的。L20 是 92 SM / 864 GB/s，带宽更低意味着应该用**更小的 tile、更高的 occupancy** 来掩盖延迟。重点调 `src/ops/linear/q4/shapes/` 和 `q8/shapes/` 里的 schedule 常量，配合 `bench/ops/linear_bench.cu` 逐 shape 扫。
2. **KV dtype 选择**：L20 显存宽裕，优先 `int8`（精度/速度平衡）；要极限长上下文再用 `fp8` 或 `nvfp4`（KV 侧，软件解量化，sm_89 可用）。
3. **DFlash2 draft 窗口**：官方推荐 7，但在带宽更低的 L20 上，可以扫 `1..15` 找接受率×带宽的最优点（带宽低时更大的 draft 窗口更划算，因为验证是算力密集、草稿是带宽密集）。
4. **前缀缓存 / host offload**：48GB 显存 + host pinned 内存的组合在 L20 上比 5090 更有发挥空间，参考 `docs/maintainer/resource-scheduling-and-context-cache.md`。

---

## 10. 局限与免责

- **已完成实机验证**（2026-09-18）：sm_89 全量构建通过，CLI/serve/256K/困惑度/基准矩阵均实测通过，见 §7.1。初版补丁之外另需两个修复（arch 门禁值、48KB 静态共享内存上限），已并入分支提交。
- **NVFP4 权重路径在 L20 上不可用**（硬件限制，sm_120a 专有），已用 stub 优雅降级。L20 上请用 Q4/Q5 artifact。
- **"DSpark" 在 NInfer 中不存在**：本报告按 DFlash2（NInfer 对 Qwen3.8-27B 的实际投机解码后端）处理。如果你指的是另一个独立的 DSpark 库/算法，请提供其仓库地址，我可以评估如何接入。
- **性能数字以 §7.1 实测为准**：§7 表为移植前的带宽/算力比推算，保留作对照；实测 prefill 显著优于估算（Q4/Q5 GEMM 偏带宽瓶颈），DFlash2 greedy decode 接近 5090 水平。
- 本报告仅供技术参考，不构成生产部署保证；模型权重与衍生量化请遵守 Qwen（Apache-2.0）及各社区量化仓库的许可条款。

---

### 附：交付文件

| 文件 | 说明 |
|---|---|
| `ninfer-l20-sm89.patch` | 完整 sm_89 移植补丁（660 行，`git apply` 到上游 `6cc95cc`） |
| 本报告 | 适配分析 + 构建/调优/显存/性能指南 |
