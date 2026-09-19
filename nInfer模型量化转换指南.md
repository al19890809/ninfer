# nInfer 模型量化转换指南

> 本文以「Qwen3.8-27B 无审查版 → nInfer 4bit（groupwise-int）artifact」为例，记录完整转换流程与全部踩坑。将来转换其他模型时，替换对应的模型路径、recipe 名称和组件来源即可复用。

## 0. 背景知识

### nInfer 是什么

- 从零编写的 C++/CUDA 单卡推理引擎，专为**单张 RTX 5090（sm_120a）**调优，构建时拒绝其他架构。
- 使用私有 `.ninfer` 单文件格式，打包模型配置、量化权重、逻辑绑定和前端资源（聊天模板等）。
- 官方仅发布少数 artifact（Qwen3.6/3.8-27B、Qwen3.6-35B-A3B 的 groupwise-int / nvfp4 版），但**提供转换器 `tools/convert`，支持自定义权重 + 官方 recipe**。

### 关键架构事实

- **转换与运行可以分离**：转换器是纯 Python 工具（独立于 CMake 构建），可在任意带 CUDA 的机器上跑（如 L20）；运行时才要求 sm_120a（5090）。
- **组件模型**：一个 artifact 可打包多个组件，运行时按需加载：
  | 组件 | 权重来源 | 说明 |
  |---|---|---|
  | `text` | 主模型 checkpoint | 文本主体 |
  | `vision` | 主模型 checkpoint | 视觉塔，多模态输入 |
  | `mtp` | 主模型 checkpoint | MTP 投机解码草稿头（窗口 1–5） |
  | `dflash2` | **独立伴随权重**（需单独下载） | 块扩散草稿（窗口 1–15），加速上限更高 |

- **草稿头（MTP/DFlash2）只提议候选 token，最终输出由主模型验证**。因此草稿头用官方权重不影响主模型的"无审查"等特性，只影响接受率（速度）。

### recipe 选择（重要）

| recipe | 量化方案 | 适用 |
|---|---|---|
| `qwen3_8_27b` | groupwise-int：Q4/Q5 投影 + Q8 词表 | **自定义 BF16 源的唯一选择** |
| `qwen3_8_27b_nvfp4` | 导入预量化 NVFP4 权重 | 要求 `--source quantized=` 提供已量化的 NVFP4 checkpoint；**自定义权重不能用**（会把层换回原版权重） |

recipe 是 `tools/convert/official_recipes.py` 里的普通 Python 函数，可通过自定义 recipe 文件或 `--override` 微调个别层的格式。

## 1. 材料准备

### 1.1 主模型权重（必须 Safetensors 全精度）

要求：**BF16/FP16 Safetensors checkpoint，含完整 text + vision + mtp 权重**。GGUF 等已量化格式**不能**作为源（有损且无法回退）。

获取途径（按优先级）：

1. **现成的微调/无审查版 HF 仓库**（本文例：`orcarouter/Qwen3.8-27B-Uncensored`，门控 auto 型，网页点同意即时开通）；
2. **官方仓库**（ModelScope 对国内友好、通常无门控，如 `Qwen/Qwen3.8-27B`）；
3. **自己改造**：用 [heretic](https://github.com/p-e-w/heretic) 对官方权重做自动化消融（ARA），输出 Safetensors：
   ```bash
   pip install -U heretic-llm
   heretic Qwen/Qwen3.8-27B   # 显存不够可加 --quantization bnb_4bit 或走 CPU+大内存
   ```

下载命令（国内建议镜像 + token）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
HF_TOKEN=hf_xxx hf download "orcarouter/Qwen3.8-27B-Uncensored" \
  --local-dir /data/models/Qwen3.8-27B-Uncensored
```

> 门控仓库必须带 token；`hf-mirror.com` 负责加速，token 负责鉴权，两者可同时用。

### 1.2 DFlash2 伴随权重（可选组件）

**不在 Qwen 官方仓库**（ModelScope Qwen 命名空间下 404）。发布方是 Inco AI / Z-Lab（2026-08，Apache 2.0，约 1.92B 参数 / 3.85GB）：

```bash
# ModelScope（国内推荐）
modelscope download --model "z-lab/Qwen3.8-27B-DFlash2" \
  --local_dir /data/models/Qwen3.8-27B-DFlash2

# 或 HuggingFace
hf download "incoai/Qwen3.8-27B-DFlash2" --local-dir /data/models/Qwen3.8-27B-DFlash2
```

### 1.3 nInfer 源码

```bash
git clone https://github.com/Neroued/ninfer.git /home/gs/ninfer
# 聊天模板仓库自带：tools/chat_templates/qwen3_8.jinja，无需手动准备
```

### 1.4 下载后校验（必做）

```bash
python3 - <<'EOF'
import json
idx = json.load(open('/data/models/Qwen3.8-27B-Uncensored/model.safetensors.index.json'))
keys = list(idx['weight_map'])
print('tensors:', len(keys))
print('mtp   :', sum('mtp' in k.lower() for k in keys))      # 预期 15
print('vision:', sum('visual' in k.lower() for k in keys))   # 预期 333
print('总大小: %.1f GB' % (idx['metadata']['total_size']/1e9))  # 预期 ~55.6
EOF
```

若 mtp 或 vision 为 0：从官方仓库补对应组件（vision 塔和 MTP 头用官方权重无内容影响），转换时用 `--source` 挂补充源 + override recipe 指向。

### 磁盘与内存预算

| 项 | 大小 |
|---|---|
| 主模型 BF16 | ~55GB |
| DFlash2 | ~4GB |
| 输出 artifact | ~20GB |
| **合计预留** | **≥ 90GB** |

## 2. 完整转换脚本

硬件环境假设：转换机为 CUDA 12.8 驱动（r595）+ 空闲 GPU 0。脚本零修改可直接用；换模型时只需改前 4 个路径变量与 recipe 名。

```bash
#!/bin/bash
# ============================================================
# Qwen3.8-27B-Uncensored → nInfer v3 (groupwise-int 4bit)
# 组件: text + vision + mtp + dflash2
# ============================================================
set -e

MODEL_DIR=/data/models/Qwen3.8-27B-Uncensored
DFLASH2_DIR=/data/models/Qwen3.8-27B-DFlash2
NINFER_DIR=/home/gs/ninfer
OUT=/data/models/qwen3_8_27b_uncensored.ninfer
LOG=/data/models/convert_$(date +%Y%m%d_%H%M%S).log

# ---------- 0. 输入校验 ----------
echo "=== [0] 校验输入文件 ==="
for d in "$MODEL_DIR" "$DFLASH2_DIR" "$NINFER_DIR"; do
  [ -d "$d" ] || { echo "缺失目录: $d"; exit 1; }
done

python3 - "$MODEL_DIR" <<'EOF'
import json, sys, os
d = sys.argv[1]
idx = json.load(open(os.path.join(d, 'model.safetensors.index.json')))
keys = list(idx['weight_map'])
missing = [s for s in set(idx['weight_map'].values())
           if not os.path.exists(os.path.join(d, s))]
n_mtp = sum('mtp' in k.lower() for k in keys)
n_vis = sum('visual' in k.lower() for k in keys)
print(f"tensors={len(keys)}  mtp={n_mtp}  vision={n_vis}"
      f"  总大小={idx['metadata']['total_size']/1e9:.1f}GB")
if missing:
    print("缺分片:", missing); sys.exit(1)
assert n_mtp > 0, "MTP 权重缺失！"
assert n_vis > 0, "Vision 权重缺失！"
print("主模型校验通过 (text+vision+mtp 完整)")
EOF

ls "$DFLASH2_DIR"/*.safetensors >/dev/null 2>&1 || { echo "DFlash2 safetensors 缺失"; exit 1; }
[ -f "$NINFER_DIR/tools/chat_templates/qwen3_8.jinja" ] || { echo "模板缺失"; exit 1; }
echo "DFlash2 与模板校验通过"

# ---------- 1. Python 3.11 环境 (uv) ----------
echo "=== [1] 准备 Python 3.11 环境 (uv) ==="
pip install -U uv
uv venv --clear --python 3.11 /home/gs/ninfer_convert_venv
source /home/gs/ninfer_convert_venv/bin/activate

# PyPI 默认 torch 为最新版（捆绑 CUDA 13）；驱动较老则 pin "torch==2.9.1"（捆绑 cu128）
uv pip install torch numpy --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 验证 GPU 可用（驱动不支持捆绑的 CUDA 版本时为 False，需换 torch 版本）
python3 -c "import torch; print(torch.__version__, '| 捆绑CUDA:', torch.version.cuda, '| GPU可用:', torch.cuda.is_available())"

# ---------- 2. 转换器依赖 ----------
echo "=== [2] 安装转换器依赖 ==="
cd "$NINFER_DIR"
[ -f requirements.txt ] && pip install -r requirements.txt || true
[ -f tools/requirements.txt ] && pip install -r tools/requirements.txt || true
export PYTHONPATH="$NINFER_DIR:${PYTHONPATH:-}"

# ---------- 3. 转换 ----------
echo "=== [3] 开始转换，日志: $LOG ==="
CUDA_VISIBLE_DEVICES=0 python3 -m tools.convert \
  --model "$MODEL_DIR" \
  --recipe qwen3_8_27b \
  --source dflash2="$DFLASH2_DIR" \
  --components text,vision,mtp,dflash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal \
  --name qwen3.8-27b-uncensored \
  --out "$OUT" 2>&1 | tee "$LOG"

# ---------- 4. 验收 ----------
echo "=== [4] 验收 ==="
if [ -f "$OUT" ]; then
  ls -lh "$OUT"
  sha256sum "$OUT"
  grep -icE "skip|missing" "$LOG" | xargs -I{} echo "skip/missing 出现次数: {}"
  echo "=== 完成: $OUT ==="
else
  echo "未找到输出文件，最后 50 行日志："
  tail -50 "$LOG"
  exit 1
fi
```

使用方式：

```bash
nano /home/gs/convert_ninfer.sh   # 粘贴保存
nohup bash /home/gs/convert_ninfer.sh > /home/gs/convert_run.log 2>&1 &
tail -f /home/gs/convert_run.log
```

参考耗时：L20 上 27B 全量转换约 **10 分钟**（599.4s），产出 20GB artifact（1190 objects）。

## 3. 踩坑记录（FAQ）

| 症状 | 原因 | 解法 |
|---|---|---|
| `conda: command not found`（nohup 脚本内） | 交互 shell 的 conda 来自 .bashrc，非交互 shell 不加载 | 用 uv 替代 conda 管理 Python 版本（本指南方案），或把 conda.sh 的绝对路径写死 |
| `A virtual environment already exists` | uv venv 默认拒绝覆盖 | 加 `--clear`；uv 包缓存是全局的，重建不丢已下载的包 |
| pip/uv 下载小文件卡死、大文件正常 | 镜像源单连接抽风（pytorch.org 在国内尤其慢） | 全部走国内 PyPI 镜像（清华/阿里）；中断重跑即可，uv 缓存续传 |
| PyPI 装的 torch 带 `nvidia-*-cu13` 包 | PyPI 最新 torch 捆绑 CUDA 13 运行时 | **PyTorch 自带 CUDA 运行时，不用系统 toolkit**；只看驱动支持：`nvidia-smi` 的 CUDA Version ≥ 捆绑版本即可。验证 `torch.cuda.is_available()`，False 则 pin 旧版 torch（如 2.9.1 = cu128） |
| 转换跑到 `[1183/1184]` 长时间不动 | 正在量化 proposal 头（131072×5120，约 6.7 亿参数的单张量 + token 排序），是全流程最重的单步 | 正常现象，等 5–15 分钟；用 `nvidia-smi`/`top` 确认进程在干活 |
| `Access denied. This repository requires approval` | HF 门控仓库，匿名下载被拒 | 网页登录后点同意（auto 型即时开通），再 `HF_TOKEN=xxx hf download` |
| 多卡机器转换碰了正在服务的 GPU | PyTorch 默认用 cuda:0 | 命令前加 `CUDA_VISIBLE_DEVICES=0` 显式钉死空闲卡 |

## 4. 部署与验收

### 4.1 传输到运行机（5090）

```bash
scp /data/models/qwen3_8_27b_uncensored.ninfer user@目标机:/path/to/models/
# 拷完校验（应与本机一致）
sha256sum /path/to/models/qwen3_8_27b_uncensored.ninfer
```

### 4.2 启动参考

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_uncensored.ninfer \
  --max-context 240000 \
  --kv-capacity auto \
  --max-concurrency 1 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 16384 \
  --spec dflash2 --draft-tokens 7 \
  --preserve-thinking
```

切换已有 systemd 服务的模型指向：`sudo systemctl edit <服务名>`，置空并重写 `ExecStart=`（只换模型路径，其余参数保留），然后 `daemon-reload + restart`。

### 4.3 验收清单

| 项 | 方法 | 预期 |
|---|---|---|
| 基本对话 | curl 短消息 | 连贯回复 |
| 微调特性生效（如无审查） | 针对性提问 | 行为符合微调目标 |
| DFlash2 加速 | 长 prompt + 长生成，看响应 `timings` 与 `draft_n`/`draft_n_accepted` | tg 提升，接受率健康 |
| 长上下文 | 100K+ token 文档问答 | 不崩、引用准确 |
| 视觉 | 带图请求 | 正常 |
| MTP 对照 | `--spec mtp --draft-tokens 3` 重启重测 | 两种 spec 均可用，留快的 |

### 4.4 观测方式（与 llama.cpp 不同）

nInfer **没有 `--metrics` / Prometheus 端点**，观测数据在每个响应里：

- 响应自带 `timings`：`cache_n`（前缀命中量）、`prompt_per_second`、`predicted_per_second`；
- 投机解码请求带 `draft_n` / `draft_n_accepted`（接受率 = 两者相除）；
- 请求级开关：`timings_per_token: true`（流式逐 chunk 计时）、`return_progress: true`（prefill 进度）；
- 汇总报告在 stderr（systemd 下用 `journalctl -u <服务名>` 查看）。

## 5. 转换其他模型时的替换清单

1. **确认架构支持**：nInfer 对模型架构和 shape 组合是显式实现的（当前为 Qwen3.5 Dense/MoE 系），先查官方 artifact 列表里有没有同架构模型；
2. **换 `--model` 路径**与 `--name`；
3. **换 recipe**：`tools/convert/official_recipes.py` 里选同架构的官方 recipe（如 `qwen3_6_27b` / `qwen3_6_35b_a3b`），或基于它写 override；
4. **换聊天模板**：`tools/chat_templates/` 下对应文件，或模型自带的 `chat_template.jinja`；
5. **换组件来源**：MTP 在主模型内则保留 `mtp` 组件；DFlash/DFlash2 需该架构对应的伴随权重（`--source dflash=` / `dflash2=`），没有就去掉该组件；
6. **校验脚本中的张量计数**（mtp/vision 张量数随模型不同而不同，改用「> 0」断言即可，本指南脚本已是如此）。

## 附：关键参考文档

- 权重转换官方文档：`docs/weight-conversion.md`（nInfer 仓库）
- 容器格式：`docs/maintainer/artifact-container.md`
- DFlash 原理：`docs/maintainer/dflash.md`
- 服务接口：`docs/serving.md`
