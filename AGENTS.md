# AGENTS.md

## Product and architecture

NInfer is a C++/CUDA inference engine for maximum single-GPU performance, targeting `sm_120a`
on NVIDIA GeForce RTX 5090. Choose designs for functional and numerical correctness, clear
ownership, and the performance goal within the requested scope.

It implements `Qwen3_5ForCausalLM`, `Qwen3_5MoeForCausalLM`, and Text/Vision/MTP for
`Qwen4ExpForCausalLM`. Checkpoints and recipe combinations
of existing representations use the same architecture, binding and execution path; do not add
checkpoint-specific execution registration. Generation uses one GPU, one resident model, and
one to eight resident execution lanes fixed at startup.

V3 `.ninfer` is the only C++ product artifact. Generation, offline CausalScoring, CLI, serving, and
inference benchmarks use the public Engine. There is no Python model-inference route or
installed/exported C++ SDK.

This is a local, single-owner project with trusted local models, artifacts, and workflows.
Do not derive requirements from another deployment or trust model.

Model data is immutable; every Program owns its mutable state and device allocations. The loader
validates, uploads, and binds the stored representation; actual Parameters and Ops determine
execution support, without a whole-artifact capability registry.
Runtime owns execution and publication policy; product/serving own input acquisition and protocol
translation. Model code does not acquire media or own transport.

Keep architecture, binding, and execution explicit. Without a product requirement, do not introduce
generic model graphs, family base classes, plugin discovery, string-driven execution, hidden device
allocation, or runtime weight repacking. New mathematical architectures, execution platforms,
large-scale continuous batching, or priority/QoS require an explicit product change. When a task
changes product or architecture, update affected contracts and implementation together.

## Change policy

Project-owned contracts do not preserve backward compatibility. Replace behavior completely:
remove superseded aliases, fallbacks, transition branches, and their tests within the affected contract.
Advertised OpenAI and Anthropic protocols are external contracts; update affected schema tests
and serving documentation together.
Update stable requirements in their existing authoritative document; maintain one current authority.
Use Conventional Commit subjects with concise lowercase types when a commit is requested.

## Verification and reporting

Before changing Ops, numerical or state semantics, read
[Op development](docs/maintainer/op-development.md). Qualify affected production routes directly
against the contract's independent mathematical oracle or specified exact reference. Kernel parity
and plausible model output do not establish mathematical correctness.

Measure performance at the claimed scope. An Op microbenchmark establishes an Op result,
not an end-to-end improvement. For comparisons, establish the baseline, workloads, metrics,
aggregation, and acceptance criteria before evaluating results.
Distinguish new capability, fallback replacement, and improvement to an optimized implementation.
Report the baseline, hardware/toolchain, workloads or commands, run conditions, metrics, coverage,
result distribution, worst changes, and exceptions. Include small and unexplained regressions;
do not dismiss slowdowns as noise without evidence.

## References

Read [Engine architecture](docs/maintainer/engine-architecture.md) before changing execution or
ownership. [README](README.md) and executable `--help` define capabilities and exact commands.
The [documentation map](docs/README.md) routes to detailed contracts;
[Tests](tests/README.md) and [Benchmarks](bench/README.md) own their commands and execution details.

## Local environment

Use Python 3.11 via `/home/neroued/miniconda3/envs/py311/bin/python` explicitly; use `python3` only
after selecting the maintainer environment or checking its version.
The usual local model is `out/qwen3_8_27b_nvfp4.ninfer`. Select artifacts by explicit path, never glob order,
modification time, or unqualified “latest”. Source checkpoints and large artifacts are prerequisites;
download or regenerate them only when that work is in scope.

| Change | Typical evidence |
|---|---|
| Documentation | affected links/references and `git diff --check` |
| C++ runtime/API | affected build targets and behavioral tests |
| Python tooling | Python 3.11 `py_compile` and affected tests |
| Artifact framing/binding/conversion | affected contract tests; real artifact when semantics require it |
| CUDA mathematics | independent oracle at relevant shapes and route boundaries |
| Memory or lifetime | affected execution; sanitizer for a concrete lifetime question |
| Performance | measurement at the claimed scope; profiling only for unresolved attribution |
| Serving | affected schema tests and observable request/stream behavior |

Record the target, relevant hardware/toolchain, workload or command, and summarized result needed
to interpret a material claim. Hashes, clean worktrees, full command transcripts, raw report
inventories, and exact probabilistic outputs are not default requirements. Use exact comparison for
exact outputs, and appropriate numerical or behavioral criteria otherwise. State checks that could
not run and their implications.

Finish when the deliverable is usable, applicable contracts are satisfied, material claims have
sufficient evidence, relevant checks pass or their limitations are clear, and no known in-scope
issue blocks use. Expand or repeat verification only for new changes, failures, or unresolved risks
that could change the result. Supporting work is not an independent completion objective.

## Reference navigation

Read the authority relevant to the current decision; this is not a mandatory reading list.

| Decision | Entry point |
|---|---|
| Product capabilities and exact commands | `README.md`, executable `--help`; `docs/cli.md`, `docs/serving.md`, `docs/perplexity.md` |
| Execution, model/runtime ownership, scheduling, transactions, graphs | `docs/maintainer/engine-architecture.md` |
| Context resources, checkpoints, replicas; physical KV | `docs/maintainer/resource-scheduling-and-context-cache.md`; `docs/maintainer/paged-kv-cache.md` |
| Artifact, layout, codec, conversion, or model mathematics | model/artifact references and conversion guide linked from `docs/README.md` |
| Op contracts, implementation ownership, numerical/performance qualification | `docs/maintainer/op-development.md` |
| Test/benchmark commands and published performance | `tests/README.md`, `bench/README.md`, `docs/performance.md` |
| In-tree C++ interface | `include/ninfer/engine.h`, `include/ninfer/types.h` |

[Documentation map](docs/README.md) routes to narrower authorities when needed.

## Local operations

Use `cmake --build <build-dir> -j` by default. Adjust parallelism when actual resource pressure
causes failures or interferes with the task, and briefly explain why.

Use the selected Python 3.11 interpreter explicitly. On this machine it is
`/home/neroued/miniconda3/envs/py311/bin/python`; the default shell's `python3` may be a different
version. Use `python3` only after selecting the maintainer environment or checking its version.
Normal resources are `build/`, `out/qwen3_6_27b.ninfer`, its `.conversion.json` report, and
`profiles/ncu/`, `profiles/nsys/`, `profiles/bench/`; the local toolchain is CUDA 13.1.
Select model artifacts by explicit path, never glob order, modification time, or unqualified
“latest”. Source checkpoints and large artifacts are prerequisites; download or regenerate them
only when that work is in scope. Install or upgrade dependencies only when the task needs it.

Create commits only when requested. Use Conventional Commit subjects with concise lowercase types
such as `feat`, `fix`, `perf`, `bench`, `test`, `build`, `refactor`, `docs`, or `chore`.

## Remote deployment (gs@192.168.0.16 — NVIDIA L20)

### Hardware and toolchain
- GPU: NVIDIA L20 (sm_89, Ada architecture, 46,068 MiB, 864 GB/s)
- CUDA: 12.8, driver 13.2, nvcc at `/usr/local/cuda-12.8/bin/nvcc`
- Python: system python3 3.12.3 (stdlib-only tools; no conda needed)
- Systemd-managed services: `vllm-30000.service` (GPU0), `ninfer-30002.service` (GPU1)

### Build configuration
```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build -j128
```

### Production service (ninfer-30002)
- Unit file: `/etc/systemd/system/ninfer-30002.service`
- Model: `/home/gs/models/qwen3_8_27b.ninfer` (Qwen3.8-27B official Q4/Q5 + DFlash2, 20.4 GB)
- Log: `/data/model-logs/ninfer-30002-gpu1.log` (same naming convention as vllm)
- Current config: `--max-context 204800 --max-concurrency 2 --kv-dtype bf16 --spec dflash2 --draft-tokens 3 --lm-head-draft --vision --temperature 0.6 --presence-penalty 0.0 --pending-timeout-ms 300000`
- Management: `sudo systemctl start|stop|restart|status ninfer-30002`

### L20 port (sm_89 branch)
- Branch: `ninfer-L20` on Gitee (`https://gitee.com/al19890809/ninfer`)
- Canonical commits: `52b7c3de` (port) → `0d4e4e13` (arch guard fix) → `b7afcf57` (smem fix) → `35a18fd9` (patches commit)
- Critical sm_89 constraints: static shared mem cap = 48KB (0xc000), enforced at nvlink; dynamic pool up to ~99KB with `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`
- Dual-path pattern for all Q8 kernels: `constexpr bool kDynamicShared = sizeof(Storage) > kStaticSharedLimitBytes; __shared__ unsigned char static_shared[kDynamicShared ? 1 : sizeof(Storage)]; extern __shared__ unsigned char dynamic_shared[];`
- Patch directory `patches/` contains deployment artifacts: `ninfer-l20-sm89.patch` (initial port) and `ninfer-l20-sm89-full.patch` (cumulative, includes all fixes). The full patch is required for deployment from clean source.

### Deployment from Gitee
```bash
git clone https://gitee.com/al19890809/ninfer.git
cd ninfer
git apply patches/ninfer-l20-sm89-full.patch
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build -j128
```

### Performance characteristics (L20, Qwen3.8-27B, bf16 KV)
| Scenario | Throughput | Notes |
|---|---|---|
| prefill pp512 / pp2048 | 1,625 / 1,672 tok/s | Q4/Q5 GEMM is bandwidth-bound, not compute-bound |
| decode (no spec) | 38.5 tok/s | ≈ 864 GB/s bandwidth ceiling × 77% |
| decode DFlash2 (w3, temp=0.6) | 75.5–80.5 tok/s | Acceptance ~49%, draft-tokens 3 optimal |
| decode DFlash2 (w7, temp=1.0) | 63.0 tok/s | Acceptance ~23%, window 7 suboptimal at temp=1.0 |
| perplexity quick (261K tokens) | int8 4.34333 vs bf16 4.34303 | Δ +0.007%, int8 KV numerically lossless |

Key insight: L20's compute (BF16 119 TFLOPS) is weaker than 5090 (825 TFLOPS), but decode is bandwidth-bound on both. The advantage is memory (46 GB vs 24 GB), enabling larger KV pools and longer contexts. Speculative decoding acceptance is strongly temperature-dependent: temp=1.0 → ~23%, temp=0.6 → ~30%, greedy → 88%. For code generation, temperature ≤ 0.6 is recommended.

### GPU topology (gs@192.168.0.16)
| GPU | Service | Memory |
|---|---|---|
| GPU0 | vllm-30000 (Qwen3.6-35B-A3B-AWQ) | ~41.6 GB |
| GPU1 | ninfer-30002 (Qwen3.8-27B) | ~43.6 GB |

Services are mutually exclusive per GPU. Switching models requires stopping the other service first.
Store all agent-generated temporary files under the repository-root `.agent-tmp/<task>/`,
including temporary plans, experimental code and binaries, test data, logs, profiling captures,
performance reports, and intermediate artifacts. This directory is ignored by Git; do not commit
its contents. Remove temporary files that are no longer needed when the task ends.
Reserve `out/` for model weights, their companion files, and explicitly retained important artifacts.
Do not use `out/` for temporary experiments or task output.
