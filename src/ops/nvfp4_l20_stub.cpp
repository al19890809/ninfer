// NINFER_L20 / sm_89 (Ada) port — NVFP4 / TMA-MoE weight-op stubs.
//
// sm_89 has no hardware NVFP4 path, and the upstream FP8/NVFP4 MoE expert
// kernels are TMA-based (sm_90+), so every affected translation unit is
// excluded from this build (see NINFER_NO_NVFP4_WEIGHTS in
// src/ops/CMakeLists.txt). The dispatchers that would route to those kernels
// are still compiled; this file provides throwing definitions for the symbols
// they reference, so any such request on an L20 build fails fast with a clear
// error instead of a link error.

#include <stdexcept>

#include "ops/linear/kernels/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear/add/nvfp4/dispatch.h"
#include "ops/linear/split/nvfp4/shapes/n14336_k5120/launch.h"
#include "ops/linear/split/nvfp4/shapes/n16384_k5120/launch.h"
#include "ops/linear_attention/gated_delta_net/input_conv/nvfp4/tma_launch.h"
#include "ops/linear/projection/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/swiglu/nvfp4/tma_launch.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/plan.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/launch.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/launch.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/plan.h"
#include "ops/moe/kernels/cuda/block_scaled/fp8/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/nvfp4/launch.h"

namespace ninfer::ops::detail {

[[noreturn]] static void nvfp4_unsupported() {
    throw std::runtime_error(
        "NVFP4 operation is not supported in this build (NINFER_L20 / sm_89 port: "
        "NVFP4 kernels excluded, stub called)");
}

[[noreturn]] static void fp8_moe_unsupported() {
    throw std::runtime_error(
        "FP8 block-scaled MoE is not supported in this build (NINFER_L20 / sm_89 port: "
        "the upstream FP8 expert kernels are TMA-based, sm_90+ only)");
}

// ---- linear: nvfp4 a4 quantize + TMA entry points -------------------------

void launch_nvfp4_a4_quantize(const Tensor&, const Weight&, Nvfp4A4Workspace, Nvfp4ScaleLayout,
                              cudaStream_t) {
    nvfp4_unsupported();
}
void launch_nvfp4_a4_tma_gdn(const Nvfp4A4Operands&, __nv_bfloat16*, __nv_bfloat16*, cudaStream_t) {
    nvfp4_unsupported();
}
void launch_nvfp4_add_n5120_k6144_tma(const Tensor&, const Weight&, Tensor&, Nvfp4A4Workspace,
                                      cudaStream_t) {
    nvfp4_unsupported();
}
void launch_nvfp4_add_n5120_k17408_tma(const Tensor&, const Weight&, Tensor&, Nvfp4A4Workspace,
                                       cudaStream_t) {
    nvfp4_unsupported();
}
namespace linear_split::nvfp4_n14336_k5120 {
void launch_a4_tma(const Nvfp4A4Operands&, std::span<const Tensor>, cudaStream_t) {
    nvfp4_unsupported();
}
} // namespace linear_split::nvfp4_n14336_k5120
namespace linear_split::nvfp4_n16384_k5120 {
void launch_a4_tma(const Nvfp4A4Operands&, std::span<const Tensor>, cudaStream_t) {
    nvfp4_unsupported();
}
} // namespace linear_split::nvfp4_n16384_k5120

// ---- linear projection / swiglu nvfp4 entries (host dispatchers) ---------

std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t, std::int32_t, LinearPolicy,
                                                  std::int32_t, std::int32_t) {
    nvfp4_unsupported();
}
void nvfp4_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy, MemoryArena*,
                    cudaStream_t) {
    nvfp4_unsupported();
}
void launch_nvfp4_swiglu_a4_tma(const Nvfp4A4Operands&, __nv_bfloat16*, cudaStream_t) {
    nvfp4_unsupported();
}

// ---- dense causal attention: NVFP4 / K8V4 KV plans + kernels --------------

std::size_t nvfp4_kv_workspace_bytes(int, int, int, int, CausalAttentionExecutionEnvelope, int) {
    nvfp4_unsupported();
}
std::size_t k8v4_kv_workspace_bytes(int, int, int, int, CausalAttentionExecutionEnvelope, int) {
    nvfp4_unsupported();
}
void nvfp4_kv_append_attention(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                               const Tensor&, const Tensor&, float, PagedKVBatchLayerView,
                               CausalAttentionExecutionEnvelope, MemoryArena&, Tensor&,
                               CudaExecutionView) {
    nvfp4_unsupported();
}
void nvfp4_kv_cached_attention(const Tensor&, const Tensor&, float, const PagedKVLayerView&,
                               CausalAttentionExecutionEnvelope, MemoryArena&, Tensor&,
                               CudaExecutionView) {
    nvfp4_unsupported();
}
void k8v4_kv_append_attention(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                              const Tensor&, const Tensor&, float, PagedKVBatchLayerView,
                              CausalAttentionExecutionEnvelope, MemoryArena&, Tensor&,
                              CudaExecutionView) {
    nvfp4_unsupported();
}
void k8v4_kv_cached_attention(const Tensor&, const Tensor&, float, const PagedKVLayerView&,
                              CausalAttentionExecutionEnvelope, MemoryArena&, Tensor&,
                              CudaExecutionView) {
    nvfp4_unsupported();
}

// ---- MoE block-scaled experts (TMA-based, sm_90+) -------------------------

namespace moe_cuda::block_scaled::fp8 {
void launch_input(const Tensor&, const CudaSwiGLUExperts&, const Storage&, CudaExecutionView,
                  bool) {
    fp8_moe_unsupported();
}
void launch_experts(const Tensor&, const CudaSwiGLUExperts&, const ExpertRoutes&, const Plan&,
                    const Storage&, CudaExecutionView) {
    fp8_moe_unsupported();
}
void launch_vector(const Tensor&, const CudaSwiGLUExperts&, const ExpertRoutes&, ExpertSubset,
                   bool, MoeEpilogue, Tensor&, const Storage&, CudaExecutionView) {
    fp8_moe_unsupported();
}
} // namespace moe_cuda::block_scaled::fp8

namespace moe_cuda::block_scaled::nvfp4 {
void launch_input(const Tensor&, const CudaSwiGLUExperts&, const Storage&, CudaExecutionView,
                  bool) {
    nvfp4_unsupported();
}
void launch_experts(const Tensor&, const CudaSwiGLUExperts&, const ExpertRoutes&, const Plan&,
                    const Storage&, CudaExecutionView) {
    nvfp4_unsupported();
}
void launch_vector(const Tensor&, const CudaSwiGLUExperts&, const ExpertRoutes&, ExpertSubset,
                   bool, MoeEpilogue, Tensor&, const Storage&, CudaExecutionView) {
    nvfp4_unsupported();
}
} // namespace moe_cuda::block_scaled::nvfp4

} // namespace ninfer::ops::detail
