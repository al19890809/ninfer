// NINFER_L20 / sm_89 (Ada Lovelace) port -- NVFP4 weight-kernel stubs.
//
// sm_89 has no MXF4/NVFP4 block-scale MMA (that PTX is sm_120a-only), so the
// upstream NVFP4 *weight* kernels cannot be built for the L20. This translation
// unit provides stub definitions for every NVFP4 weight symbol referenced by
// the retained (architecture-independent) dispatch layers. Each stub throws a
// clear std::runtime_error, so an NVFP4-weight artifact fails loudly at load /
// first use instead of crashing. L20 builds are expected to run the
// groupwise-int (Q4/Q5) artifacts; see the L20 port report.
//
// This file is compiled only when NINFER_L20_NO_PDL is defined (sm_89 build).

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_nvfp4_non_rdc_launch.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void nvfp4_unsupported() {
    throw std::runtime_error(
        "NVFP4 weights are not supported on this build (sm_89 / NVIDIA L20): the "
        "mxf4nvf4 block-scale MMA is sm_120a-only. Use the groupwise-int (Q4/Q5) "
        "artifact instead, e.g. qwen3_8_27b.ninfer.");
}

} // namespace

// --- nvfp4_format.h ---------------------------------------------------------
Nvfp4WeightGeometry validate_nvfp4_weight(const Weight&, const char*) { nvfp4_unsupported(); }

// --- nvfp4_dispatch.h -------------------------------------------------------
std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t, std::int32_t, LinearPolicy,
                                                  std::int32_t, std::int32_t) {
    nvfp4_unsupported();
}
void nvfp4_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy, WorkspaceArena*,
                    cudaStream_t) {
    nvfp4_unsupported();
}

// --- nvfp4_attn_input_plan.h -------------------------------------------------
std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    nvfp4_unsupported();
}
void nvfp4_attn_input_decode_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                    Tensor&, cudaStream_t) {
    nvfp4_unsupported();
}
void nvfp4_attn_input_small_t_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                     Tensor&, cudaStream_t) {
    nvfp4_unsupported();
}
void nvfp4_attn_input_w4a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                                  Nvfp4W4a4Workspace, cudaStream_t) {
    nvfp4_unsupported();
}
void nvfp4_attn_input_dispatch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                               LinearPolicy, WorkspaceArena*, cudaStream_t) {
    nvfp4_unsupported();
}

// --- nvfp4_gdn_input_plan.h --------------------------------------------------
std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    nvfp4_unsupported();
}
void nvfp4_gdn_input_decode_launch(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t) {
    nvfp4_unsupported();
}
void nvfp4_gdn_input_small_t_launch(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t) {
    nvfp4_unsupported();
}
void nvfp4_gdn_input_w4a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                 Nvfp4W4a4Workspace, cudaStream_t) {
    nvfp4_unsupported();
}
void nvfp4_gdn_input_dispatch(const Tensor&, const Weight&, Tensor&, Tensor&, LinearPolicy,
                              WorkspaceArena*, cudaStream_t) {
    nvfp4_unsupported();
}

// --- nvfp4_gdn_snapshot_plan.h -----------------------------------------------
Nvfp4GdnConvPlan nvfp4_gdn_conv_resolve_plan(LinearPolicy, std::int32_t, std::int32_t) {
    nvfp4_unsupported();
}
std::size_t nvfp4_gdn_snapshot_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    nvfp4_unsupported();
}
void nvfp4_gdn_snapshot_decode_launch(const Tensor&, const Weight&, const Tensor&, Tensor&,
                                      const Tensor&, const Tensor&, const Tensor&, Tensor&,
                                      Tensor&, Tensor&, Tensor&, cudaStream_t) {
    nvfp4_unsupported();
}
void nvfp4_gdn_snapshot_small_t_launch(const Tensor&, const Weight&, const Tensor&, Tensor&,
                                       const Tensor&, const Tensor&, const Tensor&, Tensor&,
                                       Tensor&, Tensor&, Tensor&, cudaStream_t) {
    nvfp4_unsupported();
}
void nvfp4_gdn_record_small_t_launch(const Tensor&, const Weight&, const Tensor&, const Tensor&,
                                     const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&,
                                     Tensor&, Tensor&, cudaStream_t) {
    nvfp4_unsupported();
}
void nvfp4_gdn_snapshot_post_launch(const Tensor&, const Tensor&, Tensor&, const Tensor&,
                                    const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&,
                                    cudaStream_t) {
    nvfp4_unsupported();
}
void nvfp4_gdn_record_post_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                  const Tensor&, Tensor&, Tensor&, Tensor&, cudaStream_t) {
    nvfp4_unsupported();
}
void nvfp4_gdn_snapshot_dispatch(const Tensor&, const Weight&, const Tensor&, Tensor&,
                                 const Tensor&, const Tensor&, const Tensor&, Tensor&, Tensor&,
                                 Tensor&, Tensor&, LinearPolicy, WorkspaceArena&, cudaStream_t) {
    nvfp4_unsupported();
}

// --- nvfp4_linear_add_plan.h -------------------------------------------------
std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t, std::int32_t, LinearPolicy,
                                                      std::int32_t, std::int32_t) {
    nvfp4_unsupported();
}
void nvfp4_linear_add_decode_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    nvfp4_unsupported();
}
void nvfp4_linear_add_small_t_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    nvfp4_unsupported();
}
void nvfp4_linear_add_w4a4_launch(const Tensor&, const Weight&, Tensor&, Nvfp4W4a4Workspace,
                                  cudaStream_t) {
    nvfp4_unsupported();
}
void nvfp4_linear_add_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy, WorkspaceArena&,
                               cudaStream_t) {
    nvfp4_unsupported();
}

// --- nvfp4_linear_swiglu_plan.h ----------------------------------------------
std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy, std::int32_t,
                                                         std::int32_t) {
    nvfp4_unsupported();
}
void nvfp4_linear_swiglu_decode_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    nvfp4_unsupported();
}
void nvfp4_linear_swiglu_small_t_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    nvfp4_unsupported();
}
void nvfp4_linear_swiglu_w4a4_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                                     cudaStream_t) {
    nvfp4_unsupported();
}
void nvfp4_linear_swiglu_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy,
                                  WorkspaceArena&, cudaStream_t) {
    nvfp4_unsupported();
}

// --- nvfp4_geometry.h (geometry structs used by wrappers) --------------------
// Geometry constants are constexpr in the header; nothing to stub there.

// --- prompt_nvfp4_non_rdc_launch.h -------------------------------------------
// The warp-specialized (setmaxnreg/TMA) NVFP4-KV prompt attention kernel lives
// in the sm_120a-only non-RDC archive. On sm_89 the NVFP4 KV cache path is not
// offered (use --kv-dtype fp8 / int8 / bf16); these stubs keep prompt_nvfp4.cu
// linkable while failing loudly if it is ever reached.
void causal_attention_prompt_nvfp4_kernel_launch(const Tensor&, const Tensor&, float,
                                                 const PagedKVLayerView&, Tensor&, cudaStream_t) {
    nvfp4_unsupported();
}
void causal_attention_prompt_nvfp4_batch_kernel_launch(const Tensor&, const Tensor&, const Tensor&,
                                                       const Tensor&, float,
                                                       const PagedKVBatchLayerView&, Tensor&,
                                                       cudaStream_t) {
    nvfp4_unsupported();
}

} // namespace ninfer::ops::detail
