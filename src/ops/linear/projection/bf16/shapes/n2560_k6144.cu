#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/projection/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using S0 = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<2, 1, 1, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    6144>;
using S1 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 8, 64, 2>, 6144>;
using S2 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<32, 8, 8, 64, 2>, 6144>;
using S3 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<32, 16, 4, 64, 2>, 6144>;
using S4 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 256, 16, 8, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    6144>;
#ifndef NINFER_L20_NO_PDL
using S5 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 128, 32, 16, 2, 2>, 6144>;
#endif
#ifndef NINFER_L20_NO_PDL
using S6 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 128, 16, 16, 2, 2>, 6144>;
#endif
#ifndef NINFER_L20_NO_PDL
using S7 = Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 1, Bf16MmaRaster::Grouped, 4>, 6144>;
#endif
#ifndef NINFER_L20_NO_PDL
using S8 = Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>;
#endif
} // namespace

Bf16Launch select_bf16_n2560_k6144(std::int32_t tokens) {
    if (tokens <= 1) return launch_bf16_gemv<S0>;
    if (tokens <= 8) return launch_bf16_sliced_k_mma<S1>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<S2>;
    if (tokens <= 32) return launch_bf16_sliced_k_mma<S3>;
    if (tokens <= 64) return launch_bf16_mma<S4>;
    if (tokens <= 96) return launch_bf16_mma<S4>;
    if (tokens <= 128) return launch_bf16_mma<S4>;
    if (tokens <= 512) return launch_bf16_mma<S4>;
    return launch_bf16_mma<S4>;
}
} // namespace ninfer::ops::detail
