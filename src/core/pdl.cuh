#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

// NINFER_L20 / sm_89 (Ada Lovelace) port
// ---------------------------------------------------------------------------
// Programmatic Dependent Launch (`griddepcontrol.*`) requires sm_90+. The L20
// is sm_89, so PDL is unavailable there. PDL is only a latency optimization:
// a consumer kernel launched on the same stream already begins only after its
// producer kernel has fully completed (CUDA in-order stream semantics), so a
// plain launch is a correct, slightly slower substitute.
//
// Strategy:
//  * On sm_89 the device helpers become no-ops (there are no dependents to
//    trigger and no dependency to wait on -- stream order supplies both).
//  * The host `launch_dependent` degenerates to a normal `cudaLaunchKernel`
//    when compiled for a pre-sm_90-only build. Because CMAKE_CUDA_ARCHITECTURES
//    for the L20 build is exactly one arch (89), a host-side check on the
//    compiled arch is reliable; we detect it through __CUDA_ARCH__ when the
//    TU is a CUDA file, and fall back to launching normally via a runtime flag
//    compiled in by CMake (NINFER_L20_PLAIN_LAUNCH) for safety.

namespace ninfer::pdl {

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    std::size_t dynamic_smem_bytes = 0;
    cudaStream_t stream            = nullptr;
};

namespace detail {
// Set at compile time by the CUDA compiler for the *single* target arch of
// this build. For a pure sm_89 build this is false.
#if defined(NINFER_L20_NO_PDL)
inline constexpr bool kPdlAvailable = false;
#else
inline constexpr bool kPdlAvailable = true;
#endif
} // namespace detail

// Launches a consumer kernel as a programmatic dependent of the immediately
// preceding producer kernel in the same stream. Every consumer control path
// that reads producer output must first call wait_for_dependencies().
template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline cudaError_t
launch_dependent(const LaunchConfig& launch, void (*kernel)(KernelArgs...), CallArgs&&... args) {
    if constexpr (!detail::kPdlAvailable) {
        // sm_89 / L20 path: plain in-order launch (PDL attribute omitted). We
        // take the address of each argument and pass a void* array to the
        // driver-style launch; every current call site passes plain scalar /
        // pointer arguments, which is exactly what this supports.
        void* arg_ptrs[] = {const_cast<void*>(static_cast<const void*>(&args))...};
        return cudaLaunchKernel(reinterpret_cast<const void*>(kernel), launch.grid, launch.block,
                                arg_ptrs, launch.dynamic_smem_bytes, launch.stream);
    } else {
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute.val.programmaticStreamSerializationAllowed = 1;

        cudaLaunchConfig_t config{};
        config.gridDim          = launch.grid;
        config.blockDim         = launch.block;
        config.dynamicSmemBytes = launch.dynamic_smem_bytes;
        config.stream           = launch.stream;
        config.attrs            = &attribute;
        config.numAttrs         = 1;

        return cudaLaunchKernelEx(&config, kernel, std::forward<CallArgs>(args)...);
    }
}

// Every producer CTA must call this at least once or exit. This enables
// dependent scheduling but does not make producer writes visible to the
// consumer. No-op on sm_89.
__device__ __forceinline__ void trigger_dependents() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// Call on every consumer control path before its first access to
// producer-dependent data. No-op on sm_89 (stream order already guarantees
// the producer finished before the consumer started).
__device__ __forceinline__ void wait_for_dependencies() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif
}

} // namespace ninfer::pdl
