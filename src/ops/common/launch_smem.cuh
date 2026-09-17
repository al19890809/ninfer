#pragma once

// Shared-memory-aware kernel launch.
//
// sm_89 (Ada) caps statically allocated shared memory at 48 KB per CTA, while
// sm_120a accepts static storage up to the full per-CTA shared memory. Kernels
// whose shared storage exceeds 48 KB therefore declare it as dynamic
// (extern __shared__) and every launch site sizes the dynamic allocation from
// the storage type. This helper performs that launch: it passes the storage
// size as dynamic shared bytes (0 when the storage still fits statically) and
// raises the per-function dynamic shared limit only when required.

#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer::ops::detail {

inline constexpr std::size_t kStaticSharedLimitBytes = 48 * 1024;

// Dynamic shared bytes to launch with for a kernel shared storage of this size.
inline constexpr std::size_t dynamic_shared_bytes_for(std::size_t storage_bytes) {
    return storage_bytes > kStaticSharedLimitBytes ? storage_bytes : 0;
}

// Returns the launch error; call sites also pass through CUDA_CHECK
// (cudaGetLastError) as with the other kernel launches in this library.
// Kernel arguments are taken by value -- the launch ABI copies them anyway,
// and by-value copies keep the argument-pointer array free of const issues.
template <class Kernel, class... Args>
inline cudaError_t launch_kernel_with_smem(Kernel kernel, dim3 grid, dim3 block,
                                           std::size_t storage_bytes, cudaStream_t stream,
                                           Args... args) {
    const std::size_t dynamic_bytes = dynamic_shared_bytes_for(storage_bytes);
    if (dynamic_bytes > 0) {
        const cudaError_t attribute = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<std::int32_t>(dynamic_bytes));
        if (attribute != cudaSuccess) return attribute;
    }
    void* arg_ptrs[] = {static_cast<void*>(&args)...};
    return cudaLaunchKernel(reinterpret_cast<const void*>(kernel), grid, block, arg_ptrs,
                            dynamic_bytes, stream);
}

} // namespace ninfer::ops::detail
