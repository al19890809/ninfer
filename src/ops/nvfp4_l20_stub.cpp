// NINFER_L20 / sm_89 (Ada) port — NVFP4 weight-op stubs.
//
// sm_89 has no hardware NVFP4 weight path, so every NVFP4 kernel translation
// unit is excluded from this build (see NINFER_NO_NVFP4_WEIGHTS in
// src/ops/CMakeLists.txt). The dispatchers that would route to those kernels
// are still compiled; this file provides throwing definitions for the symbols
// they reference. Any NVFP4-weight artifact or NVFP4 KV-cache request on an
// L20 build therefore fails fast with a clear error instead of a link error.

#include <stdexcept>

namespace ninfer::ops::detail {

[[noreturn]] inline void nvfp4_unsupported() {
    throw std::runtime_error(
        "NVFP4 operation is not supported in this build (NINFER_L20 / sm_89 port: "
        "NVFP4 kernels excluded, stub called)");
}

} // namespace ninfer::ops::detail
