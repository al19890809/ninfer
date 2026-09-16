target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/append/kv_cache_append.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/append/launch.cu"
)

# NINFER_L20 / sm_89 (Ada) port: the NVFP4/K8V4 KV-cache encode kernels use the
# hardware `cvt .e2m1x2` FP4 conversion, which is sm_100a/sm_120a-only. Exclude
# them on sm_89 and provide throwing stubs (see nvfp4_l20_stub.cpp); the
# bf16/int8/fp8 KV representations are unaffected.
if(NOT NINFER_NO_NVFP4_WEIGHTS)
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/append/nvfp4_launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/append/k8v4_launch.cu"
  )
endif()
