target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/causal_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_fp8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_fp8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_nvfp4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/packed_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/context_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/sliding_window_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/launch.cu"
)

# NINFER_L20 / sm_89 (Ada) port: the NVFP4/K8V4 KV-cache attention kernels
# decode NVFP4 KV through the hardware e2m1 conversion (sm_100a/sm_120a-only),
# and their encode side (kv_cache/sources.cmake) can never be populated on
# sm_89, so the whole representation is excluded and covered by throwing stubs
# in nvfp4_l20_stub.cpp. prompt_nvfp4.cu is pure host composition over the
# non-RDC kernel launchers, which are stubbed on sm_89; it stays. The
# bf16/int8/fp8 KV attention kernels are unaffected.
if(NOT NINFER_NO_NVFP4_WEIGHTS)
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_nvfp4.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_k8v4.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_k8v4.cu"
  )
  target_sources(ninfer_nvfp4_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_nvfp4_non_rdc.cu"
  )
endif()
