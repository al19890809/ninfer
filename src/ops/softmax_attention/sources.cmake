target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dense/plain/softmax_attention.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/causal_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/bf16/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/bf16/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/fp8/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/fp8/tiled_launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/fp8/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/int8/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/int8/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/packed_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/context_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/sliding_window_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/plan.cpp"
)

# NINFER_L20 / sm_89 (Ada) port: the NVFP4/K8V4 KV-cache attention kernels
# decode NVFP4 KV through the hardware e2m1 conversion (sm_100a/sm_120a-only),
# so the whole representation is excluded on sm_89 (entry points covered by
# throwing stubs in nvfp4_l20_stub.cpp). bf16/int8/fp8 KV attention kernels
# are unaffected.
if(NOT NINFER_NO_NVFP4_WEIGHTS)
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/plan.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/k8v4/launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/k8v4/tiled_launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/k8v4/plan.cpp"
  )
  target_sources(ninfer_nvfp4_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/tiled_launch.cu"
  )
endif()

include("${CMAKE_CURRENT_LIST_DIR}/qsa/sources.cmake")
