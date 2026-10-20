# NINFER_L20 / sm_89 (Ada) port: NVFP4 weight kernels require sm_100a/sm_120a
# hardware conversion; excluded on sm_89 (stubs in nvfp4_l20_stub.cpp).
if(NOT NINFER_NO_NVFP4_WEIGHTS)
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/nvfp4_dispatch.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/shapes/n14336_k5120.cu"
    "${CMAKE_CURRENT_LIST_DIR}/shapes/n16384_k5120.cu"
    "${CMAKE_CURRENT_LIST_DIR}/shapes/n34816_k5120.cu"
    "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k6144.cu"
    "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k17408.cu"
  )

  target_sources(ninfer_nvfp4_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a4_tma.cu"
  )
endif()
