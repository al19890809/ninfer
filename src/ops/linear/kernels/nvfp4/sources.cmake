if(NOT NINFER_NO_NVFP4_WEIGHTS)
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a4.cu"
  )
endif()
