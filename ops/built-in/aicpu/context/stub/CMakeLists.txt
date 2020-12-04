if("x${PRODUCT_SIDE}" STREQUAL "xdevice")
  set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-g++)
  set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-gcc)

  add_library(aicpu_sharder SHARED
    ./aicpu_sharder.cc
  )

  target_compile_options(aicpu_sharder PRIVATE
    -std=c++11
  )
endif()