cmake_minimum_required(VERSION 3.14.1)
project(cpu_kernels_context)
get_filename_component(TOP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(proto_src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/proto/cpu_attr.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/proto/cpu_node_def.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/proto/cpu_tensor.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/proto/cpu_tensor_shape.proto
)
protobuf_generate(aicpu PROTO_SRCS PROTO_HDRS ${proto_src_files})

set (local_context_src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/node_def.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/node_def_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/tensor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/tensor_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/tensor_shape.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/tensor_shape_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/attr_value.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/attr_value_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/device.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/context.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/device_cpu_kernel.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/cpu_kernel_register.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/cpu_kernel_utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/host_sharder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/device_sharder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/eigen_threadpool.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/cpu_kernel_cache.cc
    ${PROTO_SRCS}
)

set(local_context_stub_files
   ${CMAKE_CURRENT_SOURCE_DIR}/stub/aicpu_sharder.cc
)

if(BUILD_OPEN_PROJECT)
  set(_proto_include "${PROTO_BINARY_DIR}/aicpu")

  set(local_context_inc_path
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/stub
    ${AICPU_OPP_ENV}/inc
    ${_proto_include}
    ${Protobuf_INCLUDE}
    ${FWKACLLIB_INCLUDE}
    ${FWKACLLIB_INCLUDE}/aicpu/common
    ${EIGEN_INCLUDE}
  )

  add_library(cpu_kernels_context_static STATIC
    ${local_context_src_files}
    $<$<NOT:$<STREQUAL:${PRODUCT_SIDE},device>>:${local_context_stub_files}>
  )

  target_include_directories(cpu_kernels_context_static PRIVATE
    ${local_context_inc_path}
  )

  target_compile_definitions(cpu_kernels_context_static PRIVATE
    _FORTIFY_SOURCE=2
    google=ascend_private
    $<$<NOT:$<STREQUAL:${PRODUCT_SIDE},device>>:LOG_CPP>
  )

  target_compile_options(cpu_kernels_context_static PRIVATE
    -O2
    -std=c++11
    -ftrapv
    -fstack-protector-all
    -fPIC
    $<$<STREQUAL:${PRODUCT_SIDE},device>:-fvisibility-inlines-hidden>
    $<$<STREQUAL:${PRODUCT_SIDE},device>:-fvisibility=hidden>
  )

  target_link_libraries(cpu_kernels_context_static PRIVATE
    $<BUILD_INTERFACE:intf_pub_aicpu>
  )

  set(cpu_kernels_context_static ${CMAKE_CURRENT_BINARY_DIR}/libcpu_kernels_context_static.a)

  set_target_properties(cpu_kernels_context_static
      PROPERTIES
      OUTPUT_NAME cpu_kernels_context
  )
  if("x${PRODUCT_SIDE}" STREQUAL "xdevice")
    if (MINRC)
      set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
      set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
    else()
      set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-g++)
      set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-gcc)
    endif()
    add_subdirectory(stub)

    set(OPS_AICPU_PATH "${INSTALL_PATH}/aicpu")
    cann_install(
      TARGET      cpu_kernels_context_static
      FILES       $<TARGET_FILE:cpu_kernels_context_static>
      DESTINATION "${OPS_AICPU_PATH}"
    )
  endif()

else()
set(local_context_inc_path
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto
    ${TOP_DIR}/inc
    ${TOP_DIR}/inc/aicpu
    ${TOP_DIR}/inc/aicpu/common
    ${TOP_DIR}/inc/aicpu/cpu_kernels
    ${TOP_DIR}/inc/aicpu/aicpu_schedule/aicpu_sharder
    ${TOP_DIR}/inc/external/aicpu
    ${TOP_DIR}/libc_sec/include
    ${TOP_DIR}/third_party/protobuf/include
    ${TOP_DIR}/open_source/eigen
    ${TOP_DIR}/out/${product}
    ${CMAKE_BINARY_DIR}/proto/aicpu/proto/cpu_proto/proto
    ${CMAKE_BINARY_DIR}/proto/aicpu/proto
    ${CMAKE_BINARY_DIR}/proto/aicpu
)

add_library(cpu_kernels_context SHARED
    ${local_context_src_files}
    ${local_context_stub_files}
)

add_library(cpu_kernels_context_static STATIC
    ${local_context_src_files}
    ${local_context_stub_files}
)

target_link_libraries(cpu_kernels_context PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    slog
    PUBLIC c_sec
    ascend_protobuf
    -ldl
)


##cpu_kernels_context
target_include_directories(cpu_kernels_context PRIVATE
     ${local_context_inc_path}
)

target_compile_definitions(cpu_kernels_context PRIVATE
    _FORTIFY_SOURCE=2
    $<$<STREQUAL:${PRODUCT_SIDE},host>:VISIBILITY>
    google=ascend_private
)

target_compile_options(cpu_kernels_context PRIVATE
    -O2
    -std=c++11
    -ftrapv
    $<$<STREQUAL:${PRODUCT_SIDE},host>:-fvisibility-inlines-hidden>
    $<$<STREQUAL:${PRODUCT_SIDE},host>:-fvisibility=hidden>
)

target_link_options(cpu_kernels_context PRIVATE
    -Wl,-z,relro,-z,now
    -s
    $<$<STREQUAL:${PRODUCT_SIDE},host>:-Wl,-Bsymbolic -Wl,--exclude-libs,ALL>
)

target_include_directories(cpu_kernels_context_static PRIVATE
     ${local_context_inc_path}
)

target_compile_definitions(cpu_kernels_context_static PRIVATE
    _FORTIFY_SOURCE=2
    google=ascend_private
    $<$<NOT:$<STREQUAL:${PRODUCT_SIDE},device>>:LOG_CPP>
)

target_compile_options(cpu_kernels_context_static PRIVATE
    -O2
    -std=c++11
    -ftrapv
    -fstack-protector-all
    $<$<STREQUAL:${PRODUCT_SIDE},device>:-fvisibility-inlines-hidden>
    $<$<STREQUAL:${PRODUCT_SIDE},device>:-fvisibility=hidden>
)

target_link_libraries(cpu_kernels_context_static PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -Wl,-z,relro,-z,now
    -s
    -ldl
    -shared
)

set(INSTALL_LIBRARY_DIR lib)

install(TARGETS cpu_kernels_context OPTIONAL
    EXPORT cpu_kernels_context-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(TARGETS cpu_kernels_context_static OPTIONAL
    EXPORT cpu_kernels_context_static-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)

set(cpu_kernels_context_static ${CMAKE_CURRENT_BINARY_DIR}/libcpu_kernels_context_static.a)

set_target_properties(cpu_kernels_context_static
    PROPERTIES
    OUTPUT_NAME cpu_kernels_context
)
endif()
