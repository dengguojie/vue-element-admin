cmake_minimum_required(VERSION 3.14.1)
project(constant_folding_ops)
get_filename_component(TOP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if(BUILD_OPEN_PROJECT)
set(local_folding_inc_path
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${METADEF_INCLUDE}
    ${METADEF_INCLUDE}/external
    ${METADEF_INCLUDE}/external/graph
    ${GRAPHENGINE_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}/../context/inc
    ${AICPU_OPP_ENV}/inc
    ${FWKACLLIB_INCLUDE}
)
else()
set(local_folding_inc_path
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${METADEF_INCLUDE}
    ${METADEF_INCLUDE}/external
    ${METADEF_INCLUDE}/external/graph
    ${GRAPHENGINE_INCLUDE}
    ${TOP_DIR}/inc
    ${TOP_DIR}/inc/aicpu/cpu_kernels
    ${TOP_DIR}/inc/external/aicpu
)
endif()

set (local_folding_src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/folding.cc
)

add_library(constant_folding_ops SHARED
    ${local_folding_src_files}
)

target_compile_options(constant_folding_ops PRIVATE
    -O2
    -ftrapv
    -std=c++11
    -fstack-protector-all
    -fvisibility-inlines-hidden
    -fvisibility=hidden
)

target_compile_definitions(constant_folding_ops PRIVATE
    _FORTIFY_SOURCE=2
    google=ascend_private
    LOG_CPP
)

target_link_options(constant_folding_ops PRIVATE
    -Wl,-z,relro,-z,now
    -Wl,-Bsymbolic -Wl,--exclude-libs,ALL
    -s
)

if(BUILD_OPEN_PROJECT)
  target_link_libraries(constant_folding_ops PRIVATE
    $<BUILD_INTERFACE:intf_pub_aicpu>
    -Wl,--whole-archive
    cpu_kernels_static
    cpu_kernels_context_static
    -Wl,--no-whole-archive
    ascend_protobuf_static
    -Wl,--no-as-needed
    alog
    graph
    register
    -ldl
    -Wl,--as-needed
  )
else()
  target_link_libraries(constant_folding_ops PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--whole-archive
    cpu_kernels_static
    cpu_kernels_context_static
    -Wl,--no-whole-archive
    ascend_protobuf_static
    -Wl,--no-as-needed
    alog
    graph
    register
    -ldl
    -Wl,--as-needed
  )
endif()

target_include_directories(constant_folding_ops PRIVATE
     ${local_folding_inc_path}
)

if(BUILD_OPEN_PROJECT)
  set(OPS_PATH "${INSTALL_PATH}/opp/op_impl/built-in/host_cpu")
  cann_install(
    TARGET      constant_folding_ops
    FILES       $<TARGET_FILE:constant_folding_ops>
    DESTINATION "${OPS_PATH}"
  )
else()
  install(
    TARGETS constant_folding_ops OPTIONAL
    EXPORT  constant_folding_ops-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
  )
endif()
