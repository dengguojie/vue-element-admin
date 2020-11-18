set(SUB_SO_WITH_VERSION libcpu_kernels_v1.0.1.so)

if(BUILD_OPEN_PROJECT)
#[[set(local_kernels_src_files
    kernels/normalized/drop_out_gen_mask_kernels.cc
    kernels/normalized/expanddims.cc
    kernels/normalized/reshape_kernels.cc
    kernels/normalized/less_kernels.cc
    kernels/normalized/top_k_kernels.cc
    kernels/normalized/edit_distance_kernels.cc
    kernels/normalized/gather_kernels.cc
    kernels/normalized/get_dynamic_dims.cc
    kernels/normalized/identity_kernels.cc
    kernels/normalized/unique_with_pad_kernels.cc
    kernels/normalized/ceil_kernels.cc
    kernels/normalized/realdiv_kernels.cc
    kernels/normalized/sparse_to_dense_kernels.cc
    kernels/normalized/deformable_offsets_kernels.cc
    kernels/normalized/cast_kernels.cc
    kernels/normalized/logging_kernels.cc
    kernels/normalized/concatv2_kernels.cc
    utils/sparse_group.cc
    utils/eigen_tensor.cc
    utils/sparse_tensor.cc
    utils/bcast.cc
)

set(local_host_kernels_src_files
    kernels/host/add_kernel.cc
    kernels/host/mul_kernel.cc
    kernels/host/random_uniform_kernel.cc
)

set(AICPU_INCLUDE /usr/local/Ascend/opp/op_impl/built-in/aicpu/aicpu_kernel/inc)

set(local_kernels_inc_path
    ./
    utils
    ${AICPU_INCLUDE}
    ${EIGEN_INCLUDE}
    ${CANN_ROOT}/third_party/fwkacllib/inc
    ${CANN_ROOT}/third_party/fwkacllib/inc/aicpu/cpu_kernels
)

add_library(cpu_kernels SHARED
    ${local_kernels_src_files}
)

add_library(cpu_kernels_host SHARED
    ${local_kernels_src_files}
    ${local_host_kernels_src_files}
)

add_library(cpu_kernels_static STATIC
    ${local_kernels_src_files}
    ${local_host_kernels_src_files}
)

add_dependencies(cpu_kernels eigen_headers)
add_dependencies(cpu_kernels_host eigen_headers)
add_dependencies(cpu_kernels_static eigen_headers)

###cpu_kernels
target_include_directories(cpu_kernels PRIVATE
    ${local_kernels_inc_path}
)

target_include_directories(cpu_kernels_host PRIVATE
    ${local_kernels_inc_path}
)

target_include_directories(cpu_kernels_static PRIVATE
    ${local_kernels_inc_path}
)

set(AICPU_CONTEXT /usr/local/Ascend/opp/op_impl/built-in/aicpu/aicpu_kernel/lib/device/)

find_library(cpu_kernels_context libcpu_kernels_context.so ${AICPU_CONTEXT})

target_link_libraries(cpu_kernels_host PRIVATE
    ${slog}
    c_sec
    ${cpu_kernels_context}
)

target_link_libraries(cpu_kernels PRIVATE
    ${slog}
    c_sec
    ${cpu_kernels_context}
)

target_link_libraries(cpu_kernels_static PRIVATE
    c_sec
)

set(OPS_AICPU_PATH "${INSTALL_PATH}/opp/op_impl/built-in/aicpu/aicpu_kernel/lib")

cann_install(
  TARGET      cpu_kernels
  FILES       $<TARGET_FILE:cpu_kernels>
  DESTINATION "${OPS_AICPU_PATH}/device"
)

cann_install(
  TARGET      cpu_kernels_host
  FILES       $<TARGET_FILE:cpu_kernels_host>
  DESTINATION "${OPS_AICPU_PATH}/host"
)

cann_install(
  TARGET      cpu_kernels_static
  FILES       $<TARGET_FILE:cpu_kernels_static>
  DESTINATION "${OPS_AICPU_PATH}/host"
)]]
else()
get_filename_component(TOP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(local_kernels_src_files
    kernels/normalized/drop_out_gen_mask_kernels.cc
    kernels/normalized/expanddims.cc
    kernels/normalized/reshape_kernels.cc
    kernels/normalized/less_kernels.cc
    kernels/normalized/top_k_kernels.cc
    kernels/normalized/edit_distance_kernels.cc
    kernels/normalized/gather_kernels.cc
    kernels/normalized/identity_kernels.cc
    kernels/normalized/unique_with_pad_kernels.cc
    kernels/normalized/ceil_kernels.cc
    kernels/normalized/realdiv_kernels.cc
    kernels/normalized/sparse_to_dense_kernels.cc
    kernels/normalized/deformable_offsets_kernels.cc
    kernels/normalized/cast_kernels.cc
    kernels/normalized/logging_kernels.cc
    kernels/normalized/concatv2_kernels.cc
    kernels/normalized/round.cc
    utils/sparse_group.cc
    utils/eigen_tensor.cc
    utils/sparse_tensor.cc
    utils/bcast.cc
)

set(local_host_kernels_src_files
    kernels/host/add_kernel.cc
    kernels/host/mul_kernel.cc
    kernels/host/random_uniform_kernel.cc
)

set(local_kernels_inc_path
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${TOP_DIR}/inc/aicpu/cpu_kernels
    ${TOP_DIR}/inc/external/aicpu
    ${TOP_DIR}/third_party/eigen/src/eigen-3.3.7
    ${TOP_DIR}/inc
    ${TOP_DIR}/libc_sec/include
)

add_library(cpu_kernels SHARED
    ${local_kernels_src_files}
)

add_library(cpu_kernels_host SHARED
    ${local_kernels_src_files}
	${local_host_kernels_src_files}
)

add_library(cpu_kernels_static STATIC
    ${local_kernels_src_files}
	${local_host_kernels_src_files}
)

###cpu_kernels
target_include_directories(cpu_kernels PRIVATE
    ${local_kernels_inc_path}
)

target_include_directories(cpu_kernels_host PRIVATE
    ${local_kernels_inc_path}
)

target_include_directories(cpu_kernels_static PRIVATE
    ${local_kernels_inc_path}
)

target_link_libraries(cpu_kernels_host PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -ldl
    slog
    c_sec
    cpu_kernels_context
)

target_link_libraries(cpu_kernels PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--whole-archive
    cpu_kernels_context_static
    -Wl,--no-whole-archive
    ascend_protobuf_static
    -Wl,--no-as-needed
    slog
    c_sec
    $<$<NOT:$<STREQUAL:${PRODUCT}, npuf10>>:aicpu_sharder>
    -ldl
    -Wl,--as-needed
)

target_compile_definitions(cpu_kernels PRIVATE
    _FORTIFY_SOURCE=2
    google=ascend_private
)

target_compile_options(cpu_kernels PRIVATE
    -O2
    -std=c++14
    -ftrapv
    -fstack-protector-all
    -fvisibility-inlines-hidden
    -fvisibility=hidden
)

target_link_options(cpu_kernels PRIVATE
    -Wl,--exclude-libs=libascend_protobuf.a
    -Wl,-z,relro,-z,now
    -s
)

target_compile_definitions(cpu_kernels_static PRIVATE
    _FORTIFY_SOURCE=2
)

target_link_libraries(cpu_kernels_static PRIVATE
    $<BUILD_INTERFACE:intf_pub>
)


target_compile_options(cpu_kernels_static PRIVATE
    -O2
    -std=c++11
    -ftrapv
	-fstack-protector-all
)

target_compile_definitions(cpu_kernels_host PRIVATE
    _FORTIFY_SOURCE=2
)

target_link_options(cpu_kernels_host PRIVATE
    -Wl,-z,relro,-z,now
    -Wl,-Bsymbolic -Wl,--exclude-libs,ALL
    -s
)

target_compile_options(cpu_kernels_host PRIVATE
    -O2
    -std=c++11
    -ftrapv
    -fvisibility-inlines-hidden
    -fvisibility=hidden
	-fstack-protector-all
)

set(INSTALL_LIBRARY_DIR lib)

install(TARGETS cpu_kernels OPTIONAL
    EXPORT cpu_kernels-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(TARGETS cpu_kernels_host OPTIONAL
    EXPORT cpu_kernels_host-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(TARGETS cpu_kernels_static OPTIONAL
    EXPORT cpu_kernels_static-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)

set(COPY_TO_PATH ${CMAKE_INSTALL_PREFIX}/${INSTALL_LIBRARY_DIR}/${SUB_SO_WITH_VERSION})
add_custom_command(TARGET cpu_kernels POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cpu_kernels> ${COPY_TO_PATH}
    COMMENT "[AICPU] copying cpu_kernels library to '${COPY_TO_PATH}'"
)

set(cpu_kernels_static ${CMAKE_CURRENT_BINARY_DIR}/libcpu_kernels_static.a)
set(cpu_kernels_host ${CMAKE_CURRENT_BINARY_DIR}/libcpu_kernels_host.so)

set_target_properties(cpu_kernels_static
    PROPERTIES
    OUTPUT_NAME cpu_kernels
)
set_target_properties(cpu_kernels_host
    PROPERTIES
    OUTPUT_NAME cpu_kernels
)
endif()


