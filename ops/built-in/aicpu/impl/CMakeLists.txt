set(SUB_SO_WITH_VERSION libcpu_kernels_v1.0.1.so)

set(local_kernels_src_files
    kernels/normalized/acos.cc
    kernels/normalized/acosh.cc
    kernels/normalized/adaptive_max_pool_2d.cc
    kernels/normalized/add.cc
    kernels/normalized/asin.cc
    kernels/normalized/asinh.cc
    kernels/normalized/atan.cc
    kernels/normalized/atanh.cc
    kernels/normalized/betainc.cc
    kernels/normalized/cache_swap_table.cc
    kernels/normalized/calc_buckets_limit_and_offset.cc
    kernels/normalized/candidate_sampler.cc
    kernels/normalized/case_condition.cc
    kernels/normalized/cast.cc
    kernels/normalized/ceil.cc
    kernels/normalized/compute_accidental_hits.cc
    kernels/normalized/concatv2.cc
    kernels/normalized/coordinates_1d_to_2d.cc
    kernels/normalized/cosh.cc
    kernels/normalized/diag.cc
    kernels/normalized/dynamic_stitch.cc
    kernels/normalized/edit_distance.cc
    kernels/normalized/elugrad.cc
    kernels/normalized/embedding_lookup.cc
    kernels/normalized/equal.cc
    kernels/normalized/expanddims.cc
    kernels/normalized/fill.cc
    kernels/normalized/get_dynamic_dims.cc
    kernels/normalized/greater.cc
    kernels/normalized/greater_equal.cc
    kernels/normalized/identity.cc
    kernels/normalized/image_warp_offsets.cc
    kernels/normalized/index_to_addr.cc
    kernels/normalized/inplace_top_k_distance.cc
    kernels/normalized/less.cc
    kernels/normalized/log_matrix_determinant.cc
    kernels/normalized/logging.cc
    kernels/normalized/log1p.cc
    kernels/normalized/masked_select.cc
    kernels/normalized/masked_select_grad.cc
    kernels/normalized/mul.cc
    kernels/normalized/neg.cc
    kernels/normalized/non_max_suppression_v3_kernels.cc
    kernels/normalized/not_equal.cc
    kernels/normalized/ones_like.cc
    kernels/normalized/random_uniform.cc
    kernels/normalized/realdiv.cc
    kernels/normalized/reshape.cc
    kernels/normalized/resize_bilinear.cc
    kernels/normalized/resize_bilinear_grad.cc
    kernels/normalized/round.cc
    kernels/normalized/scatter_elements.cc
    kernels/normalized/search_sorted.cc
    kernels/normalized/slice_write.cc
    kernels/normalized/sparse_to_dense.cc
    kernels/normalized/spatial_transformer.cc
    kernels/normalized/split_v.cc
    kernels/normalized/strided_slice.cc
    kernels/normalized/strided_slice_v2.cc
    kernels/normalized/sub.cc
    kernels/normalized/top_k.cc
    kernels/normalized/top_k_v2_d.cc
    utils/allocator_utils.cc
    utils/bcast.cc
    utils/broadcast_iterator.cc
    utils/eigen_tensor.cc
    utils/kernel_util.cc
    utils/range_sampler.cc
    utils/sparse_group.cc
    utils/sparse_tensor.cc
    kernels/normalized/top_k_pq_distance.cc
)

set(local_host_kernels_src_files
    kernels/host/assign.cc
    kernels/host/format_transfer/format_transfer_fractal_nz.cc
    kernels/host/format_transfer/format_transfer_fractal_z.cc
    kernels/host/format_transfer/format_transfer_fractalz_3d.cc
    kernels/host/format_transfer/format_transfer_hwcn_fractal_zn_lstm.cc
    kernels/host/format_transfer/format_transfer_transpose.cc
    kernels/host/format_transfer/format_transfer_utils.cc
    kernels/host/format_transfer/register_format_transfer.cc
    kernels/host/gemm.cc
    kernels/host/matmul.cc
    kernels/host/matrix_triangular_solve.cc
    kernels/host/no_op.cc
    kernels/host/trans_data_rnn.cc
    kernels/host/transdata.cc
)
set(local_device_kernels_src_files
    kernels/device/avg_pool1d_avg_matrix.cc
    kernels/device/mem_cpy.cc
    kernels/device/where.cc
)

if(BUILD_OPEN_PROJECT)
  set(AICPU_INCLUDE ${AICPU_OPP_ENV}/inc)

  set(local_kernels_inc_path
    ./
    utils
    ../context/inc
    ${AICPU_INCLUDE}
    ${EIGEN_INCLUDE}
    ${CANN_ROOT}/third_party/fwkacllib/inc
    ${CANN_ROOT}/third_party/fwkacllib/inc/aicpu/cpu_kernels
    ${C_SEC_INCLUDE}
  )

  if("x${PRODUCT_SIDE}" STREQUAL "xdevice")
    if (MINRC)
      set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
      set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
    else()
      set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-g++)
      set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-gcc)
    endif()
    add_library(cpu_kernels SHARED
      ${local_kernels_src_files}
      ${local_device_kernels_src_files}
    )

    target_include_directories(cpu_kernels PRIVATE
      ${local_kernels_inc_path}
    )

    target_link_libraries(cpu_kernels PRIVATE
      $<BUILD_INTERFACE:intf_pub_aicpu>
      -Wl,--whole-archive
      cpu_kernels_context_static
      -Wl,--no-whole-archive
      ascend_protobuf_static
      -Wl,--no-as-needed
      aicpu_sharder
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
      -Wl,-Bsymbolic
    )

    add_dependencies(cpu_kernels eigen_headers)

    set(OPS_AICPU_PATH "${INSTALL_PATH}/aicpu")

    cann_install(
      TARGET      cpu_kernels
      FILES       $<TARGET_FILE:cpu_kernels>
      DESTINATION "${INSTALL_PATH}/aicpu/aicpu_kernels_device"
    )

    set(COPY_TO_PATH ${OPS_AICPU_PATH}/${SUB_SO_WITH_VERSION})
    add_custom_command(TARGET cpu_kernels POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cpu_kernels> ${COPY_TO_PATH}
      COMMENT "[AICPU] copying cpu_kernels library to '${COPY_TO_PATH}'"
    )
  else()
    add_library(cpu_kernels_static STATIC
        ${local_kernels_src_files}
        ${local_host_kernels_src_files}
    )

    target_include_directories(cpu_kernels_static PRIVATE
        ${local_kernels_inc_path}
    )

    target_compile_definitions(cpu_kernels_static PRIVATE
        _FORTIFY_SOURCE=2
        $<$<NOT:$<STREQUAL:${PRODUCT_SIDE},device>>:LOG_CPP>
    )

    target_link_libraries(cpu_kernels_static PRIVATE
        $<BUILD_INTERFACE:intf_pub_aicpu>
    )

    target_compile_options(cpu_kernels_static PRIVATE
        -O2
        -std=c++11
        -ftrapv
        -fstack-protector-all
    )

    add_dependencies(cpu_kernels_static eigen_headers)

    set(cpu_kernels_static ${CMAKE_CURRENT_BINARY_DIR}/libcpu_kernels.a)
    set_target_properties(cpu_kernels_static
        PROPERTIES
        OUTPUT_NAME cpu_kernels
    )
  endif()
else()
set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(local_kernels_inc_path
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${TOP_DIR}/inc/aicpu/cpu_kernels
    ${TOP_DIR}/inc/external/aicpu
    ${TOP_DIR}/open_source/eigen
    ${TOP_DIR}/inc
    ${TOP_DIR}/libc_sec/include
    ${TOP_DIR}/abl/libc_sec/include
    ${TOP_DIR}/ace/comop/inc
)

add_library(cpu_kernels SHARED
    ${local_kernels_src_files}
    ${local_device_kernels_src_files}

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
    $<BUILD_INTERFACE:slog_headers>
    -ldl
    slog
    c_sec
    cpu_kernels_context
)

target_link_libraries(cpu_kernels PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:slog_headers>
    -Wl,--whole-archive
    cpu_kernels_context_static
    -Wl,--no-whole-archive
    ascend_protobuf_static
    -Wl,--no-as-needed
    slog
    c_sec
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
    -Wl,-Bsymbolic
)

target_compile_definitions(cpu_kernels_static PRIVATE
    _FORTIFY_SOURCE=2
    $<$<NOT:$<STREQUAL:${PRODUCT_SIDE},device>>:LOG_CPP>
)

target_link_libraries(cpu_kernels_static PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:slog_headers>
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

if("x${PRODUCT}" STREQUAL "xascend310" OR "x${PRODUCT}" STREQUAL "xascend910" OR "x${PRODUCT}" STREQUAL "xascend710" OR "x${PRODUCT}" STREQUAL "xascend310rc")
set(INSTALL_LIBRARY_DIR lib/${PRODUCT})
else()
set(INSTALL_LIBRARY_DIR lib/)
endif()

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
