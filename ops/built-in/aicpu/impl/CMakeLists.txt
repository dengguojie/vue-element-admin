set(SUB_SO_WITH_VERSION libcpu_kernels_v1.0.1.so)

set(local_kernels_src_files
    kernels/normalized/acos.cc
    kernels/normalized/acosh.cc
    kernels/normalized/adaptive_max_pool_2d.cc
    kernels/normalized/add.cc
    kernels/normalized/asin.cc
    kernels/normalized/asinh.cc
    kernels/normalized/atan.cc
    kernels/normalized/atan2.cc
    kernels/normalized/atanh.cc
    kernels/normalized/betainc.cc
    kernels/normalized/cache_swap_table.cc
    kernels/normalized/calc_buckets_limit_and_offset.cc
    kernels/normalized/candidate_sampler.cc
    kernels/normalized/case_condition.cc
    kernels/normalized/cast.cc
    kernels/normalized/ceil.cc
    kernels/normalized/check_numerics.cc
    kernels/normalized/combinations.cc
    kernels/normalized/compute_accidental_hits.cc
    kernels/normalized/concatv2.cc
    kernels/normalized/conj.cc
    kernels/normalized/coordinates_1d_to_2d.cc
    kernels/normalized/cosh.cc
    kernels/normalized/crop_and_resize.cc
    kernels/normalized/cumsum.cc
    kernels/normalized/dynamic_stitch.cc
    kernels/normalized/edit_distance.cc
    kernels/normalized/elugrad.cc
    kernels/normalized/embedding_lookup.cc
    kernels/normalized/equal.cc
    kernels/normalized/expanddims.cc
    kernels/normalized/expand.cc
    kernels/normalized/exponential.cc
    kernels/normalized/fill.cc
    kernels/normalized/floor.cc
    kernels/normalized/floordiv.cc
    kernels/normalized/gatherv2.cc
    kernels/normalized/geometric.cc
    kernels/normalized/get_dynamic_dims.cc
    kernels/normalized/greater.cc
    kernels/normalized/greater_equal.cc
    kernels/normalized/identity.cc
    kernels/normalized/image_warp_offsets.cc
    kernels/normalized/index_to_addr.cc
    kernels/normalized/inplace_top_k_distance.cc
    kernels/normalized/invgrad.cc
    kernels/normalized/is_close.cc
    kernels/normalized/is_inf.cc
    kernels/normalized/is_nan.cc
    kernels/normalized/less.cc
    kernels/normalized/log1p.cc
    kernels/normalized/log_matrix_determinant.cc
    kernels/normalized/logging.cc
    kernels/normalized/logsoftmaxv2.cc
    kernels/normalized/masked_select.cc
    kernels/normalized/masked_select_grad.cc
    kernels/normalized/mul.cc
    kernels/normalized/multinomial_alias_draw.cc
    kernels/normalized/multinomial_alias_setup.cc
    kernels/normalized/neg.cc
    kernels/normalized/nms_with_mask.cc
    kernels/normalized/non_max_suppression_v3_kernels.cc
    kernels/normalized/non_zero_with_value_shape.cc
    kernels/normalized/not_equal.cc
    kernels/normalized/ones_like.cc
    kernels/normalized/pinverse.cc
    kernels/normalized/poisson.cc
    kernels/normalized/random_uniform.cc
    kernels/normalized/realdiv.cc
    kernels/normalized/reshape.cc
    kernels/normalized/resize_bilinear.cc
    kernels/normalized/resize_bilinear_grad.cc
    kernels/normalized/reverse_sequence.cc
    kernels/normalized/rightshift.cc
    kernels/normalized/round.cc
    kernels/normalized/rsqrt.cc
    kernels/normalized/scatter_elements.cc
    kernels/normalized/search_sorted.cc
    kernels/normalized/sigmoid.cc
    kernels/normalized/sign.cc
    kernels/normalized/slice_write.cc
    kernels/normalized/softmaxv2.cc
    kernels/normalized/sparse_to_dense.cc
    kernels/normalized/spatial_transformer.cc
    kernels/normalized/split_v.cc
    kernels/normalized/square.cc
    kernels/normalized/squareddifference.cc
    kernels/normalized/strided_slice.cc
    kernels/normalized/strided_slice_v2.cc
    kernels/normalized/sub.cc
    kernels/normalized/tan.cc
    kernels/normalized/tanhgrad.cc
    kernels/normalized/top_k.cc
    kernels/normalized/top_k_pq_distance.cc
    kernels/normalized/top_k_v2_d.cc
    kernels/normalized/zeroslike.cc
    kernels/normalized/zeta.cc
    kernels/normalized/tile_with_axis.cc
    kernels/normalized/reduce_sum.cc
    kernels/normalized/linspace.cc
    kernels/normalized/environ_create.cc
    kernels/normalized/environ_get.cc
    kernels/normalized/environ_set.cc
    kernels/normalized/environ_destroy_all.cc
    utils/allocator_utils.cc
    utils/bcast.cc
    utils/broadcast_iterator.cc
    utils/eigen_tensor.cc
    utils/kernel_util.cc
    utils/range_sampler.cc
    utils/sparse_group.cc
    utils/sparse_tensor.cc
    utils/environ_manager.cc
)
if(WITH_CUSTOM_OP)
    list(APPEND local_kernels_src_files
        kernels/normalized/custom/research/abs.cc
        kernels/normalized/custom/research/diag.cc
        kernels/normalized/custom/research/diag_part.cc
        kernels/normalized/custom/research/fixed_unigram_candidate_sampler.cc
        kernels/normalized/custom/research/ifft.cc
        kernels/normalized/custom/research/lu.cc
        kernels/normalized/custom/research/minimum.cc
        kernels/normalized/custom/research/qr.cc
        kernels/normalized/custom/research/random_gamma.cc
        kernels/normalized/custom/research/sigmoid_grad.cc
        kernels/normalized/custom/research/sin.cc
        kernels/normalized/custom/research/sinh.cc
        kernels/normalized/custom/research/softplus.cc
        kernels/normalized/custom/research/softplusgrad.cc
        kernels/normalized/custom/research/sqrtgrad.cc
        kernels/normalized/custom/research/svd.cc
        kernels/normalized/custom/research/unpack.cc
    )
endif()
message("# ADD with custom op files: " ${WITH_CUSTOM_OP}\r\n ${local_kernels_src_files})

set(local_host_kernels_src_files
    kernels/host/assign.cc
    kernels/host/format_transfer/format_transfer_fractal_nz.cc
    kernels/host/format_transfer/format_transfer_fractal_z.cc
    kernels/host/format_transfer/format_transfer_fractalz_3d.cc
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
    kernels/device/stateless_drop_out_gen_mask.cc
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
      $<$<STREQUAL:${TARGET_LINUX_DISTRIBUTOR_ID},lhisiandroid>:RUN_ANDROID>
    )

    target_compile_options(cpu_kernels PRIVATE
      -O2
      -std=c++14
      -fstack-protector-all
      -fvisibility-inlines-hidden
      -fvisibility=hidden
      -frename-registers
      -fpeel-loops
      -Werror
      -Wno-return-type
      -Wno-attributes
      -DEIGEN_NO_DEBUG
	  -DEIGEN_MPL2_ONLY
      -DNDEBUG
      -DEIGEN_HAS_CXX11_MATH
      -DEIGEN_OS_GNULINUX
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
        $<$<STREQUAL:${TARGET_LINUX_DISTRIBUTOR_ID},lhisiandroid>:RUN_ANDROID>
    )

    target_link_libraries(cpu_kernels_static PRIVATE
        $<BUILD_INTERFACE:intf_pub_aicpu>
    )

    target_compile_options(cpu_kernels_static PRIVATE
        -O2
        -std=c++11
        -ftrapv
        -fstack-protector-all
        -DEIGEN_MPL2_ONLY
    )

    add_dependencies(cpu_kernels_static eigen_headers c_sec)

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
    $<$<STREQUAL:${TARGET_LINUX_DISTRIBUTOR_ID},lhisiandroid>:RUN_ANDROID>
)

target_compile_options(cpu_kernels PRIVATE
    -O2
    -std=c++14
    -fstack-protector-all
    -fvisibility-inlines-hidden
    -fvisibility=hidden
    -frename-registers
    -fpeel-loops
    -DEIGEN_NO_DEBUG
	-DEIGEN_MPL2_ONLY
    -DNDEBUG
    -DEIGEN_HAS_CXX11_MATH
    -DEIGEN_OS_GNULINUX
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
    $<$<STREQUAL:${TARGET_LINUX_DISTRIBUTOR_ID},lhisiandroid>:RUN_ANDROID>
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
    -DEIGEN_MPL2_ONLY
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
    -DEIGEN_MPL2_ONLY
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
