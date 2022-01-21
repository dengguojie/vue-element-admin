# Copyright 2019-2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

project(cpu_kernels_ut)

set(CMAKE_CXX_STANDARD 11)

if(DEFINED ENV{ASCEND_CUSTOM_PATH})
  set(AICPU_OPP_ENV $ENV{ASCEND_CUSTOM_PATH}/opp/op_impl/built-in/aicpu/aicpu_kernel)
else()
  set(AICPU_OPP_ENV /usr/local/Ascend/opp/op_impl/built-in/aicpu/aicpu_kernel)
endif()

set(_proto_srcs
  "${CMAKE_CURRENT_SOURCE_DIR}/proto/cpu_tensor.proto"
  "${CMAKE_CURRENT_SOURCE_DIR}/proto/cpu_attr.proto"
  "${CMAKE_CURRENT_SOURCE_DIR}/proto/cpu_tensor_shape.proto"
  "${CMAKE_CURRENT_SOURCE_DIR}/proto/cpu_node_def.proto"
)

# the output path of generated files
set(_proto_include "${PROTO_BINARY_DIR}/opp")
protobuf_generate(opp _proto_cc _proto_h ${_proto_srcs})

set(CPU_KERNELS_DIR ${CANN_ROOT}/ops/built-in/aicpu/impl)
set(CPU_CONTEXT_DIR ${CANN_ROOT}/ops/built-in/aicpu/context)

set(_cpu_context
  ${CPU_CONTEXT_DIR}/cpu_proto/node_def.cc
  ${CPU_CONTEXT_DIR}/cpu_proto/node_def_impl.cc
  ${CPU_CONTEXT_DIR}/cpu_proto/tensor.cc
  ${CPU_CONTEXT_DIR}/cpu_proto/tensor_impl.cc
  ${CPU_CONTEXT_DIR}/cpu_proto/tensor_shape.cc
  ${CPU_CONTEXT_DIR}/cpu_proto/tensor_shape_impl.cc
  ${CPU_CONTEXT_DIR}/cpu_proto/attr_value.cc
  ${CPU_CONTEXT_DIR}/cpu_proto/attr_value_impl.cc
  ${CPU_CONTEXT_DIR}/common/device.cc
  ${CPU_CONTEXT_DIR}/common/context.cc
  ${CPU_CONTEXT_DIR}/common/device_cpu_kernel.cc
  ${CPU_CONTEXT_DIR}/common/cpu_kernel_register.cc
  ${CPU_CONTEXT_DIR}/common/cpu_kernel_utils.cc
  ${CPU_CONTEXT_DIR}/common/host_sharder.cc
  ${CPU_CONTEXT_DIR}/common/device_sharder.cc
  ${CPU_CONTEXT_DIR}/common/eigen_threadpool.cc
  ${CPU_CONTEXT_DIR}/common/cpu_kernel_cache.cc
  ${CPU_CONTEXT_DIR}/common/async_cpu_kernel.cc
  ${CPU_CONTEXT_DIR}/common/async_event_util.cc
  ${CPU_CONTEXT_DIR}/stub/aicpu_sharder.cc
)

set(_cpu_kernels_src
  ${CPU_KERNELS_DIR}/kernels/device/avg_pool1d_avg_matrix.cc
  ${CPU_KERNELS_DIR}/kernels/device/mem_cpy.cc
  ${CPU_KERNELS_DIR}/kernels/device/where.cc
  ${CPU_KERNELS_DIR}/kernels/device/stateless_drop_out_gen_mask.cc
  ${CPU_KERNELS_DIR}/kernels/host/assign.cc
  ${CPU_KERNELS_DIR}/kernels/host/format_transfer/format_transfer_fractal_nz.cc
  ${CPU_KERNELS_DIR}/kernels/host/format_transfer/format_transfer_fractal_z.cc
  ${CPU_KERNELS_DIR}/kernels/host/format_transfer/format_transfer_fractalz_3d.cc
  ${CPU_KERNELS_DIR}/kernels/host/format_transfer/format_transfer_transpose.cc
  ${CPU_KERNELS_DIR}/kernels/host/format_transfer/format_transfer_utils.cc
  ${CPU_KERNELS_DIR}/kernels/host/format_transfer/register_format_transfer.cc
  ${CPU_KERNELS_DIR}/kernels/host/gemm.cc
  ${CPU_KERNELS_DIR}/kernels/host/matmul.cc
  ${CPU_KERNELS_DIR}/kernels/host/matrix_triangular_solve.cc
  ${CPU_KERNELS_DIR}/kernels/host/no_op.cc
  ${CPU_KERNELS_DIR}/kernels/host/trans_data_rnn.cc
  ${CPU_KERNELS_DIR}/kernels/host/transdata.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/acos.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/acosh.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/adaptive_max_pool_2d.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/add.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/asin.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/asinh.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/atan.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/atan2.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/atanh.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/betainc.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/cache_swap_table.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/calc_buckets_limit_and_offset.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/candidate_sampler.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/case_condition.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/cast.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/ceil.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/check_numerics.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/combinations.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/compute_accidental_hits.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/concatv2.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/conj.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/coordinates_1d_to_2d.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/cosh.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/crop_and_resize.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/reduce_sum.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/cumsum.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/custom/research/diag.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/custom/research/diag_part.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/custom/research/fixed_unigram_candidate_sampler.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/custom/research/ifft.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/custom/research/lu.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/custom/research/minimum.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/custom/research/qr.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/custom/research/random_gamma.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/custom/research/sigmoid_grad.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/custom/research/sin.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/custom/research/sinh.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/custom/research/softplus.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/custom/research/sqrtgrad.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/custom/research/svd.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/custom/research/unpack.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/dynamic_stitch.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/edit_distance.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/elugrad.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/embedding_lookup.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/equal.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/expand.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/exponential.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/fill.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/floor.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/floordiv.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/gatherv2.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/geometric.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/get_dynamic_dims.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/greater.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/greater_equal.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/identity.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/image_warp_offsets.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/index_to_addr.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/inplace_top_k_distance.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/invgrad.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/is_close.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/is_inf.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/is_nan.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/less.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/log1p.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/log_matrix_determinant.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/logging.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/logsoftmaxv2.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/masked_select.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/masked_select_grad.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/mul.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/multinomial_alias_draw.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/multinomial_alias_setup.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/neg.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/nms_with_mask.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/non_max_suppression_v3_kernels.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/non_zero_with_value_shape.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/not_equal.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/ones_like.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/pinverse.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/poisson.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/random_uniform.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/realdiv.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/reshape.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/resize_bilinear.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/resize_bilinear_grad.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/reverse_sequence.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/rightshift.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/round.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/rsqrt.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/scatter_elements.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/search_sorted.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/sigmoid.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/sign.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/slice_write.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/softmaxv2.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/sparse_to_dense.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/spatial_transformer.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/split_v.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/square.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/squareddifference.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/strided_slice.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/strided_slice_v2.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/sub.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/tan.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/tanhgrad.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/top_k.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/top_k_pq_distance.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/zeroslike.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/zeta.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/tile_with_axis.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/linspace.cc
  ${CPU_KERNELS_DIR}/utils/allocator_utils.cc
  ${CPU_KERNELS_DIR}/utils/bcast.cc
  ${CPU_KERNELS_DIR}/utils/broadcast_iterator.cc
  ${CPU_KERNELS_DIR}/utils/eigen_tensor.cc
  ${CPU_KERNELS_DIR}/utils/kernel_util.cc
  ${CPU_KERNELS_DIR}/utils/range_sampler.cc
  ${CPU_KERNELS_DIR}/utils/sparse_group.cc
  ${CPU_KERNELS_DIR}/utils/sparse_tensor.cc
)

set(cpu_kernels_llt_src
  ${_cpu_context}
  ${_cpu_kernels_src}
  ${_proto_cc}
  ${_proto_h}
)

add_library(cpu_kernels_llt STATIC
  ${cpu_kernels_llt_src}
)

add_dependencies(cpu_kernels_llt eigen_headers)

set(cpu_kernels_llt_include
  ${CANN_ROOT}/third_party/air/base/exec_runtime
  ${CPU_KERNELS_DIR}/kernels/host/format_transfer
  ${CPU_KERNELS_DIR}
  ${CPU_KERNELS_DIR}/utils
  ${CPU_CONTEXT_DIR}/inc
  ${CPU_CONTEXT_DIR}/common
  ${CPU_CONTEXT_DIR}/stub
  ${CPU_CONTEXT_DIR}/cpu_proto
  ${AICPU_OPP_ENV}/inc
  ${_proto_include}
  ${Protobuf_INCLUDE}
  ${EIGEN_INCLUDE}
  ${FWKACLLIB_INCLUDE}
  ${FWKACLLIB_INCLUDE}/aicpu/common
)

target_include_directories(cpu_kernels_llt PUBLIC
  ${cpu_kernels_llt_include}
)

target_compile_options(cpu_kernels_llt PUBLIC
  -D_GLIBCXX_USE_CXX11_ABI=0
  -g
  -Dgoogle=ascend_private
  -DRUN_TEST
)

target_link_libraries(cpu_kernels_llt
  PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
  PUBLIC
    ascend_protobuf
    c_sec
    -ldl
)

set(_cpu_kernels_ut_files
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../aicpu/context/common/node_def_builder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/aicpu_read_file.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/aicpu_test_utils.cpp
  testcase/SpatialTransformer/src/SpatialTransformer_utest.cpp
  testcase/acos/src/acos_utest.cpp
  testcase/acosh/src/acosh_utest.cpp
  testcase/adaptiveMaxPool/adaptiveMaxPool2d_utest.cpp
  testcase/add/src/add_utest.cpp
  testcase/asin/src/asin_utest.cpp
  testcase/asinh/src/asinh_utest.cpp
  testcase/assign/src/assign_utest.cpp
  testcase/atan/src/atan_utest.cpp
  testcase/atan2/src/atan2_utest.cpp
  testcase/atanh/src/atanh_utest.cpp
  testcase/avg_pool1d_avg_matrix/src/avg_pool1d_avg_matrix_utest.cpp
  testcase/betainc/src/betainc_utest.cpp
  testcase/cache_swap_table/src/cache_swap_table_kernel_utest.cpp
  testcase/calc_buckets_limit_and_offset/src/calc_buckets_limit_and_offset_utest.cpp
  testcase/candidate_sampler/src/candidate_sampler_kernel_utest.cpp
  testcase/case_condition/src/case_condition_utest.cpp
  testcase/cast/src/cast_utest.cpp
  testcase/ceil/src/ceil_utest.cpp
  testcase/check_numerics/src/check_numerics_utest.cpp
  testcase/combinations/src/combinations_utest.cpp
  testcase/tile_with_axis/src/tile_with_axis_utest.cpp
  testcase/linspace/src/linspace_utest.cpp
  testcase/common/src/async_cpu_kernel_utest.cpp
  testcase/common/src/async_event_util_utest.cpp
  testcase/common/src/attr_value_utest.cpp
  testcase/common/src/context_utest.cpp
  testcase/common/src/cpu_kernels_register_utest.cpp
  testcase/common/src/device_utest.cpp
  testcase/common/src/eigen_threadpool_utest.cpp
  testcase/common/src/kernel_util_utest.cpp
  testcase/common/src/node_def_utest.cpp
  testcase/common/src/sharder_utest.cpp
  testcase/common/src/tensor_utest.cpp
  testcase/compute_accidental_hits/src/compute_accidental_hits_kernel_utest.cpp
  testcase/concatv2/src/concatv2_utest.cpp
  testcase/conj/src/conj_utest.cpp
  testcase/coordinates_1d_to_2d/src/coordinates_1d_to_2d_utest.cpp
  testcase/cosh/src/cosh_utest.cpp
  testcase/crop_and_resize/src/crop_and_resize_kernel_utest.cpp
  testcase/reduce_sum/src/reduce_sum_utest.cpp
  testcase/cumsum/src/cumsum_utest.cpp
  testcase/diag/src/diag_utest.cpp
  testcase/diag_part/src/diag_part_utest.cpp
  testcase/dynamic_stitch/src/dynamic_stitch_kernel_utest.cpp
  testcase/edit_distance/src/edit_distance_kernel_utest.cpp
  testcase/elugrad/src/elugrad_utest.cpp
  testcase/embedding_lookup/src/embedding_lookup_kernel_utest.cpp
  testcase/equal/src/equal_utest.cpp
  testcase/expand/src/expand_utest.cpp
  testcase/exponential/src/exponential_utest.cpp
  testcase/fill/src/fill_utest.cpp
  testcase/fixed_unigram_candidate_sampler/src/fixed_unigram_candidate_sampler_utest.cpp
  testcase/floor/src/floor_utest.cpp
  testcase/floordiv/src/floordiv_utest.cpp
  testcase/gatherv2/src/gatherv2_utest.cpp
  testcase/gemm/src/gemm_utest.cpp
  testcase/geometric/src/geometric_utest.cpp
  testcase/get_dynamic_dims/src/get_dynamic_dims_utest.cpp
  testcase/greater/src/greater_utest.cpp
  testcase/greater_equal/src/greater_equal_utest.cpp
  testcase/identity/src/identity_utest.cpp
  testcase/ifft/src/ifft_utest.cpp
  testcase/image_warp_offsets/src/image_warp_offsets_utest.cpp
  testcase/index_to_addr/src/index_to_addr_utest.cpp
  testcase/inplace_top_k_distance/src/inplace_top_k_distance_utest.cpp
  testcase/invgrad/src/invgrad_utest.cpp
  testcase/is_close/src/is_close_utest.cpp
  testcase/is_inf/src/is_inf_utest.cpp
  testcase/is_nan/src/is_nan_utest.cpp
  testcase/less/src/less_utest.cpp
  testcase/log1p/src/log1p_utest.cpp
  testcase/log_matrix_determinant/src/log_matrix_determinant_utest.cpp
  testcase/logging/src/logging_utest.cpp
  testcase/logsoftmaxv2/src/logsoftmaxv2_utest.cpp
  testcase/lu/src/lu_utest.cpp
  testcase/masked_select/src/masked_select_utest.cpp
  testcase/masked_select_grad/src/masked_select_grad_utest.cpp
  testcase/matmul/src/matmul_utest.cpp
  testcase/matrix_triangular_solve/src/matrix_triangular_solve_utest.cpp
  testcase/minimum/src/minimum_utest.cpp
  testcase/mul/src/mul_utest.cpp
  testcase/multinomial_alias_draw/multinomial_alias_draw_utest.cpp
  testcase/multinomial_alias_setup/src/multinomial_alias_setup_utests.cpp
  testcase/neg/src/neg_utest.cpp
  testcase/no_op/src/no_op_utest.cpp
  testcase/nms_with_mask/src/nms_with_mask_utest.cpp
  testcase/non_max_suppression_v3/src/non_max_suppression_v3_utest.cpp
  testcase/non_zero_with_value_shape/src/non_zero_with_value_shape_utest.cpp
  testcase/not_equal/src/not_equal_utest.cpp
  testcase/ones_like/src/ones_like_utest.cpp
  testcase/pinverse/src/pinverse_utest.cpp
  testcase/poisson/src/poisson_utest.cpp
  testcase/qr/src/qr_utest.cpp
  testcase/random_gamma/src/random_gamma_utest.cpp
  testcase/random_uniform/src/random_uniform_utest.cpp
  testcase/realdiv/src/realdiv_utest.cpp
  testcase/reshape/src/reshape_utest.cpp
  testcase/resize_bilinear/src/resize_bilinear_grad_utest.cpp
  testcase/resize_bilinear/src/resize_bilinear_utest.cpp
  testcase/reverse_sequence/src/reverse_sequence_kernel_utest.cpp
  testcase/rightshift/src/rightshift_utest.cpp
  testcase/round/src/round_utest.cpp
  testcase/rsqrt/src/rsqrt_utest.cpp
  testcase/scatter_elements/src/scatter_elements_utest.cpp
  testcase/search_sorted/src/search_sorted_kernel_utest.cpp
  testcase/sigmoid/src/sigmoid_utest.cpp
  testcase/sign/src/sign_utest.cpp
  testcase/sigmoid_grad/src/sigmoid_grad_utest.cpp
  testcase/sin/src/sin_utest.cpp
  testcase/sinh/src/sinh_utest.cpp
  testcase/slice_write/src/slice_write_utest.cpp
  testcase/softmaxv2/src/softmaxv2_utest.cpp
  testcase/softplus/src/softplus_utest.cpp
  testcase/sparse_to_dense/src/sparse_to_dense_utest.cpp
  testcase/split_v/src/split_v_utest.cpp
  testcase/sqrtgrad/src/sqrtgrad_utest.cpp
  testcase/square/src/square_utest.cpp
  testcase/squareddifference/src/squareddifference_utest.cpp
  testcase/stateless_drop_out_gen_mask/src/stateless_drop_out_gen_mask_ut.cpp
  testcase/strided_slice/src/strided_slice_utest.cpp
  testcase/strided_slice_v2/src/strided_slice_v2_utest.cpp
  testcase/sub/src/sub_utest.cpp
  testcase/svd/src/svd_utest.cpp
  testcase/tan/src/tan_utest.cpp
  testcase/tanhgrad/src/tanhgrad_utest.cpp
  testcase/tile_with_axis/src/tile_with_axis_utest.cpp
  testcase/top_k/src/top_k_utest.cpp
  testcase/top_k_pq_distance/src/top_k_pq_distance_utest.cpp
  testcase/trans_data/src/format_transfer_hwcn_fractalz_utest.cpp
  testcase/trans_data/src/trans_data_utest.cpp
  testcase/trans_data_rnn/src/trans_data_rnn_utest.cpp
  testcase/unpack/src/unpack_utest.cpp
  testcase/where/src/where_utest.cpp
  testcase/zeroslike/src/zeroslike_utest.cpp
  testcase/zeta/src/zeta_utest.cpp
  testcase/add_block/src/add_block_utest.cpp
)

add_executable(cpu_kernels_ut
  ${_cpu_kernels_ut_files}
)

target_include_directories(cpu_kernels_ut PRIVATE
  ${GTEST_INCLUDE}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../utils
)

target_link_libraries(cpu_kernels_ut PRIVATE
  -Wl,--whole-archive
  cpu_kernels_llt
  -Wl,--no-whole-archive
)

target_compile_definitions(cpu_kernels_ut PUBLIC
  D_GLIBCXX_USE_CXX11_ABI=0
  google=ascend_private
  RUN_TEST
)

if(NOT UT_NO_EXEC)
  add_custom_command(
    TARGET  cpu_kernels_ut POST_BUILD
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR}/testcase && python3.7 run_all_gen_data.py
    COMMAND cpu_kernels_ut
    COMMENT "Run cpu_kernels_ut"
  )
endif()

