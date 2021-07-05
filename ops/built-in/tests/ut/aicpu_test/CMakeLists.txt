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
  ${CPU_CONTEXT_DIR}/stub/aicpu_sharder.cc
)

set(_cpu_kernels_src
  ${CPU_KERNELS_DIR}/kernels/device/avg_pool1d_avg_matrix.cc
  ${CPU_KERNELS_DIR}/kernels/device/where.cc
  ${CPU_KERNELS_DIR}/kernels/host/format_transfer/format_transfer_fractal_nz.cc
  ${CPU_KERNELS_DIR}/kernels/host/format_transfer/format_transfer_fractal_z.cc
  ${CPU_KERNELS_DIR}/kernels/host/format_transfer/format_transfer_fractalz_3d.cc
  ${CPU_KERNELS_DIR}/kernels/host/format_transfer/format_transfer_transpose.cc
  ${CPU_KERNELS_DIR}/kernels/host/format_transfer/format_transfer_utils.cc
  ${CPU_KERNELS_DIR}/kernels/host/format_transfer/register_format_transfer.cc
  ${CPU_KERNELS_DIR}/kernels/host/no_op.cc
  ${CPU_KERNELS_DIR}/kernels/host/assign.cc
  ${CPU_KERNELS_DIR}/kernels/host/transdata.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/adaptive_max_pool_2d.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/add.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/asin.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/cast.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/ceil.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/concatv2.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/dynamic_stitch.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/equal.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/fill.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/get_dynamic_dims.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/identity.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/image_warp_offsets.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/less.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/logging.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/masked_select.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/masked_select_grad.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/mul.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/not_equal.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/random_uniform.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/realdiv.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/reshape.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/round.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/scatter_elements.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/search_sorted.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/sparse_to_dense.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/spatial_transformer.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/split_v.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/strided_slice.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/strided_slice_v2.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/top_k.cc
  ${CPU_KERNELS_DIR}/utils/bcast.cc
  ${CPU_KERNELS_DIR}/utils/broadcast_iterator.cc
  ${CPU_KERNELS_DIR}/utils/eigen_tensor.cc
  ${CPU_KERNELS_DIR}/utils/kernel_util.cc
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
  -O0
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
  ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/aicpu_read_file.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/aicpu_test_utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/node_def_builder.cpp
  testcase/adaptiveMaxPool/adaptiveMaxPool2d_utest.cpp
  testcase/add/src/add_utest.cpp
  testcase/asin/src/asin_utest.cpp
  testcase/avg_pool1d_avg_matrix/src/avg_pool1d_avg_matrix_utest.cpp
  testcase/cast/src/cast_utest.cpp
  testcase/ceil/src/ceil_utest.cpp
  testcase/common/src/attr_value_utest.cpp
  testcase/common/src/context_utest.cpp
  testcase/common/src/cpu_kernels_register_utest.cpp
  testcase/common/src/device_utest.cpp
  testcase/common/src/eigen_threadpool_utest.cpp
  testcase/common/src/kernel_util_utest.cpp
  testcase/common/src/node_def_utest.cpp
  testcase/common/src/sharder_utest.cpp
  testcase/common/src/tensor_utest.cpp
  testcase/concatv2/src/concatv2_utest.cpp
  testcase/dynamic_stitch/src/dynamic_stitch_kernel_utest.cpp
  testcase/equal/src/equal_utest.cpp
  testcase/fill/src/fill_utest.cpp
  testcase/get_dynamic_dims/src/get_dynamic_dims_utest.cpp
  testcase/identity/src/identity_utest.cpp
  testcase/image_warp_offsets/src/image_warp_offsets_utest.cpp
  testcase/less/src/less_utest.cpp
  testcase/logging/src/logging_utest.cpp
  testcase/masked_select/src/masked_select_utest.cpp
  testcase/masked_select_grad/src/masked_select_grad_utest.cpp
  testcase/mul/src/mul_utest.cpp
  testcase/no_op/src/no_op_utest.cpp
  testcase/assign/src/assign_utest.cpp
  testcase/not_equal/src/not_equal_utest.cpp
  testcase/random_uniform/src/random_uniform_utest.cpp
  testcase/realdiv/src/realdiv_utest.cpp
  testcase/reshape/src/reshape_utest.cpp
  testcase/round/src/round_utest.cpp
  testcase/scatter_elements/src/scatter_elements_utest.cpp
  testcase/search_sorted/src/search_sorted_kernel_utest.cpp
  testcase/sparse_to_dense/src/sparse_to_dense_utest.cpp
  testcase/SpatialTransformer/src/SpatialTransformer_utest.cpp
  testcase/split_v/src/split_v_utest.cpp
  testcase/strided_slice/src/strided_slice_utest.cpp
  testcase/strided_slice_v2/src/strided_slice_v2_utest.cpp
  testcase/top_k/src/top_k_utest.cpp
  testcase/trans_data/src/format_transfer_hwcn_fractalz_utest.cpp
  testcase/trans_data/src/trans_data_utest.cpp
  testcase/trans_data/src/trans_data_utest.cpp
  testcase/where/src/where_utest.cpp
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

