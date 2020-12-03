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
  ${CPU_KERNELS_DIR}/utils/bcast.cc
  ${CPU_KERNELS_DIR}/utils/eigen_tensor.cc
  ${CPU_KERNELS_DIR}/utils/kernel_util.cc
  ${CPU_KERNELS_DIR}/utils/sparse_group.cc
  ${CPU_KERNELS_DIR}/utils/sparse_tensor.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/concatv2.cc
  ${CPU_KERNELS_DIR}/kernels/host/transdata.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/get_dynamic_dims.cc
  ${CPU_KERNELS_DIR}/kernels/normalized/round.cc
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
)

set(_cpu_kernels_ut_files
  ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/node_def_builder.cpp
  testcase/concatv2_utest.cpp
  testcase/trans_data_utest.cpp
  testcase/get_dynamic_dims_utest.cpp
  testcase/round_utest.cpp
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
#[[
set(AICPU_UTEST "${INSTALL_PATH}/tests/ops/ut/aicpu")
cann_install(
  TARGET      cpu_kernels_ut
  FILES       $<TARGET_FILE:cpu_kernels_ut>
  DESTINATION "${AICPU_UTEST}"
)
]]
add_custom_command(
  TARGET  cpu_kernels_ut POST_BUILD
  COMMAND cpu_kernels_ut
  COMMENT "Run cpu_kernels_ut"
)
