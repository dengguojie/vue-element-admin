/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file test_stride_slice_grad_tiling.cpp
 * \brief
 */
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "all_ops.h"
#include "common/utils/ut_op_util.h"
#include "op_tiling/op_tiling_util.h"

using namespace std;

class stried_slice_grad_tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "stried_slice_grad_tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "stried_slice_grad_tiling TearDown" << std::endl;
  }
};

const int64_t profiling_test_num = 1;
static void run_case(std::vector<int64_t> input_shape, std::string data_dtype, std::vector<int32_t> const_shape_value,
                     std::vector<int32_t> const_begin, std::vector<int32_t> const_end,
                     std::vector<int32_t> const_strides, std::string compile_info, std::string expect_tiling,
                     std::string case_name) {
  using namespace ut_util;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("StridedSliceGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto test_op = op::StridedSliceGrad("StridedSliceGrad");
  int64_t input_len = input_shape.size();
  std::vector<int64_t> const_shape{input_len};

  TENSOR_INPUT_WITH_SHAPE(test_op, dy, input_shape, StringToDtype(data_dtype), FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, shape, const_shape, DT_INT32, FORMAT_ND, const_shape_value);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, begin, const_shape, DT_INT32, FORMAT_ND, const_begin);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, end, const_shape, DT_INT32, FORMAT_ND, const_end);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, strides, const_shape, DT_INT32, FORMAT_ND, const_strides);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(test_op, iter->second, compile_info, runInfo);
  if (expect_tiling != "") {
    EXPECT_EQ(to_string_int64(runInfo.GetAllTilingData()), expect_tiling);
  }
  for (int64_t i = 0; i < profiling_test_num; i++) {
    RUN_TILING_V3(test_op, iter->second, compile_info, runInfo);
  }
}

TEST_F(stried_slice_grad_tiling, stried_slice_grad_tiling_no_mask) {
  vector<vector<int64_t>> input_shapes = {
      {4}, {4}, {4}, {4}, {4, 4, 4, 4},
  };

  vector<string> dtypes = {"int32", "int32", "int32", "int32", "float16"};

  vector<int32_t> shape_value = {3, 3, 3, 3};
  vector<int32_t> begin = {1, 1, 1, 1};
  vector<int32_t> end = {3, 3, 3, 3};
  vector<int32_t> strides = {1, 1, 1, 1};

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65536, \"core_num\": 32, \"dtype_rate\": 2, \"begin_mask\": 0, \"end_mask\": 0, "
      "\"ellipsis_mask\": 0, \"new_axis_mask\": 0, \"shrink_axis_mask\": 0}}";
  std::string expect_tiling = "2 1 1 1 1 2 2 2 4 0 0 0 0 0 0 0 0 1 0 1 0 1 0 2 0 2 ";
  run_case(input_shapes[4], dtypes[4], shape_value, begin, end, strides, compileInfo, expect_tiling,
           this->test_info_->name());
}
