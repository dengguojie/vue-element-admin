/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file test_moving_sum_with_sigmoid_tiling.cpp
 * \brief
 */
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include <graph/utils/type_utils.h>
#define private public
#include "register/op_tiling_registry.h"
#include "all_ops.h"
#include "test_common.h"
#include "op_tiling/op_tiling_util.h"
#include "common/utils/ut_op_util.h"

using namespace std;
using namespace ge;
class MovingSumWithSigmoidTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MovingSumWithSigmoidTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MovingSumWithSigmoidTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream& tiling_data) {
  auto data = tiling_data.str();
  string result;
  int32_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }

  return result;
}

TEST_F(MovingSumWithSigmoidTiling, tset_0) {
  using namespace ut_util;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("MovingSumWithSigmoid");
 
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compile_info = "{\"vars\": {\"core_num\": 32}}";
  auto test_op = op::MovingSumWithSigmoid("MovingSumWithSigmoid");

  std::vector<int64_t> input_shape_0{5120};
  std::vector<int64_t> input_shape_1{1};

  TENSOR_INPUT_WITH_SHAPE(test_op, alpha, input_shape_0, DT_FLOAT,
                          FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(test_op, energy, input_shape_0, DT_FLOAT,
                          FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(test_op, beam_size, input_shape_1, DT_INT32,
                          FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(test_op, frame_size, input_shape_1, DT_INT32,
                          FORMAT_ND, {});

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(test_op, iter->second, compile_info, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 ");
}