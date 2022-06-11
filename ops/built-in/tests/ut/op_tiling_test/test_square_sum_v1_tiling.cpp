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
#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"

using namespace std;
using namespace ge;
#include "common/utils/ut_op_util.h"
using namespace ut_util;

class SquareSumV1Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SquareSumV1Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SquareSumV1Tiling TearDown" << std::endl;
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

TEST_F(SquareSumV1Tiling, test_SquareSumV1Tiling_1) {
  std::string op_name = "SquareSumV1";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  // dynamic_tile_d_llt_case_1
std::string compileInfo = R"({ "_pattern": "CommReduce",
"_reduce_vars": {"1000400": [20000, 20001,30000, 40000]},
 "push_status": 0, "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5, 4, 9], "_ub_info": [16256, 16000, 16256],
 "_ub_info_rf": [16256, 16000, 16256]})";
  std::vector<int64_t> input{16, 16, 16, 16};
  std::vector<int64_t> output{1};
  std::vector<int32_t> axis = {0, 1, 2, 3};
  ge::DataType input_dtype = ge::DT_FLOAT;

  auto opParas = op::SquareSumV1("SquareSumV1");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, input_dtype, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, input_dtype, ge::FORMAT_ND, {});
  opParas.SetAttr("axis", axis);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 65536 2048 1 ");
}

TEST_F(SquareSumV1Tiling, test_SquareSumV1Tiling_2) {
  std::string op_name = "SquareSumV1";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  // dynamic_tile_d_llt_case_1
std::string compileInfo = R"({ "_pattern": "CommReduce",
"_reduce_vars": {"1000400": [20000, 20001,30000, 40000]},
 "push_status": 0, "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5, 4, 9], "_ub_info": [16256, 16000, 16256],
 "_ub_info_rf": [16256, 16000, 16256]})";
  std::vector<int64_t> input{16, 16, 16, 16};
  std::vector<int64_t> output{1};
  std::vector<int32_t> axis{};
  ge::DataType input_dtype = ge::DT_FLOAT;

  auto opParas = op::SquareSumV1("SquareSumV1");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, input_dtype, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, input_dtype, ge::FORMAT_ND, {});
  opParas.SetAttr("axis", axis);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 65536 2048 1 ");
}
