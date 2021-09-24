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

#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "selection_ops.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"
#include "register/op_tiling_registry.h"

using namespace std;
using namespace ge;

class ArgMaxWithValueTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ArgMaxWithValueTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ArgMaxWithValueTiling TearDown" << std::endl;
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

using namespace ge;
#include "test_common.h"
/*
.INPUT(x, TensorType({DT_FLOAT,DT_FLOAT16}))
    .OUTPUT(indice,TensorType({DT_INT32}))
    .OUTPUT(values, TensorType({DT_FLOAT,DT_FLOAT16}))
    .REQUIRED_ATTR(dimension, Int)
    .ATTR(keep_dims, Bool, false)
*/

TEST_F(ArgMaxWithValueTiling, ArgMaxWithValue_tiling_0) {
  std::string op_name = "ArgMaxWithValue";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"axis\": 1}}";

  std::vector<int64_t> input = {35, 5, 128};

  TensorDesc tensor_input(ge::Shape(input), ge::FORMAT_ND, ge::DT_FLOAT16);

  auto opParas = op::ArgMaxWithValue("ArgMaxWithValue");
  TENSOR_INPUT(opParas, tensor_input, x);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "7 35 5 128 18 2 1 0 0 0 0 0 128 128 0 0 128 128 0 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second(opParas, op_compile_info, runInfo);
  }
}

TEST_F(ArgMaxWithValueTiling, ArgMaxWithValue_tiling_1) {
  std::string op_name = "ArgMaxWithValue";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"axis\": 1}}";

  std::vector<int64_t> input{35, 128};

  TensorDesc tensor_input(ge::Shape(input), ge::FORMAT_ND, ge::DT_FLOAT16);

  auto opParas = op::ArgMaxWithValue("ArgMaxWithValue");
  TENSOR_INPUT(opParas, tensor_input, x);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 35 128 128 3 16 3 1 128 0 0 0 16 0 0 0 3 0 0 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second(opParas, op_compile_info, runInfo);
  }
}

TEST_F(ArgMaxWithValueTiling, ArgMaxWithValue_tiling_2) {
  std::string op_name = "ArgMaxWithValue";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"axis\": 1}}";

  std::vector<int64_t> input{35, 96};

  TensorDesc tensor_input(ge::Shape(input), ge::FORMAT_ND, ge::DT_FLOAT16);

  auto opParas = op::ArgMaxWithValue("ArgMaxWithValue");
  TENSOR_INPUT(opParas, tensor_input, x);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 35 96 96 3 16 3 1 240 0 0 0 16 0 0 0 3 0 0 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second(opParas, op_compile_info, runInfo);
  }
}

TEST_F(ArgMaxWithValueTiling, ArgMaxWithValue_tiling_3) {
  std::string op_name = "ArgMaxWithValue";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"axis\": 1}}";

  std::vector<int64_t> input{35, 10000};

  TensorDesc tensor_input(ge::Shape(input), ge::FORMAT_ND, ge::DT_FLOAT16);

  auto opParas = op::ArgMaxWithValue("ArgMaxWithValue");
  TENSOR_INPUT(opParas, tensor_input, x);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "3 35 10000 10000 3 16 3 0 0 0 10000 0 16 0 0 0 3 0 0 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second(opParas, op_compile_info, runInfo);
  }
}

TEST_F(ArgMaxWithValueTiling, ArgMaxWithValue_tiling_4) {
  std::string op_name = "ArgMaxWithValue";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 63488, \"core_num\": 32, \"axis\": 1}}";

  std::vector<int64_t> input{35, 10000};

  TensorDesc tensor_input(ge::Shape(input), ge::FORMAT_ND, ge::DT_FLOAT);

  auto opParas = op::ArgMaxWithValue("ArgMaxWithValue");
  TENSOR_INPUT(opParas, tensor_input, x);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4 35 10000 10000 5 8 3 0 0 1 1808 0 8 0 0 0 3 0 0 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second(opParas, op_compile_info, runInfo);
  }
}

TEST_F(ArgMaxWithValueTiling, ArgMaxWithValue_tiling_5) {
  std::string op_name = "ArgMaxWithValue";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 63488, \"core_num\": 32, \"axis\": 1}}";

  std::vector<int64_t> input{35, 8000};

  TensorDesc tensor_input(ge::Shape(input), ge::FORMAT_ND, ge::DT_FLOAT);

  auto opParas = op::ArgMaxWithValue("ArgMaxWithValue");
  TENSOR_INPUT(opParas, tensor_input, x);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "12 35 8000 8000 5 8 3 0 0 0 8000 0 8 0 0 0 3 0 0 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second(opParas, op_compile_info, runInfo);
  }
}
