/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file test_arg_max_v2.cpp
 * \brief dynamic shape tiling test of arg_max_v2
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
#define private public
#include "register/op_tiling_registry.h"

using namespace std;
using namespace ge;

class ArgMaxV2Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ArgMaxV2Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ArgMaxV2Tiling TearDown" << std::endl;
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

TEST_F(ArgMaxV2Tiling, ArgMaxV2_tiling_0) {
  std::string op_name = "ArgMaxV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32}}";

  std::vector<int64_t> input{35, 5, 128};
  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT16);
  auto data = op::Data("data");
  data.update_input_desc_x(tensor_input);
  data.update_output_desc_y(tensor_input);
  auto input_axis = ge::Shape({1});
  TensorDesc tensor_input_axis(input_axis, FORMAT_ND, DT_INT32);
  Tensor multiples_tensor(tensor_input_axis);
  uint32_t multiples_tensor_value[2] = {1};
  multiples_tensor.SetData((uint8_t *)multiples_tensor_value, sizeof(uint32_t));
  
  auto argmax_multiples = op::Constant("dimension").set_attr_value(multiples_tensor);
  auto opParas = op::ArgMaxV2("ArgMaxV2");
  opParas.set_input_x(data);
  opParas.set_input_dimension(argmax_multiples);
  vector<Operator> inputs{data, argmax_multiples};
  vector<Operator> outputs{opParas};
  opParas.UpdateInputDesc("dimension", tensor_input_axis);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "7 35 5 128 18 2 1 0 0 0 0 0 128 128 0 0 128 128 0 ");
}

TEST_F(ArgMaxV2Tiling, ArgMaxV2_tiling_1) {
  std::string op_name = "ArgMaxV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32}}";

  std::vector<int64_t> input{35, 128};
  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT16);
  auto data = op::Data("data");
  data.update_input_desc_x(tensor_input);
  data.update_output_desc_y(tensor_input);
  auto input_axis = ge::Shape({1});
  TensorDesc tensor_input_axis(input_axis, FORMAT_ND, DT_INT32);
  Tensor multiples_tensor(tensor_input_axis);
  uint32_t multiples_tensor_value[2] = {1};
  multiples_tensor.SetData((uint8_t *)multiples_tensor_value, sizeof(uint32_t));
  
  auto argmax_multiples = op::Constant("dimension").set_attr_value(multiples_tensor);
  auto opParas = op::ArgMaxV2("ArgMaxV2");
  opParas.set_input_x(data);
  opParas.set_input_dimension(argmax_multiples);
  vector<Operator> inputs{data, argmax_multiples};
  vector<Operator> outputs{opParas};
  opParas.UpdateInputDesc("dimension", tensor_input_axis);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 35 128 128 5 8 3 1 192 0 0 0 8 0 0 0 3 0 0 ");
}

TEST_F(ArgMaxV2Tiling, ArgMaxV2_tiling_2) {
  std::string op_name = "ArgMaxV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32}}";

  std::vector<int64_t> input{35, 96};
  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT16);
  auto data = op::Data("data");
  data.update_input_desc_x(tensor_input);
  data.update_output_desc_y(tensor_input);
  auto input_axis = ge::Shape({1});
  TensorDesc tensor_input_axis(input_axis, FORMAT_ND, DT_INT32);
  Tensor multiples_tensor(tensor_input_axis);
  uint32_t multiples_tensor_value[2] = {1};
  multiples_tensor.SetData((uint8_t *)multiples_tensor_value, sizeof(uint32_t));
  
  auto argmax_multiples = op::Constant("dimension").set_attr_value(multiples_tensor);
  auto opParas = op::ArgMaxV2("ArgMaxV2");
  opParas.set_input_x(data);
  opParas.set_input_dimension(argmax_multiples);
  vector<Operator> inputs{data, argmax_multiples};
  vector<Operator> outputs{opParas};
  opParas.UpdateInputDesc("dimension", tensor_input_axis);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 35 96 96 5 8 3 1 248 0 0 0 8 0 0 0 3 0 0 ");
}

TEST_F(ArgMaxV2Tiling, ArgMaxV2_tiling_3) {
  std::string op_name = "ArgMaxV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32}}";

  std::vector<int64_t> input{35, 10000};
  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT16);
  auto data = op::Data("data");
  data.update_input_desc_x(tensor_input);
  data.update_output_desc_y(tensor_input);
  auto input_axis = ge::Shape({1});
  TensorDesc tensor_input_axis(input_axis, FORMAT_ND, DT_INT32);
  Tensor multiples_tensor(tensor_input_axis);
  uint32_t multiples_tensor_value[2] = {1};
  multiples_tensor.SetData((uint8_t *)multiples_tensor_value, sizeof(uint32_t));
  
  auto argmax_multiples = op::Constant("dimension").set_attr_value(multiples_tensor);
  auto opParas = op::ArgMaxV2("ArgMaxV2");
  opParas.set_input_x(data);
  opParas.set_input_dimension(argmax_multiples);
  vector<Operator> inputs{data, argmax_multiples};
  vector<Operator> outputs{opParas};
  opParas.UpdateInputDesc("dimension", tensor_input_axis);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "3 35 10000 10000 5 8 3 0 0 0 10000 0 8 0 0 0 3 0 0 ");
}

TEST_F(ArgMaxV2Tiling, ArgMaxV2_tiling_4) {
  std::string op_name = "ArgMaxV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 63488, \"core_num\": 32}}";

  std::vector<int64_t> input{35, 10000};
  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  auto data = op::Data("data");
  data.update_input_desc_x(tensor_input);
  data.update_output_desc_y(tensor_input);
  auto input_axis = ge::Shape({1});
  TensorDesc tensor_input_axis(input_axis, FORMAT_ND, DT_INT32);
  Tensor multiples_tensor(tensor_input_axis);
  uint32_t multiples_tensor_value[2] = {1};
  multiples_tensor.SetData((uint8_t *)multiples_tensor_value, sizeof(uint32_t));
  
  auto argmax_multiples = op::Constant("dimension").set_attr_value(multiples_tensor);
  auto opParas = op::ArgMaxV2("ArgMaxV2");
  opParas.set_input_x(data);
  opParas.set_input_dimension(argmax_multiples);
  vector<Operator> inputs{data, argmax_multiples};
  vector<Operator> outputs{opParas};
  opParas.UpdateInputDesc("dimension", tensor_input_axis);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4 35 10000 10000 5 8 3 0 0 1 1808 0 8 0 0 0 3 0 0 ");
}

TEST_F(ArgMaxV2Tiling, ArgMaxV2_tiling_5) {
  std::string op_name = "ArgMaxV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 63488, \"core_num\": 32}}";

  std::vector<int64_t> input{35, 8000};
  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  auto data = op::Data("data");
  data.update_input_desc_x(tensor_input);
  data.update_output_desc_y(tensor_input);
  auto input_axis = ge::Shape({1});
  TensorDesc tensor_input_axis(input_axis, FORMAT_ND, DT_INT64);
  Tensor multiples_tensor(tensor_input_axis);
  int64_t multiples_tensor_value[2] = {1};
  multiples_tensor.SetData((uint8_t *)multiples_tensor_value, sizeof(int64_t));
  
  auto argmax_multiples = op::Constant("dimension").set_attr_value(multiples_tensor);
  auto opParas = op::ArgMaxV2("ArgMaxV2");
  opParas.set_input_x(data);
  opParas.set_input_dimension(argmax_multiples);
  vector<Operator> inputs{data, argmax_multiples};
  vector<Operator> outputs{opParas};
  opParas.UpdateInputDesc("dimension", tensor_input_axis);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "12 35 8000 8000 5 8 3 0 0 0 8000 0 8 0 0 0 3 0 0 ");
}
