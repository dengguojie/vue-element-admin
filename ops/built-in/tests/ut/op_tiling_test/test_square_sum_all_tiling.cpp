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

#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class SquareSumAllTiling : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "SquareSumAllTiling SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "SquareSumAllTiling TearDown" << std::endl; }
};

static string to_string(const std::stringstream &tiling_data) {
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

TEST_F(SquareSumAllTiling, SquareSumAllTiling_tiling_1) {
  using namespace optiling;
  std::string op_name = "SquareSumAll";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 262080, \"core_num\": 32, \"data_each_block\": 8, \"dtype_bytes_size\": 4}}";

  std::vector<int64_t> inputX{16, 16, 64, 32};
  std::vector<int64_t> inputY{16, 16, 64, 32};
  std::vector<int64_t> outputX{1};
  std::vector<int64_t> outputY{1};

  TeOpTensor tensor_input_x;
  tensor_input_x.shape = inputX;
  tensor_input_x.dtype = "float32";
  tensor_input_x.format = "ND";
  TeOpTensor tensor_input_y;
  tensor_input_y.shape = inputY;
  tensor_input_y.dtype = "float32";
  tensor_input_y.format = "ND";
  TeOpTensor tensor_output_x;
  tensor_output_x.shape = outputX;
  tensor_output_x.dtype = "float32";
  tensor_output_x.format = "ND";
  TeOpTensor tensor_output_y;
  tensor_output_y.shape = outputY;
  tensor_output_y.dtype = "float32";
  tensor_output_y.format = "ND";

  TeOpTensorArg tensor_input_argX;
  tensor_input_argX.tensor.push_back(tensor_input_x);
  tensor_input_argX.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_input_argY;
  tensor_input_argY.tensor.push_back(tensor_input_y);
  tensor_input_argY.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_argX;
  tensor_output_argX.tensor.push_back(tensor_output_x);
  tensor_output_argX.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_argY;
  tensor_output_argY.tensor.push_back(tensor_output_y);
  tensor_output_argY.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_argX);
  opParas.inputs.push_back(tensor_input_argY);
  opParas.outputs.push_back(tensor_output_argX);
  opParas.outputs.push_back(tensor_output_argY);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "SquareSumAllTiling_1";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "32 16384 1 1 16384 16384 0 0 8 0 8 0 2048 0 2048 0 ");
}

TEST_F(SquareSumAllTiling, SquareSumAllTiling_tiling_2) {
  using namespace optiling;
  std::string op_name = "SquareSumAll";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 262080, \"core_num\": 32, \"data_each_block\": 8, \"dtype_bytes_size\": 4}}";

  std::vector<int64_t> inputX{9973, 13, 829};
  std::vector<int64_t> inputY{9973, 13, 829};
  std::vector<int64_t> outputX{1};
  std::vector<int64_t> outputY{1};

  TeOpTensor tensor_input_x;
  tensor_input_x.shape = inputX;
  tensor_input_x.dtype = "float32";
  tensor_input_x.format = "ND";
  TeOpTensor tensor_input_y;
  tensor_input_y.shape = inputY;
  tensor_input_y.dtype = "float32";
  tensor_input_y.format = "ND";
  TeOpTensor tensor_output_x;
  tensor_output_x.shape = outputX;
  tensor_output_x.dtype = "float32";
  tensor_output_x.format = "ND";
  TeOpTensor tensor_output_y;
  tensor_output_y.shape = outputY;
  tensor_output_y.dtype = "float32";
  tensor_output_y.format = "ND";

  TeOpTensorArg tensor_input_argX;
  tensor_input_argX.tensor.push_back(tensor_input_x);
  tensor_input_argX.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_input_argY;
  tensor_input_argY.tensor.push_back(tensor_input_y);
  tensor_input_argY.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_argX;
  tensor_output_argX.tensor.push_back(tensor_output_x);
  tensor_output_argX.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_argY;
  tensor_output_argY.tensor.push_back(tensor_output_y);
  tensor_output_argY.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_argX);
  opParas.inputs.push_back(tensor_input_argY);
  opParas.outputs.push_back(tensor_output_argX);
  opParas.outputs.push_back(tensor_output_argY);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "SquareSumAllTiling_2";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "32 3358719 102 102 32760 32760 17199 17212 9 8 9 8 4095 2150 4095 2152 ");
}

TEST_F(SquareSumAllTiling, SquareSumAllTiling_tiling_3) {
  using namespace optiling;
  std::string op_name = "SquareSumAll";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 262080, \"core_num\": 8, \"data_each_block\": 8, \"dtype_bytes_size\": 4}}";

  std::vector<int64_t> inputX{108, 154};
  std::vector<int64_t> inputY{108, 1, 154};
  std::vector<int64_t> outputX{1};
  std::vector<int64_t> outputY{1};

  TeOpTensor tensor_input_x;
  tensor_input_x.shape = inputX;
  tensor_input_x.dtype = "float32";
  tensor_input_x.format = "ND";
  TeOpTensor tensor_input_y;
  tensor_input_y.shape = inputY;
  tensor_input_y.dtype = "float32";
  tensor_input_y.format = "ND";
  TeOpTensor tensor_output_x;
  tensor_output_x.shape = outputX;
  tensor_output_x.dtype = "float32";
  tensor_output_x.format = "ND";
  TeOpTensor tensor_output_y;
  tensor_output_y.shape = outputY;
  tensor_output_y.dtype = "float32";
  tensor_output_y.format = "ND";

  TeOpTensorArg tensor_input_argX;
  tensor_input_argX.tensor.push_back(tensor_input_x);
  tensor_input_argX.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_input_argY;
  tensor_input_argY.tensor.push_back(tensor_input_y);
  tensor_input_argY.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_argX;
  tensor_output_argX.tensor.push_back(tensor_output_x);
  tensor_output_argX.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_argY;
  tensor_output_argY.tensor.push_back(tensor_output_y);
  tensor_output_argY.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_argX);
  opParas.inputs.push_back(tensor_input_argY);
  opParas.outputs.push_back(tensor_output_argX);
  opParas.outputs.push_back(tensor_output_argY);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "SquareSumAllTiling_3";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}
