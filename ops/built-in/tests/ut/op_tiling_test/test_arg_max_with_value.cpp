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

#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

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

TEST_F(ArgMaxWithValueTiling, ArgMaxWithValue_tiling_0) {
  using namespace optiling;
  std::string op_name = "ArgMaxWithValue";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"axis\": 1}}";

  std::vector<int64_t> input{35, 5, 128};
  std::vector<int64_t> output{35, 128};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "ND";
  tensor_input.ori_format = "ND";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "int32";
  tensor_output.format = "ND";
  tensor_output.ori_format = "ND";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234560";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "7 35 5 128 18 2 1 0 0 0 0 0 128 128 0 0 128 128 0 ");
}
TEST_F(ArgMaxWithValueTiling, ArgMaxWithValue_tiling_1) {
  using namespace optiling;
  std::string op_name = "ArgMaxWithValue";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"axis\": 1}}";

  std::vector<int64_t> input{35, 128};
  std::vector<int64_t> output{35};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "ND";
  tensor_input.ori_format = "ND";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "int32";
  tensor_output.format = "ND";
  tensor_output.ori_format = "ND";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234561";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 35 128 128 3 16 3 1 128 0 0 0 16 0 0 0 3 0 0 ");
}
TEST_F(ArgMaxWithValueTiling, ArgMaxWithValue_tiling_2) {
  using namespace optiling;
  std::string op_name = "ArgMaxWithValue";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"axis\": 1}}";

  std::vector<int64_t> input{35, 96};
  std::vector<int64_t> output{35};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "ND";
  tensor_input.ori_format = "ND";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "int32";
  tensor_output.format = "ND";
  tensor_output.ori_format = "ND";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234562";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "2 35 96 96 3 16 3 1 240 0 0 0 16 0 0 0 3 0 0 ");
}
TEST_F(ArgMaxWithValueTiling, ArgMaxWithValue_tiling_3) {
  using namespace optiling;
  std::string op_name = "ArgMaxWithValue";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"axis\": 1}}";

  std::vector<int64_t> input{35, 10000};
  std::vector<int64_t> output{35};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "ND";
  tensor_input.ori_format = "ND";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "int32";
  tensor_output.format = "ND";
  tensor_output.ori_format = "ND";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234563";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "3 35 10000 10000 3 16 3 0 0 0 10000 0 16 0 0 0 3 0 0 ");
}
TEST_F(ArgMaxWithValueTiling, ArgMaxWithValue_tiling_4) {
  using namespace optiling;
  std::string op_name = "ArgMaxWithValue";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 63488, \"core_num\": 32, \"axis\": 1}}";

  std::vector<int64_t> input{35, 10000};
  std::vector<int64_t> output{35};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float32";
  tensor_input.format = "ND";
  tensor_input.ori_format = "ND";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "int32";
  tensor_output.format = "ND";
  tensor_output.ori_format = "ND";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234564";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "4 35 10000 10000 5 8 3 0 0 1 1808 0 8 0 0 0 3 0 0 ");
}
TEST_F(ArgMaxWithValueTiling, ArgMaxWithValue_tiling_5) {
  using namespace optiling;
  std::string op_name = "ArgMaxWithValue";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 63488, \"core_num\": 32, \"axis\": 1}}";

  std::vector<int64_t> input{35, 8000};
  std::vector<int64_t> output{35};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float32";
  tensor_input.format = "ND";
  tensor_input.ori_format = "ND";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "int32";
  tensor_output.format = "ND";
  tensor_output.ori_format = "ND";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234565";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "12 35 8000 8000 5 8 3 0 0 0 8000 0 8 0 0 0 3 0 0 ");
}
