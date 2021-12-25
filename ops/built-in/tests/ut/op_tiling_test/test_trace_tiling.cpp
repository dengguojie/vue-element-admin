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
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class TraceTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TraceTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TraceTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream& tiling_data) {
  auto data = tiling_data.str();
  string result;
  uint64_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(uint64_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }

  return result;
}

TEST(TraceTiling, trace_tiling_0)
{
  using namespace optiling;
  std::string op_name = "Trace";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Trace");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32}}";
  std::vector<int64_t> input_tensor_shape{8, 9};
  std::vector<int64_t> output_tensor_shape{1};

  TeOpTensor tensor_input;
  tensor_input.shape = input_tensor_shape;
  tensor_input.dtype = "float32";
  TeOpTensor tensor_output;
  tensor_output.shape = output_tensor_shape;
  tensor_output.dtype = "float32";

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
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "8 9 1 1 ");
}

TEST(TraceTiling, trace_tiling_1)
{
  using namespace optiling;
  std::string op_name = "Trace";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Trace");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32}}";
  std::vector<int64_t> input_tensor_shape{50, 50};
  std::vector<int64_t> output_tensor_shape{1};

  TeOpTensor tensor_input;
  tensor_input.shape = input_tensor_shape;
  tensor_input.dtype = "float32";
  TeOpTensor tensor_output;
  tensor_output.shape = output_tensor_shape;
  tensor_output.dtype = "float32";

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
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "50 50 2 2 ");
}

TEST(TraceTiling, trace_tiling_2)
{
  using namespace optiling;
  std::string op_name = "Trace";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Trace");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32}}";
  std::vector<int64_t> input_tensor_shape{2048, 2048};
  std::vector<int64_t> output_tensor_shape{1};

  TeOpTensor tensor_input;
  tensor_input.shape = input_tensor_shape;
  tensor_input.dtype = "float32";
  TeOpTensor tensor_output;
  tensor_output.shape = output_tensor_shape;
  tensor_output.dtype = "float32";

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
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "2048 2048 2 32 ");
}

TEST(TraceTiling, trace_tiling_3)
{
  using namespace optiling;
  std::string op_name = "Trace";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Trace");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32}}";
  std::vector<int64_t> input_tensor_shape{2048, 2048};
  std::vector<int64_t> output_tensor_shape{1};

  TeOpParas opParas;
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234563";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}

TEST(TraceTiling, trace_tiling_4)
{
  using namespace optiling;
  std::string op_name = "Trace";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Trace");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 0}}";
  std::vector<int64_t> input_tensor_shape{2048, 2048};
  std::vector<int64_t> output_tensor_shape{1};

  TeOpTensor tensor_input;
  tensor_input.shape = input_tensor_shape;
  tensor_input.dtype = "float32";
  TeOpTensor tensor_output;
  tensor_output.shape = output_tensor_shape;
  tensor_output.dtype = "float32";

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
  ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}

TEST(TraceTiling, trace_tiling_5)
{
  using namespace optiling;
  std::string op_name = "Trace";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Trace");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 2}}";
  std::vector<int64_t> input_tensor_shape{30, 30};
  std::vector<int64_t> output_tensor_shape{1};

  TeOpTensor tensor_input;
  tensor_input.shape = input_tensor_shape;
  tensor_input.dtype = "float16";
  TeOpTensor tensor_output;
  tensor_output.shape = output_tensor_shape;
  tensor_output.dtype = "float16";

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
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "30 30 1 1 ");
}

TEST(TraceTiling, trace_tiling_6)
{
  using namespace optiling;
  std::string op_name = "Trace";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Trace");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32}}";
  std::vector<int64_t> input_tensor_shape{2048, 2048};
  std::vector<int64_t> output_tensor_shape{1};

  TeOpParas opParas;
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234566";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}