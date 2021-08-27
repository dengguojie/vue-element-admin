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
 * \file test_depth_and_space_tiling.cpp
 * \brief dynamic shape tiling test of depth and space ops
 */
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class DepthAndSpaceTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DepthAndSpaceTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DepthAndSpaceTiling TearDown" << std::endl;
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

TEST_F(DepthAndSpaceTiling, depthandspace_tiling_0) {
  using namespace optiling;
  std::string op_name = "DepthToSpace";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 8192, \"core_num\": 32, \"dtype\": \"float16\", \"block_size\": 2, \"mode\": \"DCR\"}}";

  std::vector<int64_t> input{16, 2, 2, 16};
  std::vector<int64_t> output{16, 4, 4, 4};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NHWC";
  tensor_input.ori_format = "NHWC";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";
  tensor_output.format = "NHWC";
  tensor_output.ori_format = "NHWC";

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
}

TEST_F(DepthAndSpaceTiling, depthandspace_tiling_1) {
  using namespace optiling;
  std::string op_name = "SpaceToDepth";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 8192, \"core_num\": 32, \"dtype\": \"float16\", \"block_size\": 2, \"mode\": \"DCR\"}}";

  std::vector<int64_t> input{16, 4, 4, 4};
  std::vector<int64_t> output{16, 2, 2, 16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NHWC";
  tensor_input.ori_format = "NHWC";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";
  tensor_output.format = "NHWC";
  tensor_output.ori_format = "NHWC";

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
}

TEST_F(DepthAndSpaceTiling, depthandspace_tiling_2) {
  using namespace optiling;
  std::string op_name = "DepthToSpace";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 8192, \"core_num\": 32, \"dtype\": \"float16\", \"block_size\": 2, \"mode\": \"CRD\"}}";

  std::vector<int64_t> input{1, 8, 2, 3};
  std::vector<int64_t> output{1, 2, 4, 6};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NCHW";
  tensor_input.ori_format = "NCHW";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";
  tensor_output.format = "NCHW";
  tensor_output.ori_format = "NCHW";

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
  op_compile_info.key = "depthandspace_tiling_2";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(DepthAndSpaceTiling, depthandspace_tiling_3) {
  using namespace optiling;
  std::string op_name = "DepthToSpace";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 8192, \"core_num\": 32, \"dtype\": \"float16\", \"block_size\": 2, \"mode\": \"DCR\"}}";

  std::vector<int64_t> input{1, 8, 2, 3};
  std::vector<int64_t> output{1, 2, 4, 6};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NCHW";
  tensor_input.ori_format = "NCHW";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";
  tensor_output.format = "NCHW";
  tensor_output.ori_format = "NCHW";

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
  op_compile_info.key = "depthandspace_tiling_3";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(DepthAndSpaceTiling, depthandspace_tiling_4) {
  using namespace optiling;
  std::string op_name = "DepthToSpace";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 8192, \"core_num\": 32, \"dtype\": \"float16\", \"block_size\": 2, \"mode\": \"CRD\"}}";

  std::vector<int64_t> input{1, 2, 3, 8};
  std::vector<int64_t> output{1, 4, 6, 2};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NHWC";
  tensor_input.ori_format = "NHWC";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";
  tensor_output.format = "NHWC";
  tensor_output.ori_format = "NHWC";

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
  op_compile_info.key = "depthandspace_tiling_4";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}