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
 * \file test_max_pool_v3_tiling.cpp
 * \brief dynamic tiling test of max_pool
 */
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class MaxPoolV3Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MaxPoolV3Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MaxPoolV3Tiling TearDown" << std::endl;
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

TEST_F(MaxPoolV3Tiling, max_pool_v3_tiling_0) {
  using namespace optiling;
  std::string op_name = "MaxPoolV3";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"ksize_h\": 3, \"ksize_w\": 3, \"strides_h\": 2, "
      "\"strides_w\": 2, \"padding\": 2, \"ceil_mode\": 0, \"pad_top\": 1, \"pad_bottom\": 1, \"pad_left\": 1, \"pad_right\": 1, \"global\": 0}}";

  std::vector<int64_t> input{1, 4, 56, 56, 16};
  std::vector<int64_t> output{1, 4, 28, 28, 16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NC1HWC0";
  tensor_input.ori_format = "NHwC";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";
  tensor_output.format = "NC1HWC0";
  tensor_output.ori_format = "NHwC";

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
  op_compile_info.key = "max_pool_v3_tiling_0";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "2 4 1 1 56 56 28 28 57 57 1 0 1 0 1 11 1 2 6 2 6 4 ");
}

TEST_F(MaxPoolV3Tiling, maxpool_tiling_1) {
  using namespace optiling;
  std::string op_name = "MaxPool";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_ele\": 130560, \"core_num\": 32, \"ksize_h\": 1, \"ksize_w\": 1, \"strides_h\": 1, "
      "\"strides_w\": 1, \"padding\": 0}}";

  std::vector<int64_t> input{16, 13, 79, 69, 16};
  std::vector<int64_t> output{16, 13, 79, 69, 16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NC1HWC0";
  tensor_input.ori_format = "NHwC";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";
  tensor_output.format = "NC1HWC0";
  tensor_output.ori_format = "NHwC";

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
  op_compile_info.key = "maxpool_tiling_1";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "0 32 35432 35416 79 69 79 69 79 69 0 0 0 0 1 1 1 4 2792 4 2776 0 ");
}
TEST_F(MaxPoolV3Tiling, max_pool_v3_tiling_2) {
  using namespace optiling;
  std::string op_name = "MaxPoolV3";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"ksize_h\": 3, \"ksize_w\": 3, \"strides_h\": 2, "
      "\"strides_w\": 2, \"padding\": 2, \"ceil_mode\": 1, \"pad_top\": 1, \"pad_bottom\": 1, \"pad_left\": 1, \"pad_right\": 1, \"global\": 0}}";

  std::vector<int64_t> input{1, 4, 56, 56, 16};
  std::vector<int64_t> output{1, 4, 29, 29, 16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NC1HWC0";
  tensor_input.ori_format = "NHwC";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";
  tensor_output.format = "NC1HWC0";
  tensor_output.ori_format = "NHwC";

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
  op_compile_info.key = "max_pool_v3_tiling_2";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "2 4 1 1 56 56 29 29 59 59 1 2 1 2 1 10 1 2 9 2 9 4 ");
}
TEST_F(MaxPoolV3Tiling, max_pool_v3_tiling_global) {
  using namespace optiling;
  std::string op_name = "MaxPoolV3";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"ksize_h\": 1, \"ksize_w\": 3, \"strides_h\": 1, "
      "\"strides_w\": 1, \"padding\": 1, \"ceil_mode\": 0, \"pad_top\": 0, \"pad_bottom\": 0, \"pad_left\": 0, \"pad_right\": 0, \"global\": 1}}";

  std::vector<int64_t> input{32, 1, 3, 3, 16};
  std::vector<int64_t> output{32, 1, 1, 1, 16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NC1HWC0";
  tensor_input.ori_format = "NHWC";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";
  tensor_output.format = "NC1HWC0";
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
  op_compile_info.key = "max_pool_v3_tiling_global";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "6 32 1 1 3 3 1 1 3 3 0 0 0 0 1 1 1 0 0 0 0 32 ");
}