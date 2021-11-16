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
 * \file test_maxpoolgrad_tiling.cpp
 * \brief dynamic tiling test of max_pool_grad
 */
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class MaxPoolGradTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MaxPoolGradTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MaxPoolGradTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream &tiling_data) {
  auto data = tiling_data.str();
  string result;
  int64_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }
    std::cout << "MaxPoolGradTiling tiling data is" << std::endl;
    std::cout << result << std::endl;
  return result;
}

TEST_F(MaxPoolGradTiling, maxpoolgrad_tiling_0) {
  using namespace optiling;
  std::string op_name = "MaxPoolGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo ="{\"vars\": {\"ub_size\": 131072, \"l1_size\": 1048576, \"core_num\": 32, \"kh\": 3, \"kw\":3, \"sh\": 2, \"sw\": 2, \"padding\":0}}";


  std::vector<int64_t> input{33,2,300,300,16};
  std::vector<int64_t> output{33,2,149,149,16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NC1HWC0";
  tensor_input.ori_format = "NHWC";
  
  TeOpTensor tensor_input1;
  tensor_input1.shape = output;
  tensor_input1.dtype = "float16";
  tensor_input1.format = "NC1HWC0";
  tensor_input1.ori_format = "NHWC";
  
  TeOpTensor tensor_input2;
  tensor_input2.shape = output;
  tensor_input2.dtype = "float16";
  tensor_input2.format = "NC1HWC0";
  tensor_input2.ori_format = "NHWC";
  
  
  TeOpTensor tensor_output;
  tensor_output.shape = input;
  tensor_output.dtype = "float16";
  tensor_output.format = "NC1HWC0";
  tensor_output.ori_format = "NHWC";

  TeOpTensorArg tensor_input_arg;
  TeOpTensorArg tensor_input_arg1;
  TeOpTensorArg tensor_input_arg2;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  tensor_input_arg1.tensor.push_back(tensor_input1);
  tensor_input_arg1.arg_type = TA_SINGLE;
  tensor_input_arg2.tensor.push_back(tensor_input2);
  tensor_input_arg2.arg_type = TA_SINGLE;
  
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_input_arg1);
  opParas.inputs.push_back(tensor_input_arg2);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123";
  OpRunInfo runInfo;
  std::cout << "maxpoolgrad_tiling_0" << std::endl;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 33 2 300 300 149 149 0 0 0 0 1 1 1 1 66 32 149 149 300 300 1 149 2 2 2 3 3 300 300 0 149 0 0 14400 0 225 2880000 710432 0 0 900 179100 149 44253 10 19 16 2 0 38 0 0 18 40 600 358800 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1200 0 0 0 0 0 0 0 0 0 298 299 1200 9600 4800 600 1200 1200 0 9600 0 150 1800 0 358200 2400 -600 357600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}
TEST_F(MaxPoolGradTiling, maxpoolgrad_tiling_1) {
  using namespace optiling;
  std::string op_name = "MaxPoolGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo ="{\"vars\": {\"ub_size\": 131072, \"l1_size\": 1048576, \"core_num\": 32, \"kh\": 34, \"kw\":3, \"sh\": 4, \"sw\": 4, \"padding\":0}}";


  std::vector<int64_t> input{3,2,60,4,16};
  std::vector<int64_t> output{3,2,15,1,16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NC1HWC0";
  tensor_input.ori_format = "NHWC";
  
  TeOpTensor tensor_input1;
  tensor_input1.shape = output;
  tensor_input1.dtype = "float16";
  tensor_input1.format = "NC1HWC0";
  tensor_input1.ori_format = "NHWC";
  
  TeOpTensor tensor_input2;
  tensor_input2.shape = output;
  tensor_input2.dtype = "float16";
  tensor_input2.format = "NC1HWC0";
  tensor_input2.ori_format = "NHWC";

  TeOpTensor tensor_output;
  tensor_output.shape = input;
  tensor_output.dtype = "float16";
  tensor_output.format = "NC1HWC0";
  tensor_output.ori_format = "NHWC";

  TeOpTensorArg tensor_input_arg;
  TeOpTensorArg tensor_input_arg1;
  TeOpTensorArg tensor_input_arg2;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  tensor_input_arg1.tensor.push_back(tensor_input1);
  tensor_input_arg1.arg_type = TA_SINGLE;
  tensor_input_arg2.tensor.push_back(tensor_input2);
  tensor_input_arg2.arg_type = TA_SINGLE;
  
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_input_arg1);
  opParas.inputs.push_back(tensor_input_arg2);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234";
  OpRunInfo runInfo;
  std::cout << "maxpoolgrad_tiling_1" << std::endl;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "3 3 2 60 4 15 1 0 0 0 0 30 -1 -30 0 90 32 1 1 34 4 1 1 26 26 2 3 34 4 4 0 1 0 0 2176 0 34 7680 480 34 0 136 344 1 29 1 1 8 1 0 2 0 0 0 8 0 688 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 15 0 0 0 1 32 1 1 8 1 0 0 0 30 0 2 0 272 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}
TEST_F(MaxPoolGradTiling, maxpoolgrad_tiling_2) {
  using namespace optiling;
  std::string op_name = "MaxPoolGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo ="{\"vars\": {\"ub_size\": 131072, \"l1_size\": 1048576, \"core_num\": 32, \"kh\": 7, \"kw\":1, \"sh\": 6, \"sw\": 2, \"padding\":0}}";


  std::vector<int64_t> input{10,4,11,1,16};
  std::vector<int64_t> output{10,42,1,1,16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NC1HWC0";
  tensor_input.ori_format = "NHWC";
  
  TeOpTensor tensor_input1;
  tensor_input1.shape = output;
  tensor_input1.dtype = "float16";
  tensor_input1.format = "NC1HWC0";
  tensor_input1.ori_format = "NHWC";
  
  TeOpTensor tensor_input2;
  tensor_input2.shape = output;
  tensor_input2.dtype = "float16";
  tensor_input2.format = "NC1HWC0";
  tensor_input2.ori_format = "NHWC";

  TeOpTensor tensor_output;
  tensor_output.shape = input;
  tensor_output.dtype = "float16";
  tensor_output.format = "NC1HWC0";
  tensor_output.ori_format = "NHWC";

  TeOpTensorArg tensor_input_arg;
  TeOpTensorArg tensor_input_arg1;
  TeOpTensorArg tensor_input_arg2;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  tensor_input_arg1.tensor.push_back(tensor_input1);
  tensor_input_arg1.arg_type = TA_SINGLE;
  tensor_input_arg2.tensor.push_back(tensor_input2);
  tensor_input_arg2.arg_type = TA_SINGLE;
  
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_input_arg1);
  opParas.inputs.push_back(tensor_input_arg2);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "12345";
  OpRunInfo runInfo;
  std::cout << "maxpoolgrad_tiling_2" << std::endl;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "0 10 4 11 1 1 1 0 0 0 0 1 0 4 -1 40 32 1 1 11 1 1 1 8 8 1 2 11 1 0 0 0 0 0 176 48 2 704 672 0 0 11 33 1 3 1 1 8 1 0 2 0 0 0 8 0 66 0 0 0 0 22 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 12 1 1 8 1 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}
TEST_F(MaxPoolGradTiling, maxpoolgrad_tiling_3) {
  using namespace optiling;
  std::string op_name = "MaxPoolGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo ="{\"vars\": {\"ub_size\": 131072, \"l1_size\": 1048576, \"core_num\": 32, \"kh\": 3, \"kw\":3, \"sh\": 4, \"sw\": 4, \"padding\":0}}";


  std::vector<int64_t> input{9,4,300,10,16};
  std::vector<int64_t> output{9,4,75,10,16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NC1HWC0";
  tensor_input.ori_format = "NHWC";
  
  TeOpTensor tensor_input1;
  tensor_input1.shape = output;
  tensor_input1.dtype = "float16";
  tensor_input1.format = "NC1HWC0";
  tensor_input1.ori_format = "NHWC";
  
  TeOpTensor tensor_input2;
  tensor_input2.shape = output;
  tensor_input2.dtype = "float16";
  tensor_input2.format = "NC1HWC0";
  tensor_input2.ori_format = "NHWC";

  TeOpTensor tensor_output;
  tensor_output.shape = input;
  tensor_output.dtype = "float16";
  tensor_output.format = "NC1HWC0";
  tensor_output.ori_format = "NHWC";

  TeOpTensorArg tensor_input_arg;
  TeOpTensorArg tensor_input_arg1;
  TeOpTensorArg tensor_input_arg2;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  tensor_input_arg1.tensor.push_back(tensor_input1);
  tensor_input_arg1.arg_type = TA_SINGLE;
  tensor_input_arg2.tensor.push_back(tensor_input2);
  tensor_input_arg2.arg_type = TA_SINGLE;
  
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_input_arg1);
  opParas.inputs.push_back(tensor_input_arg2);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234560";
  OpRunInfo runInfo;
  std::cout << "maxpoolgrad_tiling_3" << std::endl;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 9 4 300 10 75 10 0 0 0 0 -1 -1 0 -30 36 32 75 10 300 10 9 10 4 4 1 2 36 10 10 0 8 0 0 5760 0 90 192000 48000 35 0 360 11640 90 2910 6 12 8 1 0 24 0 0 1 16 -20 23260 0 0 0 0 120 11880 0 0 0 30 2970 2 4 0 0 0 0 23740 0 0 0 0 1 80 1 2 16 9 1 0 3 12 240 20 300 740 5920 -160 -20 740 740 0 5920 32 92 720 0 23280 720 0 23280 11 350 11650 10 110 11890 250 1 3 0 260 2080 260 260 0 2080 32 32 240 480 23760 240 480 23760 0 0 ");
}
TEST_F(MaxPoolGradTiling, maxpoolgrad_tiling_4) {
  using namespace optiling;
  std::string op_name = "MaxPoolGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo ="{\"vars\": {\"ub_size\": 131072, \"l1_size\": 1048576, \"core_num\": 32, \"kh\": 3, \"kw\":3, \"sh\": 4, \"sw\": 4, \"padding\":0}}";


  std::vector<int64_t> input{2,4,300,300,16};
  std::vector<int64_t> output{2,4,75,75,16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NC1HWC0";
  tensor_input.ori_format = "NHWC";
  
  TeOpTensor tensor_input1;
  tensor_input1.shape = output;
  tensor_input1.dtype = "float16";
  tensor_input1.format = "NC1HWC0";
  tensor_input1.ori_format = "NHWC";
  
  TeOpTensor tensor_input2;
  tensor_input2.shape = output;
  tensor_input2.dtype = "float16";
  tensor_input2.format = "NC1HWC0";
  tensor_input2.ori_format = "NHWC";

  TeOpTensor tensor_output;
  tensor_output.shape = input;
  tensor_output.dtype = "float16";
  tensor_output.format = "NC1HWC0";
  tensor_output.ori_format = "NHWC";

  TeOpTensorArg tensor_input_arg;
  TeOpTensorArg tensor_input_arg1;
  TeOpTensorArg tensor_input_arg2;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  tensor_input_arg1.tensor.push_back(tensor_input1);
  tensor_input_arg1.arg_type = TA_SINGLE;
  tensor_input_arg2.tensor.push_back(tensor_input2);
  tensor_input_arg2.arg_type = TA_SINGLE;
  
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_input_arg1);
  opParas.inputs.push_back(tensor_input_arg2);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  std::cout << "maxpoolgrad_tiling_4" << std::endl;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "3 2 4 300 300 75 75 0 0 0 0 -1 -1 0 0 40 32 15 75 60 300 1 75 8 8 1 2 4 300 300 0 15 0 1 2880 0 45 5760000 360000 3 0 900 359100 75 22425 5 10 8 1 0 20 0 0 9 24 600 718200 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 0 0 0 0 2400 0 0 0 0 0 0 0 0 0 150 0 1800 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 300 ");
}
TEST_F(MaxPoolGradTiling, maxpoolgrad_tiling_5) {
  using namespace optiling;
  std::string op_name = "MaxPoolGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo ="{\"vars\": {\"ub_size\": 131072, \"l1_size\": 1048576, \"core_num\": 32, \"kh\": 3, \"kw\":3, \"sh\": 2, \"padding\": 4, \"sw\": 2}}";


  std::vector<int64_t> input{33,2,300,300,16};
  std::vector<int64_t> output{33,2,149,149,16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NC1HWC0";
  tensor_input.ori_format = "NHWC";
  
  TeOpTensor tensor_input1;
  tensor_input1.shape = output;
  tensor_input1.dtype = "float16";
  tensor_input1.format = "NC1HWC0";
  tensor_input1.ori_format = "NHWC";
  
  TeOpTensor tensor_input2;
  tensor_input2.shape = output;
  tensor_input2.dtype = "float16";
  tensor_input2.format = "NC1HWC0";
  tensor_input2.ori_format = "NHWC";
  
  
  TeOpTensor tensor_output;
  tensor_output.shape = input;
  tensor_output.dtype = "float16";
  tensor_output.format = "NC1HWC0";
  tensor_output.ori_format = "NHWC";

  TeOpTensorArg tensor_input_arg;
  TeOpTensorArg tensor_input_arg1;
  TeOpTensorArg tensor_input_arg2;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  tensor_input_arg1.tensor.push_back(tensor_input1);
  tensor_input_arg1.arg_type = TA_SINGLE;
  tensor_input_arg2.tensor.push_back(tensor_input2);
  tensor_input_arg2.arg_type = TA_SINGLE;
  
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_input_arg1);
  opParas.inputs.push_back(tensor_input_arg2);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234567";
  OpRunInfo runInfo;
  std::cout << "maxpoolgrad_tiling_5" << std::endl;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "6 33 2 300 300 149 149 0 0 0 0 1 1 1 1 66 32 149 149 300 300 1 149 2 2 2 3 3 300 300 0 149 0 0 14400 0 225 2880000 710432 3 0 0 0 149 44253 10 19 16 2 0 38 0 0 18 40 0 0 0 0 0 0 0 0 0 0 1440000 0 0 0 0 0 0 0 0 0 1 0 3 300 0 1200 0 0 0 0 0 0 0 1 0 298 0 600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4800 0 ");
}
TEST_F(MaxPoolGradTiling, maxpoolgrad_tiling_6) {
  using namespace optiling;
  std::string op_name = "MaxPoolGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo ="{\"vars\": {\"ub_size\": 131072, \"l1_size\": 1048576, \"core_num\": 32, \"kh\": 2, \"kw\":4, \"sh\": 3, \"padding\": 1, \"sw\": 2}}";


  std::vector<int64_t> input{15,2,306,24,16};
  std::vector<int64_t> output{15,2,102,12,16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NC1HWC0";
  tensor_input.ori_format = "NHWC";
  
  TeOpTensor tensor_input1;
  tensor_input1.shape = output;
  tensor_input1.dtype = "float16";
  tensor_input1.format = "NC1HWC0";
  tensor_input1.ori_format = "NHWC";
  
  TeOpTensor tensor_input2;
  tensor_input2.shape = output;
  tensor_input2.dtype = "float16";
  tensor_input2.format = "NC1HWC0";
  tensor_input2.ori_format = "NHWC";
  
  
  TeOpTensor tensor_output;
  tensor_output.shape = input;
  tensor_output.dtype = "float16";
  tensor_output.format = "NC1HWC0";
  tensor_output.ori_format = "NHWC";

  TeOpTensorArg tensor_input_arg;
  TeOpTensorArg tensor_input_arg1;
  TeOpTensorArg tensor_input_arg2;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  tensor_input_arg1.tensor.push_back(tensor_input1);
  tensor_input_arg1.arg_type = TA_SINGLE;
  tensor_input_arg2.tensor.push_back(tensor_input2);
  tensor_input_arg2.arg_type = TA_SINGLE;
  
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_input_arg1);
  opParas.inputs.push_back(tensor_input_arg2);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1238";
  OpRunInfo runInfo;
  std::cout << "maxpoolgrad_tiling_6" << std::endl;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "6 15 2 306 24 102 12 0 0 1 1 -1 2 0 -2 60 32 51 12 153 24 10 12 28 28 1 2 30 24 24 0 5 0 0 12480 0 195 235008 39168 29 0 0 0 120 2328 8 15 8 1 0 30 0 0 1 32 4 0 0 0 0 0 0 0 0 -1 117504 12 2436 1 2 0 0 0 0 0 2 0 30 26 1 156 1 2 32 10 1 0 1 3 0 24 0 48 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 384 0 ");
}
TEST_F(MaxPoolGradTiling, maxpoolgrad_tiling_7) {
  using namespace optiling;
  std::string op_name = "MaxPoolGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo ="{\"vars\": {\"ub_size\": 131072, \"l1_size\": 1048576, \"core_num\": 32, \"kh\": 34, \"kw\":2, \"sh\": 4, \"padding\": 1, \"sw\": 4}}";


  std::vector<int64_t> input{15,2,300,307,16};
  std::vector<int64_t> output{15,2,75,77,16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NC1HWC0";
  tensor_input.ori_format = "NHWC";
  
  TeOpTensor tensor_input1;
  tensor_input1.shape = output;
  tensor_input1.dtype = "float16";
  tensor_input1.format = "NC1HWC0";
  tensor_input1.ori_format = "NHWC";
  
  TeOpTensor tensor_input2;
  tensor_input2.shape = output;
  tensor_input2.dtype = "float16";
  tensor_input2.format = "NC1HWC0";
  tensor_input2.ori_format = "NHWC";
  
  
  TeOpTensor tensor_output;
  tensor_output.shape = input;
  tensor_output.dtype = "float16";
  tensor_output.format = "NC1HWC0";
  tensor_output.ori_format = "NHWC";

  TeOpTensorArg tensor_input_arg;
  TeOpTensorArg tensor_input_arg1;
  TeOpTensorArg tensor_input_arg2;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  tensor_input_arg1.tensor.push_back(tensor_input1);
  tensor_input_arg1.arg_type = TA_SINGLE;
  tensor_input_arg2.tensor.push_back(tensor_input2);
  tensor_input_arg2.arg_type = TA_SINGLE;
  
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_input_arg1);
  opParas.inputs.push_back(tensor_input_arg2);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123878";
  OpRunInfo runInfo;
  std::cout << "maxpoolgrad_tiling_7" << std::endl;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "7 15 2 300 307 75 77 15 15 0 0 30 -2 -30 -1 90 32 25 77 130 307 1 12 26 26 2 3 34 48 20 5 25 6 1 9792 0 153 2947200 184800 34 46 0 0 12 11538 1 2 8 1 0 4 0 0 1 32 0 0 0 0 0 18 0 0 0 0 0 5 11545 1 1 0 0 40 0 0 3 1 34 48 0 0 0 0 0 0 0 0 0 34 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 34 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}
TEST_F(MaxPoolGradTiling, maxpoolgrad_tiling_8) {
  using namespace optiling;
  std::string op_name = "MaxPoolGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo ="{\"vars\": {\"ub_size\": 131072, \"l1_size\": 1048576, \"core_num\": 32, \"kh\": 6, \"kw\":2, \"sh\": 7, \"padding\": 1, \"sw\": 1}}";


  std::vector<int64_t> input{10,4,11,1,16};
  std::vector<int64_t> output{10,42,1,1,16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NC1HWC0";
  tensor_input.ori_format = "NHWC";
  
  TeOpTensor tensor_input1;
  tensor_input1.shape = output;
  tensor_input1.dtype = "float16";
  tensor_input1.format = "NC1HWC0";
  tensor_input1.ori_format = "NHWC";
  
  TeOpTensor tensor_input2;
  tensor_input2.shape = output;
  tensor_input2.dtype = "float16";
  tensor_input2.format = "NC1HWC0";
  tensor_input2.ori_format = "NHWC";
  
  
  TeOpTensor tensor_output;
  tensor_output.shape = input;
  tensor_output.dtype = "float16";
  tensor_output.format = "NC1HWC0";
  tensor_output.ori_format = "NHWC";

  TeOpTensorArg tensor_input_arg;
  TeOpTensorArg tensor_input_arg1;
  TeOpTensorArg tensor_input_arg2;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  tensor_input_arg1.tensor.push_back(tensor_input1);
  tensor_input_arg1.arg_type = TA_SINGLE;
  tensor_input_arg2.tensor.push_back(tensor_input2);
  tensor_input_arg2.arg_type = TA_SINGLE;
  
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_input_arg1);
  opParas.inputs.push_back(tensor_input_arg2);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1238788";
  OpRunInfo runInfo;
  std::cout << "maxpoolgrad_tiling_8" << std::endl;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "5 10 4 11 1 1 1 0 0 0 1 -1 1 4 -1 40 32 1 1 11 1 1 1 8 8 1 2 11 1 1 0 0 0 0 352 32 5 704 672 0 0 11 33 1 3 1 1 8 1 0 2 0 0 0 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 11 2 1 28 1 1 8 1 0 0 0 11 0 2 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}
TEST_F(MaxPoolGradTiling, maxpoolgrad_tiling_9) {
  using namespace optiling;
  std::string op_name = "MaxPoolGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo ="{\"vars\": {\"ub_size\": 131072, \"l1_size\": 1048576, \"core_num\": 32, \"kh\": 4, \"kw\":4, \"sh\": 34, \"padding\": 0, \"sw\": 2}}";


  std::vector<int64_t> input{15,2,300,300,16};
  std::vector<int64_t> output{15,2,9,149,16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NC1HWC0";
  tensor_input.ori_format = "NHWC";
  
  TeOpTensor tensor_input1;
  tensor_input1.shape = output;
  tensor_input1.dtype = "float16";
  tensor_input1.format = "NC1HWC0";
  tensor_input1.ori_format = "NHWC";
  
  TeOpTensor tensor_input2;
  tensor_input2.shape = output;
  tensor_input2.dtype = "float16";
  tensor_input2.format = "NC1HWC0";
  tensor_input2.ori_format = "NHWC";
  
  
  TeOpTensor tensor_output;
  tensor_output.shape = input;
  tensor_output.dtype = "float16";
  tensor_output.format = "NC1HWC0";
  tensor_output.ori_format = "NHWC";

  TeOpTensorArg tensor_input_arg;
  TeOpTensorArg tensor_input_arg1;
  TeOpTensorArg tensor_input_arg2;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  tensor_input_arg1.tensor.push_back(tensor_input1);
  tensor_input_arg1.arg_type = TA_SINGLE;
  tensor_input_arg2.tensor.push_back(tensor_input2);
  tensor_input_arg2.arg_type = TA_SINGLE;
  
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_input_arg1);
  opParas.inputs.push_back(tensor_input_arg2);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1238789";
  OpRunInfo runInfo;
  std::cout << "maxpoolgrad_tiling_9" << std::endl;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "4 15 2 300 300 9 149 0 0 0 0 -30 2 -6 0 90 32 3 149 102 300 1 16 26 26 2 3 34 34 12 5 3 9 1 2176 0 34 2880000 42912 4 34 34 266 16 2666 1 2 8 1 0 4 0 0 2 0 0 532 1 34 0 12 12 288 0 0 0 5 2677 1 1 0 0 40 44 576 3 1 0 0 0 0 0 0 0 0 0 0 0 34 0 0 0 0 0 0 0 0 0 0 0 0 0 68 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 24 0 0 0 0 0 0 0 ");
}
TEST_F(MaxPoolGradTiling, maxpoolgrad_tiling_10) {
  using namespace optiling;
  std::string op_name = "MaxPoolGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo ="{\"vars\": {\"ub_size\": 131072, \"l1_size\": 1048576, \"core_num\": 32, \"kh\": 3, \"kw\":3, \"sh\": 34, \"sw\": 2, \"padding\":0}}";


  std::vector<int64_t> input{33,2,300,300,16};
  std::vector<int64_t> output{33,2,9,149,16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NC1HWC0";
  tensor_input.ori_format = "NHWC";
  
  TeOpTensor tensor_input1;
  tensor_input1.shape = output;
  tensor_input1.dtype = "float16";
  tensor_input1.format = "NC1HWC0";
  tensor_input1.ori_format = "NHWC";
  
  TeOpTensor tensor_input2;
  tensor_input2.shape = output;
  tensor_input2.dtype = "float16";
  tensor_input2.format = "NC1HWC0";
  tensor_input2.ori_format = "NHWC";
  
  
  TeOpTensor tensor_output;
  tensor_output.shape = input;
  tensor_output.dtype = "float16";
  tensor_output.format = "NC1HWC0";
  tensor_output.ori_format = "NHWC";

  TeOpTensorArg tensor_input_arg;
  TeOpTensorArg tensor_input_arg1;
  TeOpTensorArg tensor_input_arg2;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  tensor_input_arg1.tensor.push_back(tensor_input1);
  tensor_input_arg1.arg_type = TA_SINGLE;
  tensor_input_arg2.tensor.push_back(tensor_input2);
  tensor_input_arg2.arg_type = TA_SINGLE;
  
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_input_arg1);
  opParas.inputs.push_back(tensor_input_arg2);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "12387888";
  OpRunInfo runInfo;
  std::cout << "maxpoolgrad_tiling_10" << std::endl;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "2 33 2 300 300 9 149 0 0 0 0 -31 1 -6 1 66 32 9 149 300 300 1 16 2 2 2 3 34 33 11 5 9 9 1 1632 32 25 2880000 42912 3 33 33 267 16 2666 1 2 8 1 0 4 0 0 2 0 0 534 1 25 32 11 11 289 0 0 0 5 2677 1 1 0 0 40 44 578 0 0 0 0 0 0 0 0 0 0 0 0 0 34 0 0 0 66 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 22 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}
