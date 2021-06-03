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

class InplaceIndexAddTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "InplaceIndexAddTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "InplaceIndexAddTiling TearDown" << std::endl;
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

TEST_F(InplaceIndexAddTiling, inplace_index_add_tiling_0) {
  using namespace optiling;
  std::string op_name = "InplaceIndexAdd";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("InplaceIndexAdd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\":4, \"indices_size\":4, \"vconv_size\":2, \"axis\":1}}";

  std::vector<int64_t> inputA{2, 3, 8};
  std::vector<int64_t> inputB{2};
  std::vector<int64_t> inputC{2, 2, 8};
  std::vector<int64_t> output{2, 3, 8};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "int32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "int32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234560";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 2 2 2 1 16 24 8 ");
}
