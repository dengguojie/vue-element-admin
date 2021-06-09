/*
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class PReluTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "PReluTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "PReluTiling TearDown" << std::endl;
  }
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

TEST_F(PReluTiling, PReluTiling_test_1) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("PRelu");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
      {32},
      {1},
  };

  vector<string> dtypes = {
    "float16",
    "float16",
  };

  for (size_t i = 0; i < input_shapes.size(); i++) {
    tensorInputs.tensor.clear();
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);
  }

  TeOpTensor tensorOutput;
  tensorOutput.shape = input_shapes[0];
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "PRelu";

  std::string compileInfo = R"({"broadcast_weight_shape": [1], "_fusion_index": [[0]], "push_status": 1, "_pattern": "Broadcast", "_flag_info": [false, false, true, false, false, false, false], "_base_info": {"000": [32, 2, 31728, 15856]}, "_elewise_vars": {"0": [10000, 20000, 30000]}, "_vars": {"0": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"0": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"0": []}, "_custom_vars": {"0": []}})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "PReluTiling_test_1";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "PReluTiling tiling_data:" << to_string(runInfo.tiling_data) << std::endl;

  EXPECT_EQ(to_string(runInfo.tiling_data), "32 32 32 ");
}
