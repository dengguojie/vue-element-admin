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
#define private public
#include "register/op_tiling_registry.h"
#include "nonlinear_fuc_ops.h"
#include "array_ops.h"

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

using namespace ge;
#include "common/utils/ut_op_util.h"
using namespace ut_util;

TEST_F(PReluTiling, PReluTiling_test_1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("PRelu");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  auto opParas = op::PRelu("PRelu");

  vector<vector<int64_t>> input_shapes = {
      {32},
      {1},
  };

  vector<ge::DataType> dtypes = {
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
  };

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, weight, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});

  std::string compileInfo = R"({"broadcast_weight_shape": [1], "_fusion_index": [[0]], "push_status": 1, "_pattern": "Broadcast", "_flag_info": [false, false, true, false, false, false, false], "_outs_uint1": false, "_base_info": {"000": [32, 2, 31728, 15856]}, "_elewise_vars": {"0": [10000, 20000, 30000]}, "_vars": {"0": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"0": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"0": []}, "_custom_vars": {"0": []}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "32 32 32 ");
}

TEST_F(PReluTiling, PReluTiling_test_2) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("PRelu");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  auto opParas = op::PRelu("PRelu");

  vector<vector<int64_t>> input_shapes = {
      {32},
      {1},
  };

  vector<ge::DataType> dtypes = {
    ge::DT_FLOAT,
    ge::DT_FLOAT,
  };

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, weight, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});

  std::string compileInfo = R"({"broadcast_weight_shape": [1], "_fusion_index": [[0]], "push_status": 1, "_pattern": "Broadcast", "_flag_info": [false, false, true, false, false, false, false], "_outs_uint1": false, "_base_info": {"000": [32, 2, 31728, 15856]}, "_elewise_vars": {"0": [10000, 20000, 30000]}, "_vars": {"0": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"0": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"0": []}, "_custom_vars": {"0": []}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "32 32 32 ");
}
