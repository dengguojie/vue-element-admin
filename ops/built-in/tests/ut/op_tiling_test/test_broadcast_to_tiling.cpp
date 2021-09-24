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
#include "pad_ops.h"
#include "array_ops.h"

using namespace std;

class BroadcastToTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BroadcastToTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BroadcastToTiling TearDown" << std::endl;
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

using namespace ge;
#include "common/utils/ut_op_util.h"
using namespace ut_util;
/*
.INPUT(x, TensorType::BasicType())
    .INPUT(shape, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(y, TensorType::BasicType())
*/

TEST_F(BroadcastToTiling, BroadcastTo_tiling_test_1) {
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find("BroadcastTo");
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  auto opParas = op::BroadcastTo("BroadcastTo");

  vector<vector<int64_t>> input_shapes = {
      {1, 1, 5},
      {3},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32};

  vector<int32_t> buf = {3, 1, 5};
  vector<int64_t> output_shape = {3, 1, 5};
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, shape, input_shapes[1], dtypes[1], FORMAT_ND, buf);

  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shape, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  std::string compileInfo =
      R"( {"_pattern": "Broadcast", "push_status": 0,"_flag_info": [false, false, true, false, false, false, false], "_base_info": {"000": [32, 2, 43680, 21840]}, "_elewise_vars": {"0": [10000, 10100], "1": [10000, 10100, 20000, 30000], "2": [10000, 10100, 20000, 30001], "3": [10000, 10100, 20000, 30002], "5": [10000, 10100, 20001, 30001], "6": [10000, 10100, 20001, 30002], "9": [10000, 10100, 20002, 30002]}, "_vars": {"0": ["_dim_0_0", "_dim_1_0"], "1": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_1_0", "_block_factor_2", "_ub_factor_2"]}})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 ");
}
