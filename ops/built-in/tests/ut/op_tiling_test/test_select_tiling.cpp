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
#include "selection_ops.h"
#include "array_ops.h"

using namespace std;

class SelectTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SelectTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SelectTiling TearDown" << std::endl;
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
.INPUT(condition, TensorType({DT_BOOL}))
    .INPUT(x1,TensorType::BasicType())
    .INPUT(x2,TensorType::BasicType())
    .OUTPUT(y,TensorType::BasicType())
*/

TEST_F(SelectTiling, Select_tiling1) {
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find("Select");
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  auto opParas = op::Select("Select");

  vector<vector<int64_t>> input_shapes = {
      {2, 2, 2, 2},
      {2, 2, 2, 2},
      {2, 2, 2, 2},
  };

  vector<ge::DataType> dtypes = {ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_SHAPE(opParas, condition, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x1, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x2, input_shapes[2], dtypes[2], ge::FORMAT_ND, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[2], ge::DT_FLOAT16, ge::FORMAT_ND, {});
  std::string compileInfo =
      R"({ "_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, false, true, false, false, false], "_base_info": {"100": [32, 4, 32768, 16384]}, "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] }, "boardcast_condition_fill": []})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "16 16 16 ");
}

TEST_F(SelectTiling, Select_tiling2) {
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find("Select");
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  auto opParas = op::Select("Select");

  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {4, 4, 4, 4},
      {4, 4, 4, 4},
  };

  vector<ge::DataType> dtypes = {ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_SHAPE(opParas, condition, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x1, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x2, input_shapes[2], dtypes[2], ge::FORMAT_ND, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[2], ge::DT_FLOAT16, ge::FORMAT_ND, {});
  std::string compileInfo =
      R"({ "_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, false, true, false, false, false], "_base_info": {"100": [32, 4, 32768, 16384]}, "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] }, "boardcast_condition_fill": []})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "256 256 256 ");
}

TEST_F(SelectTiling, Select_tiling3) {
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find("Select");
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  auto opParas = op::Select("Select");

  vector<vector<int64_t>> input_shapes = {
      {4},
      {4, 4, 4, 4},
      {4, 4, 4, 4},
  };

  vector<ge::DataType> dtypes = {ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_SHAPE(opParas, condition, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x1, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, x2, input_shapes[2], dtypes[2], ge::FORMAT_ND, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[2], ge::DT_FLOAT16, ge::FORMAT_ND, {});
  std::string compileInfo =
      R"({ "_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, false, true, false, false, false], "_base_info": {"100": [32, 4, 32768, 16384]}, "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] }, "boardcast_condition_fill": [1,1,1]})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4 4 4 ");
}
