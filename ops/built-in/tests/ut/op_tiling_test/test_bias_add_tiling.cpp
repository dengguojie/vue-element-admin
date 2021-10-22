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
#include "elewise_calculation_ops.h"
#include "array_ops.h"

using namespace std;

class BiasAddTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BiasAddTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BiasAddTiling TearDown" << std::endl;
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
.INPUT(x, TensorType::NumberType())
    .INPUT(bias, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(data_format, String, "NHWC")
*/

TEST_F(BiasAddTiling, BiasAdd_tiling1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAdd("BiasAdd");

  vector<vector<int64_t>> input_shapes = {
      {1, 1, 4},
      {4},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, bias, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_ND, {});
  std::string compileInfo =
      R"({"_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "210": [32, 2, 43680, 21840]}, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "221000000": [10000, 10100], "221000001": [10000, 10100, 20000, 30000], "221000002": [10000, 10100, 20000, 30001], "221000004": [10000, 10100, 20001, 30001]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "221000000": ["_dim_0_0", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_1"]}, "boardcast_bias_shape": [1, 1, 4]})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4 4 4 ");
}

TEST_F(BiasAddTiling, BiasAdd_tiling2) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAdd("BiasAdd");

  vector<vector<int64_t>> input_shapes = {
      {1999, 1999, 4},
      {4},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, bias, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_ND, {});
  std::string compileInfo =
      R"({"_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "210": [32, 2, 43680, 21840]}, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "221000000": [10000, 10100], "221000001": [10000, 10100, 20000, 30000], "221000002": [10000, 10100, 20000, 30001], "221000004": [10000, 10100, 20001, 30001]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "221000000": ["_dim_0_0", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_1"]}, "boardcast_bias_shape": [1, 1, -1]})";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "3996001 4 124876 10407 ");
}
