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
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"
#include "selection_ops.h"
#include "array_ops.h"

using namespace std;

class TileDTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TileDTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TileDTiling TearDown" << std::endl;
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
.INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .REQUIRED_ATTR(multiples, ListInt)
*/

TEST_F(TileDTiling, TileD_tiling1) {
  std::string op_name = "TileD";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  // dynamic_tile_d_llt_case_1
  std::string compileInfo =
      R"({ "_pattern": "Broadcast", "push_status": 0, "_flag_info": [false, false, true, false, false, false, false], "_base_info": {"000": [32, 4, 32768, 16384]}, "_elewise_vars": {"1":[10200, 20000, 30000], "2":[10200, 20000, 30001], "3":[10200, 20000, 30002], "5": [10200, 20001, 30001], "6":[10200, 20001, 30002], "9":[10200, 20002, 30002]}, "_vars": {"1":["_dim_2_0", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_2_0", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_2_0", "_block_factor_0", "_ub_factor_2"],  "5": ["_dim_2_0", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_2_0", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_2_0", "_block_factor_2", "_ub_factor_2"]}, "tiling_info": [1, 0, 1, 1, -1, 42763, 16, -1]})";

  std::vector<int64_t> inputA{777};
  std::vector<int64_t> output{42763, 16, 777};
  ge::DataType in_dtype = ge::DT_FLOAT;
  ge::DataType dtype = ge::DT_FLOAT;

  auto opParas = op::TileD("TileD");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputA, in_dtype, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, dtype, ge::FORMAT_ND, {});
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo.c_str());
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "777 1337 2 ");
}
