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
#define private public
#include "register/op_tiling_registry.h"
#include "reduce_ops.h"
#include "array_ops.h"

using namespace std;

class ReduceMeanTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "ReduceMeanTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "ReduceMeanTiling TearDown" << std::endl;
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
/*
.INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(axes, ListInt)
    .ATTR(keep_dims, Bool, false)
*/

TEST_F(ReduceMeanTiling, ReduceMeanTiling1) {
  std::string op_name = "ReduceMeanD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = R"({ "_ori_axis": [0], "_pattern": "CommReduce","push_status": 0,"_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5], "_ub_info": [16256], "_ub_info_rf": [16256], "_vars": {"-1000500": ["_dim_1_0", "_block_factor", "_ub_factor"]}})";

  std::vector<int64_t> input{1};
  std::vector<int64_t> output{1};
  ge::DataType in_dtype = ge::DT_FLOAT;

  auto opParas = op::ReduceMeanD("ReduceMeanD");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, in_dtype, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, in_dtype, ge::FORMAT_ND, {});
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 1 ");
}

TEST_F(ReduceMeanTiling, ReduceMeanTiling2) {
  std::string op_name = "ReduceMeanD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5, 4, 9], "_ub_info": [16256, 16000, 16256], "_ub_info_rf": [16256, 16000, 16256], "reduce_mean_cof_dtype": "float32"})";

  std::vector<int64_t> input{7, 2};
  std::vector<int64_t> output{7};
  ge::DataType in_dtype = ge::DT_FLOAT;

  auto opParas = op::ReduceMeanD("ReduceMeanD");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, in_dtype, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, in_dtype, ge::FORMAT_ND, {});
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "7 2 8 8 1056964608 ");
}

TEST_F(ReduceMeanTiling, ReduceMeanTiling3) {
  std::string op_name = "ReduceMeanD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "_common_info": [2, 1, 16, 0, 1], "_pattern_info": [5, 4, 9], "_ub_info": [31488, 31104, 31488], "_ub_info_rf": [31488, 31104, 31488], "reduce_mean_cof_dtype": "float16"})";

  std::vector<int64_t> input{7, 2};
  std::vector<int64_t> output{7};
  ge::DataType in_dtype = ge::DT_FLOAT16;

  auto opParas = op::ReduceMeanD("ReduceMeanD");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, in_dtype, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, in_dtype, ge::FORMAT_ND, {});
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "7 2 16 16 14336 ");
}

TEST_F(ReduceMeanTiling, ReduceTiling4) {
  std::string op_name = "ReduceMean";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = R"({"_idx_before_reduce": 0, "_pattern": "CommReduce", "_common_info": [32, 1, 8, 1, 1],
  "_pattern_info": [2147483646], "_ub_info": [32512], "_ub_info_rf": [32512], "_reduce_shape_known": true,
  "_const_shape_post": true, "_compile_pattern": 2147483646, "_block_dims": {"2147483646": 32}, "_atomic_flags": {"2147483646": false}})";

  std::vector<int64_t> input{1, 10};
  std::vector<int64_t> output{1, 10};
  std::vector<int64_t> input_axis{0};
  std::vector<int32_t> axis = {0};
  ge::DataType in_dtype = ge::DT_FLOAT;

  auto opParas = op::ReduceMean("ReduceMean");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, in_dtype, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, axes, input_axis, ge::DT_INT64, FORMAT_ND, axis);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, in_dtype, ge::FORMAT_ND, {});
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
}

TEST_F(ReduceMeanTiling, ReduceTiling5) {
  std::string op_name = "ReduceMean";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = R"({"_idx_before_reduce": 0, "_pattern": "CommReduce", "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [-1], "_ub_info": [32512], "_ub_info_rf": [32512], "_reduce_shape_known": true, "_const_shape_post": true, "_compile_pattern": -1, "_block_dims": {"-1": 32}, "_atomic_flags": {"-1": false},"reduce_mean_cof_dtype":"float32"})";

  std::vector<int64_t> input{0, 10,3};
  std::vector<int64_t> output{0, 10,3};
  std::vector<int64_t> input_axis{1};
  std::vector<int64_t> axis{1};
  ge::DataType in_dtype = ge::DT_FLOAT;

  auto opParas = op::ReduceMean("ReduceMean");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, in_dtype, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, axes, input_axis, ge::DT_INT64, FORMAT_ND, axis);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, in_dtype, ge::FORMAT_ND, {});
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
}

TEST_F(ReduceMeanTiling, ReduceTiling7) {
  std::string op_name = "ReduceMean";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = R"({"_idx_before_reduce": 0, "_pattern": "CommReduce", "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [-1], "_ub_info": [32512], "_ub_info_rf": [32512], "_reduce_shape_known": true, "_const_shape_post": true, "_compile_pattern": -1, "_block_dims": {"-1": 32}, "_atomic_flags": {"-1": false},"reduce_mean_cof_dtype":"float16"})";

  std::vector<int64_t> input{0, 10,3};
  std::vector<int64_t> output{0, 10,3};
  std::vector<int64_t> input_axis{1};
  std::vector<int64_t> axis{1};
  ge::DataType in_dtype = ge::DT_FLOAT16;

  auto opParas = op::ReduceMean("ReduceMean");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, in_dtype, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, axes, input_axis, ge::DT_INT64, FORMAT_ND, axis);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, in_dtype, ge::FORMAT_ND, {});
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
}

