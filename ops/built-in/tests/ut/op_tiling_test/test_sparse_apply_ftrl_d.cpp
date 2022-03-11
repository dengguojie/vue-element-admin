/*
 * Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "all_ops.h"
#include <vector>
#include <iostream>
#define private public
#include <gtest/gtest.h>
#include "common/utils/ut_op_util.h"
#include "register/op_tiling_registry.h"


using namespace ge;
using namespace std;
using namespace ut_util;

class SparseApplyFtrlDTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "SparseApplyFtrlDTiling SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "SparseApplyFtrlDTiling TearDown" << endl;
  }
};

static string to_string(const stringstream& tiling_data) {
  auto data = tiling_data.str();
  string result;
  int32_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += to_string(tmp);
    result += " ";
  }
  return result;
}

TEST_F(SparseApplyFtrlDTiling, sparseapplyftrld_tiling_0) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SparseApplyFtrlD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  string compileInfo =
      "{\"vars\": {\"core_num\": 32, \"ub_size\": 131072, \"indices_dsize\": 4, \"ub_take_parts\": 1, "
      "\"ub_block_num\":4, \"cache_threshold_col\":7}}";

  auto opParas = op::SparseApplyFtrlD("SparseApplyFtrlD");
  vector<vector<int64_t>> input_shapes = {{12, 16, 32}, {12, 16, 32}, {12, 16, 32}, {12, 16, 32}, {12}};
  vector<vector<int64_t>> output_shapes = {{12, 16, 32}, {12, 16, 32}, {12, 16, 32}};
  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32};

  TENSOR_INPUT_WITH_SHAPE(opParas, var, input_shapes[0], dtypes[0], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, accum, input_shapes[1], dtypes[1], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, linear, input_shapes[2], dtypes[2], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, grad, input_shapes[3], dtypes[3], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, input_shapes[4], dtypes[4], ge::FORMAT_NHWC, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output_shapes[0], ge::DT_FLOAT, ge::FORMAT_NHWC, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, accum, output_shapes[1], ge::DT_FLOAT, ge::FORMAT_NHWC, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, linear, output_shapes[2], ge::DT_FLOAT, ge::FORMAT_NHWC, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 12 0 1 0 0 1 512 12 0 32 0 0 0 0 0 0 0 ");
}
TEST_F(SparseApplyFtrlDTiling, sparseapplyftrld_tiling_1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SparseApplyFtrlD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  string compileInfo =
      "{\"vars\": {\"core_num\": 32, \"ub_size\": 131072, \"indices_dsize\": 4, \"ub_take_parts\": 1, "
      "\"ub_block_num\":4, \"cache_threshold_col\":7}}";

  auto opParas = op::SparseApplyFtrlD("SparseApplyFtrlD");
  vector<vector<int64_t>> input_shapes = {{12, 2, 3}, {12, 2, 3}, {12, 2, 3}, {12, 2, 3}, {12}};
  vector<vector<int64_t>> output_shapes = {{12, 2, 3}, {12, 2, 3}, {12, 2, 3}};
  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32};

  TENSOR_INPUT_WITH_SHAPE(opParas, var, input_shapes[0], dtypes[0], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, accum, input_shapes[1], dtypes[1], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, linear, input_shapes[2], dtypes[2], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, grad, input_shapes[3], dtypes[3], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, input_shapes[4], dtypes[4], ge::FORMAT_NHWC, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output_shapes[0], ge::DT_FLOAT, ge::FORMAT_NHWC, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, accum, output_shapes[1], ge::DT_FLOAT, ge::FORMAT_NHWC, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, linear, output_shapes[2], ge::DT_FLOAT, ge::FORMAT_NHWC, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 1 0 12 0 0 12 6 12 12 12 0 0 0 0 0 0 0 ");
}
TEST_F(SparseApplyFtrlDTiling, sparseapplyftrld_tiling_2) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SparseApplyFtrlD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  string compileInfo =
      "{\"vars\": {\"core_num\": 32, \"ub_size\": 131072, \"indices_dsize\": 4, \"ub_take_parts\": 1, "
      "\"ub_block_num\":4, \"cache_threshold_col\":7}}";

  auto opParas = op::SparseApplyFtrlD("SparseApplyFtrlD");
  vector<vector<int64_t>> input_shapes = {{12, 32, 32}, {12, 32, 32}, {12, 32, 32}, {12, 32, 32}, {12}};
  vector<vector<int64_t>> output_shapes = {{12, 32, 32}, {12, 32, 32}, {12, 32, 32}};
  vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32};

  TENSOR_INPUT_WITH_SHAPE(opParas, var, input_shapes[0], dtypes[0], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, accum, input_shapes[1], dtypes[1], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, linear, input_shapes[2], dtypes[2], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, grad, input_shapes[3], dtypes[3], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, input_shapes[4], dtypes[4], ge::FORMAT_NHWC, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output_shapes[0], ge::DT_FLOAT, ge::FORMAT_NHWC, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, accum, output_shapes[1], ge::DT_FLOAT, ge::FORMAT_NHWC, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, linear, output_shapes[2], ge::DT_FLOAT, ge::FORMAT_NHWC, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "3 24 0 12 0 0 12 1024 12 0 32 2 512 512 0 512 0 512 ");
}