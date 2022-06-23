/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * ragged_bin_count tiling ut case
 */
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "selection_ops.h"
#include "array_ops.h"
#define private public
#include "register/op_tiling_registry.h"
#include "common/utils/ut_op_util.h"
#include "test_common.h"
#include "math_ops.h"

using namespace ge;
using namespace ut_util;

class RaggedBinCountTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AssignTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AssignTiling TearDown" << std::endl;
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

TEST_F(RaggedBinCountTiling, RaggedBinCount_tiling_test_case_1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("RaggedBinCount");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::RaggedBinCount("RaggedBinCount");

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32}}";

  std::vector<int64_t> input_splits{6};
  std::vector<int64_t> input_values{10};
  std::vector<int64_t> input_size{1};
  std::vector<int64_t> input_weights{10};
  std::vector<int64_t> output0{5, 5};

  std::vector<ge::DataType> dtype = {ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT};
  std::vector<int32_t> size_data{5};

  TENSOR_INPUT_WITH_SHAPE(opParas, splits, input_splits, dtype[1], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, values, input_values, dtype[0], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, size, input_size, dtype[0], FORMAT_ND, size_data);
  TENSOR_INPUT_WITH_SHAPE(opParas, weights, input_weights, dtype[2], FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output, output0, dtype[2], FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "10 5 6 10 25 1 1 1983 15864 ");
}

TEST_F(RaggedBinCountTiling, RaggedBinCount_tiling_test_case_2) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("RaggedBinCount");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::RaggedBinCount("RaggedBinCount");

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32}}";

  std::vector<int64_t> input_splits{202};
  std::vector<int64_t> input_values{10};
  std::vector<int64_t> input_size{1};
  std::vector<int64_t> input_weights{10};
  std::vector<int64_t> output0{201, 5};

  std::vector<ge::DataType> dtype = {ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT};
  std::vector<int32_t> size_data{5};

  TENSOR_INPUT_WITH_SHAPE(opParas, splits, input_splits, dtype[1], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, values, input_values, dtype[0], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, size, input_size, dtype[0], FORMAT_ND, size_data);
  TENSOR_INPUT_WITH_SHAPE(opParas, weights, input_weights, dtype[2], FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output, output0, dtype[2], FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "10 5 202 10 1005 1 1 1983 15864 ");
}

TEST_F(RaggedBinCountTiling, RaggedBinCount_tiling_test_case_3) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("RaggedBinCount");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::RaggedBinCount("RaggedBinCount");

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32}}";

  std::vector<int64_t> input_splits{1002};
  std::vector<int64_t> input_values{10};
  std::vector<int64_t> input_size{1};
  std::vector<int64_t> input_weights{10};
  std::vector<int64_t> output0{1001, 5};

  std::vector<ge::DataType> dtype = {ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT};
  std::vector<int32_t> size_data{5};

  TENSOR_INPUT_WITH_SHAPE(opParas, splits, input_splits, dtype[1], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, values, input_values, dtype[0], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, size, input_size, dtype[0], FORMAT_ND, size_data);
  TENSOR_INPUT_WITH_SHAPE(opParas, weights, input_weights, dtype[2], FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output, output0, dtype[2], FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "10 5 1002 10 5005 1 1 1983 15864 ");
}

TEST_F(RaggedBinCountTiling, RaggedBinCount_tiling_test_case_4) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("RaggedBinCount");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::RaggedBinCount("RaggedBinCount");

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32}}";

  std::vector<int64_t> input_splits{5002};
  std::vector<int64_t> input_values{10};
  std::vector<int64_t> input_size{1};
  std::vector<int64_t> input_weights{10};
  std::vector<int64_t> output0{5001, 5};

  std::vector<ge::DataType> dtype = {ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT};
  std::vector<int32_t> size_data{5};

  TENSOR_INPUT_WITH_SHAPE(opParas, splits, input_splits, dtype[1], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, values, input_values, dtype[0], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, size, input_size, dtype[0], FORMAT_ND, size_data);
  TENSOR_INPUT_WITH_SHAPE(opParas, weights, input_weights, dtype[2], FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output, output0, dtype[2], FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "10 5 5002 10 25005 1 1 1983 15864 ");
}

TEST_F(RaggedBinCountTiling, RaggedBinCount_tiling_test_case_5) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("RaggedBinCount");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::RaggedBinCount("RaggedBinCount");

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32}}";

  std::vector<int64_t> input_splits{6};
  std::vector<int64_t> input_values{1000};
  std::vector<int64_t> input_size{1};
  std::vector<int64_t> input_weights{1000};
  std::vector<int64_t> output0{5, 5};

  std::vector<ge::DataType> dtype = {ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT};
  std::vector<int32_t> size_data{5};

  TENSOR_INPUT_WITH_SHAPE(opParas, splits, input_splits, dtype[1], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, values, input_values, dtype[0], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, size, input_size, dtype[0], FORMAT_ND, size_data);
  TENSOR_INPUT_WITH_SHAPE(opParas, weights, input_weights, dtype[2], FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output, output0, dtype[2], FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "32 5 6 1000 25 31 39 1983 15864 ");
}

TEST_F(RaggedBinCountTiling, RaggedBinCount_tiling_test_case_6) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("RaggedBinCount");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::RaggedBinCount("RaggedBinCount");

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32}}";

  std::vector<int64_t> input_splits{6};
  std::vector<int64_t> input_values{10000};
  std::vector<int64_t> input_size{1};
  std::vector<int64_t> input_weights{10000};
  std::vector<int64_t> output0{5, 5};

  std::vector<ge::DataType> dtype = {ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT};
  std::vector<int32_t> size_data{5};

  TENSOR_INPUT_WITH_SHAPE(opParas, splits, input_splits, dtype[1], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, values, input_values, dtype[0], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, size, input_size, dtype[0], FORMAT_ND, size_data);
  TENSOR_INPUT_WITH_SHAPE(opParas, weights, input_weights, dtype[2], FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output, output0, dtype[2], FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "32 5 6 10000 25 312 328 1983 15864 ");
}

TEST_F(RaggedBinCountTiling, RaggedBinCount_tiling_test_case_7) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("RaggedBinCount");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::RaggedBinCount("RaggedBinCount");

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32}}";

  std::vector<int64_t> input_splits{6};
  std::vector<int64_t> input_values{10000};
  std::vector<int64_t> input_size{1};
  std::vector<int64_t> input_weights{10000};
  std::vector<int64_t> output0{5, 20000};

  std::vector<ge::DataType> dtype = {ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT};
  std::vector<int32_t> size_data{20000};

  TENSOR_INPUT_WITH_SHAPE(opParas, splits, input_splits, dtype[1], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, values, input_values, dtype[0], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, size, input_size, dtype[0], FORMAT_ND, size_data);
  TENSOR_INPUT_WITH_SHAPE(opParas, weights, input_weights, dtype[2], FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output, output0, dtype[2], FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "32 20000 6 10000 100000 312 328 1983 15864 ");
}

TEST_F(RaggedBinCountTiling, RaggedBinCount_tiling_test_case_8) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("RaggedBinCount");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::RaggedBinCount("RaggedBinCount");

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32}}";

  std::vector<int64_t> input_splits{202};
  std::vector<int64_t> input_values{10000};
  std::vector<int64_t> input_size{1};
  std::vector<int64_t> input_weights{10000};
  std::vector<int64_t> output0{201, 20000};

  std::vector<ge::DataType> dtype = {ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT};
  std::vector<int32_t> size_data{20000};

  TENSOR_INPUT_WITH_SHAPE(opParas, splits, input_splits, dtype[1], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, values, input_values, dtype[0], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, size, input_size, dtype[0], FORMAT_ND, size_data);
  TENSOR_INPUT_WITH_SHAPE(opParas, weights, input_weights, dtype[2], FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output, output0, dtype[2], FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "32 20000 202 10000 4020000 312 328 1983 15864 ");
}

TEST_F(RaggedBinCountTiling, RaggedBinCount_tiling_test_case_9) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("RaggedBinCount");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::RaggedBinCount("RaggedBinCount");

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32}}";

  std::vector<int64_t> input_splits{6};
  std::vector<int64_t> input_values{2, 5};
  std::vector<int64_t> input_size{1};
  std::vector<int64_t> input_weights{2, 5};
  std::vector<int64_t> output0{5, 5};

  std::vector<ge::DataType> dtype = {ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT};
  std::vector<int32_t> size_data{5};

  TENSOR_INPUT_WITH_SHAPE(opParas, splits, input_splits, dtype[1], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, values, input_values, dtype[0], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, size, input_size, dtype[0], FORMAT_ND, size_data);
  TENSOR_INPUT_WITH_SHAPE(opParas, weights, input_weights, dtype[2], FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output, output0, dtype[2], FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "10 5 6 10 25 1 1 1983 15864 ");
}

TEST_F(RaggedBinCountTiling, RaggedBinCount_tiling_test_case_10) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("RaggedBinCount");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::RaggedBinCount("RaggedBinCount");

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32}}";

  std::vector<int64_t> input_splits{6};
  std::vector<int64_t> input_values{1000, 5000};
  std::vector<int64_t> input_size{1};
  std::vector<int64_t> input_weights{1000, 5000};
  std::vector<int64_t> output0{5, 5};

  std::vector<ge::DataType> dtype = {ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT};
  std::vector<int32_t> size_data{5};

  TENSOR_INPUT_WITH_SHAPE(opParas, splits, input_splits, dtype[1], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, values, input_values, dtype[0], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, size, input_size, dtype[0], FORMAT_ND, size_data);
  TENSOR_INPUT_WITH_SHAPE(opParas, weights, input_weights, dtype[2], FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, output, output0, dtype[2], FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "32 5 6 5000000 25 156250 156250 1983 15864 ");
}