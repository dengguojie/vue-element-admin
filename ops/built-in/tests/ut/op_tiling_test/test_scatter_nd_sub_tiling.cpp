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
#define private public
#include "register/op_tiling_registry.h"
#include "matrix_calculation_ops.h"
#include "array_ops.h"

using namespace std;

class ScatterNdSubTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ScatterNdSubTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ScatterNdSubTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream& tiling_data) {
  auto data = tiling_data.str();
  string result;
  int64_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
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
.INPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .OUTPUT(var, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32,DT_INT8,DT_UINT8}))
    .ATTR(use_locking, Bool, false)
*/

TEST_F(ScatterNdSubTiling, scatter_nd_sub_tiling_0) {
  std::string op_name = "ScatterNdSub";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterNdSub");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{2, 3, 8};
  std::vector<int64_t> inputB{2, 2};
  std::vector<int64_t> inputC{2, 8};
  std::vector<int64_t> output{2, 3, 8};

  auto opParas = op::ScatterNdSub("ScatterNdSub");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_INT32, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 8 6 8 0 4 16 0 8 24 8 0 0 0 0 0 2 2 ");
}

TEST_F(ScatterNdSubTiling, scatter_nd_sub_tiling_1) {
  std::string op_name = "ScatterNdSub";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterNdSub");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{2, 3, 88888};
  std::vector<int64_t> inputB{2, 2};
  std::vector<int64_t> inputC{2, 88888};
  std::vector<int64_t> output{2, 3, 88888};

  auto opParas = op::ScatterNdSub("ScatterNdSub");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 16672 32 88888 0 4 177776 2 25400 266664 88888 0 0 0 0 0 2 2 ");
}

TEST_F(ScatterNdSubTiling, scatter_nd_sub_tiling_2) {
  std::string op_name = "ScatterNdSub";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterNdSub");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{2, 3, 7};
  std::vector<int64_t> inputB{2, 2};
  std::vector<int64_t> inputC{2, 7};
  std::vector<int64_t> output{2, 3, 7};

  auto opParas = op::ScatterNdSub("ScatterNdSub");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "3 8 1 7 0 4 14 0 14 21 7 0 0 0 0 0 2 2 ");
}

TEST_F(ScatterNdSubTiling, scatter_nd_sub_tiling_3) {
  std::string op_name = "ScatterNdSub";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterNdSub");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{2, 3, 7};
  std::vector<int64_t> inputB{88888, 2};
  std::vector<int64_t> inputC{88888, 7};
  std::vector<int64_t> output{2, 3, 7};

  auto opParas = op::ScatterNdSub("ScatterNdSub");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4 8 1 7 5 19056 622216 0 7 21 7 0 0 0 0 0 2 88888 ");
}

TEST_F(ScatterNdSubTiling, scatter_nd_sub_tiling_4) {
  std::string op_name = "ScatterNdSub";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterNdSub");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 0, \"core_num\": 0, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{2, 3, 8};
  std::vector<int64_t> inputB{2, 2};
  std::vector<int64_t> inputC{2, 8};
  std::vector<int64_t> output{2, 3, 8};

  auto opParas = op::ScatterNdSub("ScatterNdSub");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_INT32, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ScatterNdSubTiling, scatter_nd_sub_tiling_5) {
  std::string op_name = "ScatterNdSub";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterNdSub");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{2, 3, 88888};
  std::vector<int64_t> inputB{2, 2};
  std::vector<int64_t> inputC{2, 88888};
  std::vector<int64_t> output{2, 3, 88888};

  auto opParas = op::ScatterNdSub("ScatterNdSub");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_INT32, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 16672 32 88888 0 4 177776 2 25400 266664 88888 0 0 0 0 0 2 2 ");
}

TEST_F(ScatterNdSubTiling, scatter_nd_sub_tiling_6) {
  std::string op_name = "ScatterNdSub";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterNdSub");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{2, 3, 7};
  std::vector<int64_t> inputB{2, 2};
  std::vector<int64_t> inputC{2, 7};
  std::vector<int64_t> output{2, 3, 7};

  auto opParas = op::ScatterNdSub("ScatterNdSub");
  TENSOR_INPUT_WITH_SHAPE(opParas, var, inputA, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, indices, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, updates, inputC, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_INT32, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "3 8 1 7 0 4 14 0 14 21 7 0 0 0 0 0 2 2 ");
}
