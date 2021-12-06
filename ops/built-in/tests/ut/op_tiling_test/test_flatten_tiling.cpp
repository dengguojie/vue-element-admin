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

#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#define private public
#include "register/op_tiling_registry.h"
#include "transformation_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"
using namespace std;

class FlattenTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "FlattenTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "FlattenTiling TearDown" << std::endl;
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
#include "test_common.h"
/*
.INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                          DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64,
                          DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                           DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64,
                           DT_FLOAT, DT_FLOAT16}))
    .ATTR(axis, Int, 1)
*/

TEST_F(FlattenTiling, FlattenTiling_tiling_1) {
  std::string op_name = "Flatten";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 63488, \"block_size\": 8}}";

  std::vector<int64_t> input{4, 4, 4, 4};
  std::vector<int64_t> output{4, 64};

  auto opParas = op::Flatten(op_name);
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "8 32 0 8 0 8 ");
  for (int64_t i = 0; i < 10; i++) {
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  }
}
