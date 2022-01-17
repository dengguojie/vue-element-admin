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

#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "selection_ops.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"
#define private public
#include "register/op_tiling_registry.h"
#include "common/utils/ut_op_util.h"
#include "test_common.h"

using namespace std;

class MovingSumWithSigmoidTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MovingSumWithSigmoidTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MovingSumWithSigmoidTiling TearDown" << std::endl;
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

TEST_F(MovingSumWithSigmoidTiling, tset_0) {
  using namespace optiling;
  std::string op_name = "MovingSumWithSigmoid";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = "{\"vars\": {\"core_num\": 32}}";

  // do tilling, get runInfo
  OpRunInfo runInfo;

  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234560";
  TeOpParas opParas;
  opParas.op_type = op_name;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 32 ");
}