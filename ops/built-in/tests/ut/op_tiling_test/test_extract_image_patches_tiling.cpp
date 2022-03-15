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

/*!
 * \file test_extract_image_patches_tiling.cpp
 * \brief dynamic tiling test of extract_image_patches
 */
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#define private public
#include "array_ops.h"
#include "common/utils/ut_op_util.h"
#include "transformation_ops.h"
#include "op_tiling/op_tiling_util.h"
#include "register/op_tiling_registry.h"
#include "test_common.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class ExtractImagePatchesTiling : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "ExtractImagePatchesTiling SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "ExtractImagePatchesTiling TearDown" << std::endl; }
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

TEST_F(ExtractImagePatchesTiling, ExtractImagePatches_tiling_test_1) {
  std::string op_name = "ExtractImagePatches";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo =
      "{\"_pattern\": \"ExtractImagePatches\", \"coreNum\":32, \"inSpecialDevice\": false,"
      "\"workspaceDimen\": [9, 4, 16], \"realC\": 2, \"ksizeHW\": [2, 2], \"strideHW\": [2, 2], \"dilateHW\": [1, 1],"
      "\"_vars\": {\"10000\": [\"dim_0\", \"multi_core_factor_0\"]}, \"_normal_vars\": {\"10000\":[]}, "
      "\"_attr_vars\": {\"10000\":[]}, \"_custom_vars\": {\"10000\": [\"dim_0\", \"multi_core_factor_0\"]}}";

  std::vector<int64_t> input{2, 6, 6, 2};
  std::vector<int64_t> output{2, 3, 3, 8};

  auto opParas = op::ExtractImagePatches(op_name);
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT16, FORMAT_NHWC, {});
  TransformerOpBaseFormat(opParas, "x", FORMAT_NC1HWC0);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input, DT_FLOAT16, FORMAT_NC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 1 ");
}