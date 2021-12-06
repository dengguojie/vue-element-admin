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
 * \file test_max_pool_v3_tiling.cpp
 * \brief dynamic tiling test of max_pool
 */
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "nn_pooling_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"
#include "test_common.h"
using namespace std;
using namespace ut_util;
using namespace ge;

class MaxPoolV3Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MaxPoolV3Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MaxPoolV3Tiling TearDown" << std::endl;
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

TEST_F(MaxPoolV3Tiling, max_pool_v3_tiling_0) {
  std::string op_name = "MaxPoolV3";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"ksize_h\": 3, \"ksize_w\": 3, \"strides_h\": 2, "
      "\"strides_w\": 2, \"padding\": 2, \"ceil_mode\": 0, \"pad_top\": 1, \"pad_bottom\": 1, \"pad_left\": 1, "
      "\"pad_right\": 1, \"global\": 0}}";

  std::vector<int64_t> input{1, 56, 56, 64};
  std::vector<int64_t> output{1, 4, 28, 28, 16};

  auto opParas = op::MaxPoolV3("MaxPoolV3");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT16, FORMAT_NHWC, {});
  TransformerOpBaseFormat(opParas, "x", FORMAT_NC1HWC0);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input, DT_FLOAT16, FORMAT_NC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 4 1 1 56 56 28 28 57 57 1 0 1 0 1 11 1 2 6 2 6 4 ");
  for (int64_t i = 0; i < 10; i++) {
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  }
}

TEST_F(MaxPoolV3Tiling, maxpool_tiling_1) {
  std::string op_name = "MaxPool";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_ele\": 130560, \"core_num\": 32, \"ksize_h\": 1, \"ksize_w\": 1, \"strides_h\": 1, "
      "\"strides_w\": 1, \"padding\": 0}}";

  std::vector<int64_t> input{16, 79, 69, 208};
  std::vector<int64_t> output{16, 13, 79, 69, 16};

  auto opParas = op::MaxPoolV3("MaxPoolV3");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT16, FORMAT_NHWC, {});
  TransformerOpBaseFormat(opParas, "x", FORMAT_NC1HWC0);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input, DT_FLOAT16, FORMAT_NC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 32 35432 35416 79 69 79 69 79 69 0 0 0 0 1 1 1 4 2792 4 2776 0 ");
  for (int64_t i = 0; i < 10; i++) {
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  }
}
TEST_F(MaxPoolV3Tiling, max_pool_v3_tiling_2) {
  std::string op_name = "MaxPoolV3";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"ksize_h\": 3, \"ksize_w\": 3, \"strides_h\": 2, "
      "\"strides_w\": 2, \"padding\": 2, \"ceil_mode\": 1, \"pad_top\": 1, \"pad_bottom\": 1, \"pad_left\": 1, "
      "\"pad_right\": 1, \"global\": 0}}";

  std::vector<int64_t> input{1, 56, 56, 64};
  std::vector<int64_t> output{1, 4, 29, 29, 16};

  auto opParas = op::MaxPoolV3("MaxPoolV3");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT16, FORMAT_NHWC, {});
  TransformerOpBaseFormat(opParas, "x", FORMAT_NC1HWC0);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input, DT_FLOAT16, FORMAT_NC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 4 1 1 56 56 29 29 59 59 1 2 1 2 1 10 1 2 9 2 9 4 ");
  for (int64_t i = 0; i < 10; i++) {
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  }
}
TEST_F(MaxPoolV3Tiling, max_pool_v3_tiling_global) {
  std::string op_name = "MaxPoolV3";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"ksize_h\": 1, \"ksize_w\": 3, \"strides_h\": 1, "
      "\"strides_w\": 1, \"padding\": 1, \"ceil_mode\": 0, \"pad_top\": 0, \"pad_bottom\": 0, \"pad_left\": 0, "
      "\"pad_right\": 0, \"global\": 1}}";

  std::vector<int64_t> input{32, 3, 3, 16};
  std::vector<int64_t> output{32, 1, 1, 1, 16};

  auto opParas = op::MaxPoolV3("MaxPoolV3");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT16, FORMAT_NHWC, {});
  TransformerOpBaseFormat(opParas, "x", FORMAT_NC1HWC0);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input, DT_FLOAT16, FORMAT_NC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "6 32 1 1 3 3 1 1 3 3 0 0 0 0 1 1 1 0 0 0 0 32 ");
  for (int64_t i = 0; i < 10; i++) {
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  }
}