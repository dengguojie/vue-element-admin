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
 * \file test_maxpool_tiling.cpp
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

class MaxPoolTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MaxPoolTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MaxPoolTiling TearDown" << std::endl;
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

TEST_F(MaxPoolTiling, maxpool_tiling_0) {
  std::string op_name = "MaxPool";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_ele\": 130560, \"core_num\": 32, \"ksize_h\": 1, \"ksize_w\": 1, \"strides_h\": 1, "
      "\"strides_w\": 1, \"padding\": 0}}";

  std::vector<int64_t> input{16, 79, 69, 208};
  std::vector<int64_t> output{16, 13, 79, 69, 16};

  auto opParas = op::MaxPool("MaxPool");
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

TEST_F(MaxPoolTiling, maxpool_tiling_1) {
  std::string op_name = "MaxPool";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_ele\": 130560, \"core_num\": 32, \"ksize_h\": 1, \"ksize_w\": 1, \"strides_h\": 2, "
      "\"strides_w\": 2, \"padding\": 0}}";

  std::vector<int64_t> input{16, 10, 70, 208};
  std::vector<int64_t> output{16, 13, 5, 35, 16};

  auto opParas = op::MaxPool("MaxPool");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT16, FORMAT_NHWC, {});
  TransformerOpBaseFormat(opParas, "x", FORMAT_NC1HWC0);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input, DT_FLOAT16, FORMAT_NC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 30 7 5 10 70 5 35 9 69 0 0 0 0 2 1 1 3 1 2 1 208 ");
  for (int64_t i = 0; i < 10; i++) {
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  }
}

TEST_F(MaxPoolTiling, maxpool_tiling_2) {
  std::string op_name = "MaxPool";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_ele\": 130560, \"core_num\": 32, \"ksize_h\": 1, \"ksize_w\": 1, \"strides_h\": 2, "
      "\"strides_w\": 2, \"padding\": 0}}";

  std::vector<int64_t> input{16, 62, 250, 208};
  std::vector<int64_t> output{16, 13, 31, 125, 16};

  auto opParas = op::MaxPool("MaxPool");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT16, FORMAT_NHWC, {});
  TransformerOpBaseFormat(opParas, "x", FORMAT_NC1HWC0);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input, DT_FLOAT16, FORMAT_NC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 30 7 5 62 250 31 125 61 249 0 0 0 0 1 3 1 10 1 10 1 208 ");
  for (int64_t i = 0; i < 10; i++) {
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  }
}

TEST_F(MaxPoolTiling, maxpool_tiling_3) {
  std::string op_name = "MaxPool";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_ele\": 130560, \"core_num\": 32, \"ksize_h\": 1, \"ksize_w\": 1, \"strides_h\": 2, "
      "\"strides_w\": 2, \"padding\": 0}}";

  std::vector<int64_t> input{16, 10, 2500, 208};
  std::vector<int64_t> output{16, 13, 5, 1250, 16};

  auto opParas = op::MaxPool("MaxPool");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT16, FORMAT_NHWC, {});
  TransformerOpBaseFormat(opParas, "x", FORMAT_NC1HWC0);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input, DT_FLOAT16, FORMAT_NC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "3 30 7 5 10 2500 5 1250 9 2499 0 0 0 0 1 1 680 1 570 1 570 208 ");
  for (int64_t i = 0; i < 10; i++) {
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  }
}

TEST_F(MaxPoolTiling, maxpool_tiling_4) {
  std::string op_name = "MaxPool";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_ele\": 130560, \"core_num\": 32, \"ksize_h\": 1, \"ksize_w\": 1, \"strides_h\": 2, "
      "\"strides_w\": 2, \"padding\": 1}}";

  std::vector<int64_t> input{16, 10, 70, 208};
  std::vector<int64_t> output{16, 13, 5, 35, 16};

  auto opParas = op::MaxPool("MaxPool");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT16, FORMAT_NHWC, {});
  TransformerOpBaseFormat(opParas, "x", FORMAT_NC1HWC0);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input, DT_FLOAT16, FORMAT_NC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 30 7 5 10 70 5 35 9 69 0 0 0 0 2 1 1 3 1 2 1 208 ");
  for (int64_t i = 0; i < 10; i++) {
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  }
}