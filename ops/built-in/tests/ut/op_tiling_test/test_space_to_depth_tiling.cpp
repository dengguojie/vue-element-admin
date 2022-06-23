/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file test_depth_and_space_tiling.cpp
 * \brief dynamic shape tiling test of depth and space ops
 */
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include <graph/utils/type_utils.h>
#define private public
#include <register/op_tiling.h>
#include "register/op_tiling_registry.h"
#include "op_tiling/op_tiling_util.h"
#include "transformation_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"

using namespace std;
using namespace optiling;
using namespace ge;

class SpaceToDepthTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SpaceToDepthTiling SetUp" << std::endl;
  }
  static void TearDownTestCase() {
    std::cout << "SpaceToDepthTiling TearDown" << std::endl;
  }
};

static void run_case(std::vector<int64_t> input_shape, std::vector<int64_t> output_shape, std::string data_format,
                     std::string data_dtype, std::string compile_info, std::string expect_tiling,
                     std::string case_name, int64_t block_size) {
  using namespace ut_util;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SpaceToDepth");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  // get op
  auto test_op = op::SpaceToDepth("SpaceToDepth");
  TENSOR_INPUT_WITH_SHAPE(test_op, x, input_shape, StringToDtype(data_dtype),
                          TypeUtils::SerialStringToFormat(data_format), {});
  TENSOR_OUTPUT_WITH_SHAPE(test_op, y, output_shape, StringToDtype(data_dtype),
                           TypeUtils::SerialStringToFormat(data_format), {});
  test_op.SetAttr("block_size", block_size);

  optiling::utils::OpRunInfo runInfo;
  const int64_t profiling_test_num = 1;
  RUN_TILING_V3(test_op, iter->second, compile_info, runInfo);
  if (expect_tiling != "") {
    EXPECT_EQ(to_string_int64(runInfo.GetAllTilingData()), expect_tiling);
  }
  for (int64_t i = 0; i < profiling_test_num; i++) {
    RUN_TILING_V3(test_op, iter->second, compile_info, runInfo);
  }
}

TEST_F(SpaceToDepthTiling, SpaceToDepth_tiling_0) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 8192, \"core_num\": 32, \"dtype\": \"float16\", \"block_size\": 2, \"mode\": \"DCR\"}}";

  std::vector<int64_t> output{16, 2, 2, 16};
  std::vector<int64_t> input{16, 4, 4, 4};
  std::string input_dtype = "float16";
  std::string input_format = "NHWC";
  std::string expect_tiling =
      "5 160 3 0 8 2 1 8 2 0 0 0 64 0 64 0 1 1 1 1 1024 0 0 0 1024 0 0 0 32 32 3 33 33 33 1 1 0 0 1 2 128 16 0 0 32 0 0 "
      "512 0 0 2 0 0 2 0 0 8 0 0 0 0 0 8 0 0 128 0 0 256 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 "
      "32 0 0 512 0 0 2 0 0 2 0 0 8 0 0 0 0 0 8 0 0 128 0 0 256 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 1 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  int64_t block_size = 2;
  run_case(input, output, input_format, input_dtype, compileInfo, expect_tiling, this->test_info_->name(), block_size);
}
