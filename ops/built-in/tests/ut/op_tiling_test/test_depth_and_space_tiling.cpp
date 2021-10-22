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

class DepthAndSpaceTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DepthAndSpaceTiling SetUp" << std::endl;
  }
  static void TearDownTestCase() {
    std::cout << "DepthAndSpaceTiling TearDown" << std::endl;
  }
};

static void run_case(std::vector<int64_t> input_shape, std::vector<int64_t> output_shape, std::string data_format,
                     std::string data_dtype, std::string compile_info, std::string expect_tiling,
                     std::string case_name) {
  using namespace ut_util;
  // no need out the log: OP_EVENT("OP_TILING_UTEST", "case_name = %s", case_name.c_str());
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("DepthToSpace");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  // get op
  auto test_op = op::DepthToSpace("DepthToSpace");
  TENSOR_INPUT_WITH_SHAPE(test_op, x, input_shape, StringToDtype(data_dtype),
                          TypeUtils::SerialStringToFormat(data_format), {});
  TENSOR_OUTPUT_WITH_SHAPE(test_op, y, output_shape, StringToDtype(data_dtype),
                           TypeUtils::SerialStringToFormat(data_format), {});

  optiling::utils::OpRunInfo runInfo;
  const int64_t profiling_test_num = 0;
  RUN_TILING_V3(test_op, iter->second, compile_info, runInfo);
  if (expect_tiling != "") {
    EXPECT_EQ(to_string_int64(runInfo.GetAllTilingData()), expect_tiling);
  }
  for (int64_t i = 0; i < profiling_test_num; i++) {
    RUN_TILING_V3(test_op, iter->second, compile_info, runInfo);
  }
}

TEST_F(DepthAndSpaceTiling, depthandspace_tiling_0) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 8192, \"core_num\": 32, \"dtype\": \"float16\", \"block_size\": 2, \"mode\": \"DCR\"}}";

  std::vector<int64_t> input{16, 2, 2, 16};
  std::vector<int64_t> output{16, 4, 4, 4};
  std::string input_dtype = "float16";
  std::string input_format = "NHWC";
  std::string expect_tiling =
      "4 124 4 0 8 1 8 3 1 0 0 2 0 2 0 1 0 1 0 32 0 0 0 32 0 0 0 2 2 2 16 16 16 1 1 2 0 0 2 0 0 8 0 0 0 0 0 8 0 0 128 "
      "0 0 256 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 32 0 0 32 0 0 32 1 0 0 0 1 0 0 1 1 0 0 2 1 0 0 3 1 0 0 4 1 0 0 5 1 0 0 6 1 0 0 7 "
      "1 0 0 8 1 0 0 9 1 0 0 10 1 0 0 11 1 0 0 12 1 0 0 13 1 0 0 14 1 0 0 15 1 0 0 16 1 0 0 17 1 0 0 18 1 0 0 19 1 0 0 "
      "20 1 0 0 21 1 0 0 22 1 0 0 23 1 0 0 24 1 0 0 25 1 0 0 26 1 0 0 27 1 0 0 28 1 0 0 29 1 0 0 30 1 0 0 31 ";
  run_case(input, output, input_format, input_dtype, compileInfo, expect_tiling, this->test_info_->name());
}

TEST_F(DepthAndSpaceTiling, depthandspace_tiling_1) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 8192, \"core_num\": 32, \"dtype\": \"float16\", \"block_size\": 2, \"mode\": \"DCR\"}}";

  std::vector<int64_t> input{16, 4, 4, 4};
  std::vector<int64_t> output{16, 2, 2, 16};

  std::string input_dtype = "float16";
  std::string input_format = "NHWC";
  std::string expect_tiling =
      "4 124 4 0 2 1 14 3 1 0 0 1 0 1 0 1 0 1 0 16 0 0 0 16 0 0 0 2 4 2 16 16 16 1 1 2 0 0 4 0 0 2 0 0 0 0 0 2 0 0 32 "
      "0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 64 0 0 16 0 0 16 2 0 0 0 2 0 0 2 2 0 0 4 2 0 0 6 2 0 0 8 2 0 0 10 2 0 0 12 2 0 0 "
      "14 2 0 0 16 2 0 0 18 2 0 0 20 2 0 0 22 2 0 0 24 2 0 0 26 2 0 0 28 2 0 0 30 2 0 0 32 2 0 0 34 2 0 0 36 2 0 0 38 "
      "2 0 0 40 2 0 0 42 2 0 0 44 2 0 0 46 2 0 0 48 2 0 0 50 2 0 0 52 2 0 0 54 2 0 0 56 2 0 0 58 2 0 0 60 2 0 0 62 ";
  run_case(input, output, input_format, input_dtype, compileInfo, expect_tiling, this->test_info_->name());
}

TEST_F(DepthAndSpaceTiling, depthandspace_tiling_2) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 8192, \"core_num\": 32, \"dtype\": \"float16\", \"block_size\": 2, \"mode\": \"CRD\"}}";

  std::vector<int64_t> input{1, 8, 2, 3};
  std::vector<int64_t> output{1, 2, 4, 6};

  std::string input_dtype = "float16";
  std::string input_format = "NCHW";
  std::string expect_tiling =
      "6 32 17 0 32 8192 1 1 15 5 5 16 0 6 1 12 3 24 1 2 6 12 24 2 3 2 2 2 1 2 6 12 24 0 0 0 0 16 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 16 16 0 2 0 1 0 0 0 0 0 0 0 0 0 0 0 32 16 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input, output, input_format, input_dtype, compileInfo, expect_tiling, this->test_info_->name());
}

TEST_F(DepthAndSpaceTiling, depthandspace_tiling_3) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 8192, \"core_num\": 32, \"dtype\": \"float16\", \"block_size\": 2, \"mode\": \"DCR\"}}";

  std::vector<int64_t> input{1, 8, 2, 3};
  std::vector<int64_t> output{1, 2, 4, 6};

  std::string input_dtype = "float16";
  std::string input_format = "NCHW";
  std::string expect_tiling =
      "6 28 16 0 32 8192 1 1 15 4 11 16 0 12 1 24 3 1 2 6 12 2 3 2 4 1 2 6 12 0 0 0 0 16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "16 16 0 2 0 1 0 0 0 0 0 0 0 0 0 0 32 16 0 1 1 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input, output, input_format, input_dtype, compileInfo, expect_tiling, this->test_info_->name());
}

TEST_F(DepthAndSpaceTiling, depthandspace_tiling_4) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 8192, \"core_num\": 32, \"dtype\": \"float16\", \"block_size\": 2, \"mode\": \"CRD\"}}";

  std::vector<int64_t> input{1, 2, 3, 8};
  std::vector<int64_t> output{1, 4, 6, 2};
  std::string input_dtype = "float16";
  std::string input_format = "NHWC";
  std::string expect_tiling =
      "6 32 17 0 32 8192 1 1 15 5 3 16 0 4 1 8 2 24 1 2 4 12 24 2 2 3 2 2 1 2 4 12 24 0 0 0 0 16 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 16 16 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 32 16 0 0 2 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  run_case(input, output, input_format, input_dtype, compileInfo, expect_tiling, this->test_info_->name());
}
