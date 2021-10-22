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
 * \file test_pad_tiling.cpp
 * \brief
 */
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include <graph/utils/type_utils.h>
#define private public
#include "register/op_tiling_registry.h"
#include "all_ops.h"
#include "test_common.h"
#include "op_tiling/op_tiling_util.h"
#include "common/utils/ut_op_util.h"

using namespace std;
using namespace ge;
class PadTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "PadTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "PadTiling TearDown" << std::endl;
  }
};

const int64_t profiling_test_num = 10;
static void run_case(std::vector<int64_t> input_shape, std::string data_dtype, std::vector<int32_t> const_value,
                     std::string src_ori_format, std::string src_format, std::string compile_info,
                     std::string expect_tiling, std::string case_name) {
  using namespace ut_util;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Pad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto test_op = op::Pad("Pad");
  int64_t input_len = input_shape.size();
  std::vector<int64_t> const_shape{input_len, 2};

  TENSOR_INPUT_WITH_SHAPE(test_op, x, input_shape, StringToDtype(data_dtype),
                          TypeUtils::SerialStringToFormat(src_ori_format), {});
  TransformerOpBaseFormat(test_op, "x", TypeUtils::SerialStringToFormat(src_format));
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, paddings, const_shape, DT_INT32, FORMAT_ND, const_value);

  optiling::utils::OpCompileInfo op_compile_info(case_name.c_str(), compile_info);

  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(test_op, op_compile_info, runInfo));
  if (expect_tiling != "") {
    EXPECT_EQ(to_string_int64(runInfo.GetAllTilingData()), expect_tiling);
  }
  for (int64_t i = 0; i < profiling_test_num; i++) {
    iter->second.tiling_func_v2_(test_op, op_compile_info, runInfo);
  }
}

TEST_F(PadTiling, rpad_tiling_0) {
  std::string compileInfo = "{\"vars\": {\"ub_size\": 65536, \"core_num\": 32, \"dtype_rate\": 2}}";

  std::vector<int64_t> input{64, 64, 64, 64};
  std::vector<int64_t> padding_shape{4, 2};
  std::vector<int32_t> padding_value{0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> output{64, 64, 64, 64};
  std::string expect_tiling = "0 1 1 1 1 1 1 1 33554432 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
  std::string input_dtype = "float32";
  std::string format = "ND";
  run_case(input, input_dtype, padding_value, format, format, compileInfo, expect_tiling, this->test_info_->name());
}

TEST_F(PadTiling, rpad_tiling_1) {
  std::string compileInfo = "{\"vars\": {\"ub_size\": 65536, \"core_num\": 32, \"dtype_rate\": 2}}";

  std::vector<int64_t> input{64, 64, 64, 4};
  std::vector<int64_t> padding_shape{4, 2};
  std::vector<int32_t> padding_value{1, 1, 1, 1, 1, 1, 0, 0};
  std::vector<int64_t> output{66, 66, 66, 4};
  std::string input_dtype = "float32";
  std::string format = "ND";
  std::string expect_tiling = "2 1 1 1 1 1 64 64 512 0 0 0 0 0 0 0 0 0 0 1 1 1 1 8 8 2 ";
  run_case(input, input_dtype, padding_value, format, format, compileInfo, expect_tiling, this->test_info_->name());
}
TEST_F(PadTiling, rpad_tiling_2) {
  std::string compileInfo = "{\"vars\": {\"ub_size\": 65536, \"core_num\": 32, \"dtype_rate\": 2}}";

  std::vector<int64_t> ori_input{64, 64, 64, 4};
  std::vector<int64_t> padding_shape{4, 2};
  std::vector<int32_t> padding_value{1, 1, 32, 16, 1, 1, 0, 0};

  std::string input_dtype = "float32";
  std::string input_ori_format = "NCHW";
  std::string input_format = "NC1HWC0";
  std::string expect_tiling = "1 1 1 1 1 1 64 4 8192 0 0 0 0 0 0 0 0 0 0 1 1 2 1 128 128 1 ";
  run_case(ori_input, input_dtype, padding_value, input_ori_format, input_format, compileInfo, expect_tiling,
           this->test_info_->name());
}
