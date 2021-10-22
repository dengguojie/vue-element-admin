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
 * \file test_concat_offset_tilling.cpp
 * \brief
 */
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "all_ops.h"
#include "test_common.h"
#include "common/utils/ut_op_util.h"

using namespace std;

class ConcatOffsetTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ConcatOffsetTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ConcatOffsetTiling TearDown" << std::endl;
  }
};

TEST_F(ConcatOffsetTiling, concat_offset_tiling_0) {
  using namespace ut_util;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ConcatOffset");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compile_info = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32}}";
  std::vector<int64_t> input{4};
  auto test_op = op::ConcatOffset("ConcatOffset");
  test_op.create_dynamic_input_x(1);
  TENSOR_DY_INPUT_WITH_SHAPE(test_op, x, 0, input, DT_INT32, FORMAT_ND, {});

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compile_info);
  optiling::utils::OpRunInfo run_info;
  ASSERT_TRUE(iter->second.tiling_func_v2_(test_op, op_compile_info, run_info));
  EXPECT_EQ(to_string_int64(run_info.GetAllTilingData()), "4 ");
}
