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
#include <climits>
#include "op_proto_test_util.h"
#include "util.h"
#include "graph/utils/op_desc_utils.h"
#include "array_ops.h"
#include "graph/ge_tensor.h"

using std::make_pair;
class SHAPE_SHAPEN_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SHAPE_SHAPEN_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SHAPE_SHAPEN_UT TearDown" << std::endl;
  }
};

TEST_F(SHAPE_SHAPEN_UT, RangeToStrTest) {
  std::vector<int64_t> target_value_range = {100, -1, 1, -1, 1, 20, 10, 100};
  std::vector<std::pair<int64_t, int64_t>> x_range = {make_pair(100, -1), make_pair(1, -1),
                                                      make_pair(1, 20),   make_pair(10, 100)};
  std::string range_str = ge::RangeToString(x_range);

  std::string target_range_str = "[{100,-1},{1,-1},{1,20},{10,100}]";

  EXPECT_EQ(range_str, target_range_str);
}

TEST_F(SHAPE_SHAPEN_UT, InferShapeValueRange) {
  ge::op::Shape op("Shape");
  op.UpdateInputDesc("x", create_desc({-1, -1, -1, -1}, ge::DT_INT32));
  op.UpdateOutputDesc("y", create_desc({4}, ge::DT_INT32));
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  std::vector<std::pair<int64_t, int64_t>> x_range = {make_pair(100, -1), make_pair(1, -1),
                                                      make_pair(1, 20),   make_pair(10, 100)};
  op_desc->MutableInputDesc(0)->SetShapeRange(x_range);
  auto ret = op_desc->CallInferValueRangeFunc(op);
  ASSERT_EQ(ret, 0);

  std::vector<std::pair<int64_t, int64_t>> y_value_range;
  op_desc->MutableOutputDesc(0)->GetValueRange(y_value_range);
  std::vector<int64_t> target_value_range = {100, -1, 1, -1, 1, 20, 10, 100};
  std::vector<int64_t> output_value_range;
  for (auto pair : y_value_range) {
    output_value_range.push_back(pair.first);
    output_value_range.push_back(pair.second);
  }
  EXPECT_EQ(output_value_range, target_value_range);

  ASSERT_EQ(op_desc->CallInferFormatFunc(op), ge::GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->MutableOutputDesc(0)->GetFormat(), ge::FORMAT_ND);
  ASSERT_EQ(op_desc->MutableOutputDesc(0)->GetOriginFormat(), ge::FORMAT_ND);
}
