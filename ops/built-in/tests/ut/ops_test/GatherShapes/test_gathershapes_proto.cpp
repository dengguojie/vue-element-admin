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

class GATHERSHAPES_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GATHERSHAPES_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GATHERSHAPES_UT TearDown" << std::endl;
  }
};

TEST_F(GATHERSHAPES_UT, InfershapeTest) {
  ge::op::GatherShapes op("GatherShapes");
  op.create_dynamic_input_x(2);
  op.UpdateDynamicInputDesc("x", 0, create_desc({1, 2, 3}, ge::DT_INT32));
  op.UpdateDynamicInputDesc("x", 1, create_desc({3, 4, 5}, ge::DT_INT32));
  op.SetAttr("axes",{{0,0},{0,1},{1,1},{1,2}});
  auto ret = op.InferShapeAndType();
  ASSERT_EQ(ret, 0);
  std::vector<int64_t> target_output_shape{4};
  auto shape_desc = op.GetOutputDesc("shape");
  EXPECT_EQ(shape_desc.GetShape().GetDims(), target_output_shape);
}

TEST_F(GATHERSHAPES_UT, InfershapeWithInvalidAttrTest) {
  ge::op::GatherShapes op("GatherShapes");
  op.create_dynamic_input_x(2);
  op.UpdateDynamicInputDesc("x", 0, create_desc({1, 2, 3}, ge::DT_INT32));
  op.UpdateDynamicInputDesc("x", 1, create_desc({3, 4, 5}, ge::DT_INT32));
  op.SetAttr("axes",{{3,0},{0,1},{1,1},{1,2}});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(GATHERSHAPES_UT, InfershapeWithUnknownShapeTest) {
  ge::op::GatherShapes op("GatherShapes");
  op.create_dynamic_input_x(1);
  op.UpdateDynamicInputDesc("x", 0, create_desc({-1,1,2}, ge::DT_INT32));
  std::vector<std::pair<int64_t, int64_t>> x_range = {std::make_pair(1, 20), std::make_pair(1, 1),
                                                      std::make_pair(2, 2)};
  op.SetAttr("axes",{{0,0},{0,1}});
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  op_desc->MutableInputDesc(0)->SetShapeRange(x_range);
  auto ret = op_desc->CallInferValueRangeFunc(op);
  ASSERT_EQ(ret, 0);
  std::vector<std::pair<int64_t, int64_t>> y_value_range;
  op_desc->MutableOutputDesc(0)->GetValueRange(y_value_range);
  std::vector<int64_t> target_value_range = {1,20,1,1};
  std::vector<int64_t> output_value_range;
  for (auto pair : y_value_range) {
    output_value_range.push_back(pair.first);
    output_value_range.push_back(pair.second);
  }
  ASSERT_EQ(output_value_range, target_value_range);
}

TEST_F(GATHERSHAPES_UT, InfershapeWithUnknownShapeRangeTest) {
  ge::op::GatherShapes op("GatherShapes");
  op.create_dynamic_input_x(1);
  op.UpdateDynamicInputDesc("x", 0, create_desc({-1,1,2}, ge::DT_INT32));
  std::vector<std::pair<int64_t, int64_t>> x_range = {std::make_pair(1, -1), std::make_pair(1, 1),
                                                      std::make_pair(2, 2)};
  op.SetAttr("axes",{{0,0},{0,1}});
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  op_desc->MutableInputDesc(0)->SetShapeRange(x_range);
  auto ret = op_desc->CallInferValueRangeFunc(op);
  ASSERT_EQ(ret, 0);

  std::vector<std::pair<int64_t, int64_t>> y_value_range;
  op_desc->MutableOutputDesc(0)->GetValueRange(y_value_range);
  // -1 in value range ->INT64_MAX
  std::vector<int64_t> target_value_range = {1, INT64_MAX, 1, 1};
  std::vector<int64_t> output_value_range;
  for (auto pair : y_value_range) {
    output_value_range.push_back(pair.first);
    output_value_range.push_back(pair.second);
  }
  ASSERT_EQ(output_value_range, target_value_range);
}
