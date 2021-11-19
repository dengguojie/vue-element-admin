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
#include "transformation_ops.h"
#include "graph/ge_tensor.h"

class FLATTENV2_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "FLATTENV2_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "FLATTENV2_UT TearDown" << std::endl;
  }
};

TEST_F(FLATTENV2_UT, ValidInfershapeTest) {
  ge::op::FlattenV2 op("FlattenV2");
  ge::TensorDesc x_desc = create_desc_shape_range({1, 2, 3}, ge::DT_INT32, ge::FORMAT_ND, {1, 2, 3}, ge::FORMAT_ND,
                                                  {{1, 1}, {2, 2}, {3, 3}});
  x_desc.SetRealDimCnt(3);
  op.UpdateInputDesc("x", x_desc);
  op.SetAttr("axis", 1);
  op.SetAttr("end_axis", -1);
  auto ret = op.InferShapeAndType();
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  std::vector<int64_t> target_output_shape{1, 6};
  auto y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(y_desc.GetShape().GetDims(), target_output_shape);
}

TEST_F(FLATTENV2_UT, ValidInfershapeWithUnknownShapeTest) {
  ge::op::FlattenV2 op("FlattenV2");
  ge::TensorDesc x_desc = create_desc_shape_range({1, -1, 3}, ge::DT_INT32, ge::FORMAT_ND, {1, 2, 3}, ge::FORMAT_ND,
                                                  {{1, 1}, {3, 6}, {3, 3}});
  x_desc.SetRealDimCnt(3);
  op.UpdateInputDesc("x", x_desc);
  op.SetAttr("axis", 1);
  op.SetAttr("end_axis", -1);
  auto ret = op.InferShapeAndType();
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);

  std::vector<int64_t> target_output_shape{1, -1};
  auto y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(y_desc.GetShape().GetDims(), target_output_shape);
  std::vector<std::pair<int64_t, int64_t>> target_output_range{{1, 1}, {9, 18}};
  std::vector<std::pair<int64_t, int64_t>> actual_output_range;
  (void)y_desc.GetShapeRange(actual_output_range);
  for (size_t i = 0; i < actual_output_range.size(); i++) {
    EXPECT_EQ(actual_output_range[i].first, target_output_range[i].first);
    EXPECT_EQ(actual_output_range[i].second, target_output_range[i].second);
  }
}

TEST_F(FLATTENV2_UT, InValidInfershapeTest) {
  ge::op::FlattenV2 op("FlattenV2");
  ge::TensorDesc x_desc = create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {1, 2, 3}, ge::FORMAT_ND,
                                                  {{1, 1}, {3, 6}, {3, 3}});
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
  }

  TEST_F(FLATTENV2_UT, InferShapeWithUnknownRnakTest) {
  ge::op::FlattenV2 op("FlattenV2");
  // unknown rank 
  ge::TensorDesc x_desc = create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {1, 2, 3}, ge::FORMAT_ND,
                                                  {{1, 1}, {3, 6}, {3, 3}});
  op.UpdateInputDesc("x", x_desc);
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
  }

  TEST_F(FLATTENV2_UT, InferShapeWithInvalidAxisTest) {
  ge::op::FlattenV2 op("FlattenV2");
  // unknown rank 
  ge::TensorDesc x_desc = create_desc_shape_range({1, -1, 3}, ge::DT_INT32, ge::FORMAT_ND, {1, 2, 3}, ge::FORMAT_ND,
                                                  {{1, 1}, {3, 6}, {3, 3}});
  x_desc.SetRealDimCnt(3);
  op.UpdateInputDesc("x", x_desc);
  op.SetAttr("axis", 1);
  op.SetAttr("end_axis", -8);
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(FLATTENV2_UT, ValidInfershapeWithUnknownShapeAndShapeRangeTest) {
  ge::op::FlattenV2 op("FlattenV2");
  ge::TensorDesc x_desc = create_desc_shape_range({1, -1, 3}, ge::DT_INT32, ge::FORMAT_ND, {1, 2, 3}, ge::FORMAT_ND,
                                                  {{1, 1}, {3, 6}, {3, -1}});
  x_desc.SetRealDimCnt(3);
  op.UpdateInputDesc("x", x_desc);
  op.SetAttr("axis", 1);
  op.SetAttr("end_axis", -1);
  auto ret = op.InferShapeAndType();
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);

  std::vector<int64_t> target_output_shape{1, -1};
  auto y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(y_desc.GetShape().GetDims(), target_output_shape);
  std::vector<std::pair<int64_t, int64_t>> target_output_range{{1, 1}, {9, -1}};
  std::vector<std::pair<int64_t, int64_t>> actual_output_range;
  (void)y_desc.GetShapeRange(actual_output_range);
  for (size_t i = 0; i < actual_output_range.size(); i++) {
    EXPECT_EQ(actual_output_range[i].first, target_output_range[i].first);
    EXPECT_EQ(actual_output_range[i].second, target_output_range[i].second);
  }
}

 TEST_F(FLATTENV2_UT, InferShapeWithUnknownShapeTest) {
  ge::op::FlattenV2 op("FlattenV2");
  // unknown rank 
  ge::TensorDesc x_desc = create_desc_shape_range({-1, 1, 3}, ge::DT_INT32, ge::FORMAT_ND, {1, 2, 3}, ge::FORMAT_ND,
                                                  {{1, 3}, {1, 6}, {3, 3}});
  x_desc.SetRealDimCnt(3);
  op.UpdateInputDesc("x", x_desc);
  op.SetAttr("axis", 1);
  op.SetAttr("end_axis", -1);
  auto ret = op.InferShapeAndType();
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  
  auto y_desc = op.GetOutputDesc("y");
  // shape range is static when shape dim is known
  std::vector<std::pair<int64_t, int64_t>> target_output_range{{1, 3}, {3, 3}};
  std::vector<std::pair<int64_t, int64_t>> actual_output_range;
  (void)y_desc.GetShapeRange(actual_output_range);
  for (size_t i = 0; i < actual_output_range.size(); i++) {
    EXPECT_EQ(actual_output_range[i].first, target_output_range[i].first);
    EXPECT_EQ(actual_output_range[i].second, target_output_range[i].second);
  }
}