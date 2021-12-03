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

#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

class RpnProposalsTest_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RpnProposalsTest_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RpnProposalsTest_UT TearDown" << std::endl;
  }
};

TEST_F(RpnProposalsTest_UT, InferShapeRpnProposals_000) {
  ge::op::RpnProposals op;
  op.UpdateInputDesc("rois", create_desc({4, 3, 1}, ge::DT_INT8));
  op.SetAttr("post_nms_num", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RpnProposalsTest_UT, InferShapeRpnProposals_001) {
  ge::op::RpnProposals op;
  op.UpdateInputDesc("rois", create_desc({4, 3, 1}, ge::DT_FLOAT16));
  op.SetAttr("post_nms_num", 12);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto rois_output_desc = op.GetOutputDesc("sorted_box");
  std::vector<int64_t> expected_output_shape_rois = {12, 4};
  EXPECT_EQ(rois_output_desc.GetShape().GetDims(), expected_output_shape_rois);
  EXPECT_EQ(rois_output_desc.GetDataType(), ge::DT_FLOAT16);
}