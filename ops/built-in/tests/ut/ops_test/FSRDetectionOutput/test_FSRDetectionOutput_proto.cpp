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
#include <iostream>
#include <climits>
#include "op_proto_test_util.h"
#include "util.h"
#include "graph/utils/op_desc_utils.h"
#include "nn_detect_ops.h"
#include "graph/ge_tensor.h"

class FSRDetectionOutput_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "FSRDetectionOutput_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "FSRDetectionOutput_UT TearDown" << std::endl;
  }
};

TEST_F(FSRDetectionOutput_UT, InferShapeFSRDetectionOutput_000) {
  ge::op::FSRDetectionOutput op;
  op.UpdateInputDesc(
      "rois", create_desc_with_ori({5, 5, 3200}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {5, 5, 3200}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("bbox_delta", create_desc_with_ori({10000, 2, 1, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
                                                        {10000, 2, 1, 1, 16}, ge::FORMAT_HWCN));
  op.UpdateInputDesc("score", create_desc_with_ori({10000, 1, 1, 1, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
                                                   {10000, 1, 1, 1, 16}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("im_info",
                     create_desc_with_ori({5, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {5, 16}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("actual_rois_num",
                     create_desc_with_ori({5, 8}, ge::DT_INT32, ge::FORMAT_NHWC, {5, 8}, ge::FORMAT_HWCN));
  op.SetAttr("batch_rois", 5);
  op.SetAttr("num_classes", 5);

  auto status_verify = op.VerifyAllAttr(true);
  EXPECT_EQ(status_verify, ge::GRAPH_SUCCESS);

  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto output_desc_num = op.GetOutputDesc("actual_bbox_num");
  EXPECT_EQ(output_desc_num.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape_num = {5, 5, 8};
  EXPECT_EQ(output_desc_num.GetShape().GetDims(), expected_output_shape_num);

  auto output_desc_box = op.GetOutputDesc("box");
  EXPECT_EQ(output_desc_box.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape_box = {5, 5, 1024, 8};
  EXPECT_EQ(output_desc_box.GetShape().GetDims(), expected_output_shape_box);
}
