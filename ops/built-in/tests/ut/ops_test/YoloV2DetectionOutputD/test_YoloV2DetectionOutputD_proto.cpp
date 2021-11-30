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

class YoloV2DetectionOutputDTest_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "YoloV2DetectionOutputDTest_UT test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "YoloV2DetectionOutputDTest_UT test TearDown" << std::endl;
  }
};

TEST_F(YoloV2DetectionOutputDTest_UT, InferShapeYoloV2DetectionOutputD_000) {
  ge::op::YoloV2DetectionOutputD op;
  op.UpdateInputDesc("coord_data", create_desc({}, ge::DT_INT8));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(YoloV2DetectionOutputDTest_UT, InferShapeYoloV2DetectionOutputD_001) {
  ge::op::YoloV2DetectionOutputD op;
  op.UpdateInputDesc("coord_data", create_desc({1, 4, 1}, ge::DT_INT8));
  op.SetAttr("post_nms_topn", "zero");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(YoloV2DetectionOutputDTest_UT, InferShapeYoloV2DetectionOutputD_002) {
  ge::op::YoloV2DetectionOutputD op;
  op.UpdateInputDesc("coord_data", create_desc({1, 4, 1}, ge::DT_FLOAT16));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc_box = op.GetOutputDesc("box_out");
  EXPECT_EQ(output_desc_box.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape_box = {1, 3072};
  EXPECT_EQ(output_desc_box.GetShape().GetDims(), expected_output_shape_box);

  auto output_desc_num = op.GetOutputDesc("box_out_num");
  EXPECT_EQ(output_desc_num.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape_num = {1, 8};
  EXPECT_EQ(output_desc_num.GetShape().GetDims(), expected_output_shape_num);
}