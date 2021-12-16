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

class YoloTest_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "YoloTest_UT test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "YoloTest_UT test TearDown" << std::endl;
  }
};

TEST_F(YoloTest_UT, InferShapeYolo_000) {
  ge::op::Yolo op;
  op.UpdateInputDesc("x", create_desc({1, 3, 4}, ge::DT_INT8));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(YoloTest_UT, InferShapeYolo_001) {
  ge::op::Yolo op;
  op.UpdateInputDesc("x", create_desc({1}, ge::DT_INT8));
  op.SetAttr("boxes", 0);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(YoloTest_UT, InferShapeYolo_002) {
  ge::op::Yolo op;
  op.UpdateInputDesc("x", create_desc({1, 13, 13, 13}, ge::DT_INT8));
  op.SetAttr("coords", 3);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(YoloTest_UT, InferShapeYolo_003) {
  ge::op::Yolo op;
  op.UpdateInputDesc("x", create_desc({1, 13, 13, 13}, ge::DT_INT8));
  op.SetAttr("classes", 1025);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(YoloTest_UT, InferShapeYolo_004) {
  ge::op::Yolo op;
  op.UpdateInputDesc("x", create_desc({1, 13, 13, 13}, ge::DT_INT8));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(YoloTest_UT, InferShapeYolo_005) {
  ge::op::Yolo op;
  op.UpdateInputDesc("x", create_desc({1, 425, 13, 13}, ge::DT_FLOAT16));
  op.SetAttr("boxes", 5);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc_coord = op.GetOutputDesc("coord_data");
  EXPECT_EQ(output_desc_coord.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape_coord = {1, 20, 192};
  EXPECT_EQ(output_desc_coord.GetShape().GetDims(), expected_output_shape_coord);

  auto output_desc_obj = op.GetOutputDesc("obj_prob");
  EXPECT_EQ(output_desc_obj.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape_obj = {1, 864};
  EXPECT_EQ(output_desc_obj.GetShape().GetDims(), expected_output_shape_obj);

  auto output_desc_classes = op.GetOutputDesc("classes_prob");
  EXPECT_EQ(output_desc_classes.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape_classes = {1, 80, 864};
  EXPECT_EQ(output_desc_classes.GetShape().GetDims(), expected_output_shape_classes);
}

TEST_F(YoloTest_UT, InferShapeYolo_006) {
  ge::op::Yolo op;
  op.UpdateInputDesc("x", create_desc({1, 13, 13, 13}, ge::DT_INT8));
  op.SetAttr("boxes", -1);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(YoloTest_UT, InferShapeYolo_007) {
  ge::op::Yolo op;
  op.UpdateInputDesc("x", create_desc({1, 13, 13, 13}, ge::DT_INT8));
  op.SetAttr("boxes", false);
  op.SetAttr("coords", false);
  op.SetAttr("classes", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(YoloTest_UT, InferShapeYolo_008) {
  ge::op::Yolo op;
  op.UpdateInputDesc("x", create_desc({1, 13, 13, 13}, ge::DT_INT8));
  op.SetAttr("boxes", false);
  op.SetAttr("coords", false);
  op.SetAttr("classes", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc_coord = op.GetOutputDesc("coord_data");
  EXPECT_EQ(output_desc_coord.GetDataType(), ge::DT_INT8);
  std::vector<int64_t> expected_output_shape_coord = {1, 12, 192};
  EXPECT_EQ(output_desc_coord.GetShape().GetDims(), expected_output_shape_coord);
}