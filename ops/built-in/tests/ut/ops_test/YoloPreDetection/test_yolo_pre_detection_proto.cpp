#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

class YoloPreDetectionTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "YoloPreDetection Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "YoloPreDetection Proto Test TearDown" << std::endl;
  }
};

TEST_F(YoloPreDetectionTest, yolo_pre_detection_test_case_1){
  ge::op::YoloPreDetection op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 255, 80, 80}, ge::DT_FLOAT, ge::FORMAT_ND,
                                               {1, 255, 80, 80}, ge::FORMAT_ND));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  op.SetAttr("yolo_version", "V5");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("coord_data");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {1, 12, 6416};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  output_desc = op.GetOutputDesc("obj_prob");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  expected_output_shape = {1, 19216};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  output_desc = op.GetOutputDesc("classes_prob");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  expected_output_shape = {1, 80, 19216};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(YoloPreDetectionTest, InferShapeYoloPreDetection_001) {
  ge::op::YoloPreDetection op;
  op.UpdateInputDesc("x", create_desc({1, 255, 80, 80}, ge::DT_FLOAT));
  op.SetAttr("boxes", 0);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(YoloPreDetectionTest, InferShapeYoloPreDetection_002) {
  ge::op::YoloPreDetection op;
  op.UpdateInputDesc("x", create_desc({1, 255, 80, 80}, ge::DT_FLOAT));
  op.SetAttr("coords", 3);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(YoloPreDetectionTest, InferShapeYoloPreDetection_003) {
  ge::op::YoloPreDetection op;
  op.UpdateInputDesc("x", create_desc({1, 255, 80, 80}, ge::DT_FLOAT));
  op.SetAttr("classes", 1025);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(YoloPreDetectionTest, InferShapeYoloPreDetection_004) {
  ge::op::YoloPreDetection op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 255, 80, 80}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 255, 80, 80}, ge::FORMAT_ND));
  op.SetAttr("boxes", false);
  op.SetAttr("coords", false);
  op.SetAttr("classes", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("coord_data");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {1, 12, 6416};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(YoloPreDetectionTest, InferShapeYoloPreDetection_005) {
  ge::op::YoloPreDetection op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 80, 80, 80}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 255, 80, 80}, ge::FORMAT_ND));
  op.SetAttr("boxes", false);
  op.SetAttr("coords", false);
  op.SetAttr("classes", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}