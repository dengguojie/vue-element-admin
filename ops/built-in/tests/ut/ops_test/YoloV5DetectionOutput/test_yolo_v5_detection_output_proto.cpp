#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

class YoloV5DetectionOutputTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "YoloV5DetectionOutput Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "YoloV5DetectionOutput Proto Test TearDown" << std::endl;
  }
};

TEST_F(YoloV5DetectionOutputTest, yolo_v5_detection_output_test_case_1){
  ge::op::YoloV5DetectionOutput op;
  op.create_dynamic_input_x(10);
  op.UpdateDynamicInputDesc("x", 0, create_desc_with_ori(
    {1, 12, 6416}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 12, 6416}, ge::FORMAT_ND));
  op.UpdateDynamicInputDesc("x", 1, create_desc_with_ori(
    {1, 12, 1616}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 12, 6416}, ge::FORMAT_ND));
  op.UpdateDynamicInputDesc("x", 2, create_desc_with_ori(
    {1, 12, 416}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 12, 6416}, ge::FORMAT_ND));
  op.UpdateDynamicInputDesc("x", 3, create_desc_with_ori(
    {1, 19216}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 19216}, ge::FORMAT_ND));
  op.UpdateDynamicInputDesc("x", 4, create_desc_with_ori(
    {1, 4816}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 4816}, ge::FORMAT_ND));
  op.UpdateDynamicInputDesc("x", 5, create_desc_with_ori(
    {1, 1216}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 1216}, ge::FORMAT_ND));
  op.UpdateDynamicInputDesc("x", 6, create_desc_with_ori(
    {1, 80, 19216}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 80, 19216}, ge::FORMAT_ND));
  op.UpdateDynamicInputDesc("x", 7, create_desc_with_ori(
    {1, 80, 4816}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 80, 4816}, ge::FORMAT_ND));
  op.UpdateDynamicInputDesc("x", 8, create_desc_with_ori(
    {1, 80, 1216}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 80, 1216}, ge::FORMAT_ND));
  op.UpdateDynamicInputDesc("x", 9, create_desc_with_ori(
    {1, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 4}, ge::FORMAT_ND));

  std::vector<float> bias_list = {10., 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};
  op.SetAttr("biases", bias_list);
  op.SetAttr("out_box_dim", 2);
  op.SetAttr("post_nms_topn", 1024);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("box_out");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {1, 6144};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(YoloV5DetectionOutputTest, InfershapeYoloV5DetectionOutput_001) {
  ge::op::YoloV5DetectionOutput op;
  op.create_dynamic_input_x(1);
  op.UpdateDynamicInputDesc("x", 0, create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(YoloV5DetectionOutputTest, InfershapeYoloV5DetectionOutput_002) {
  ge::op::YoloV5DetectionOutput op;
  op.create_dynamic_input_x(1);
  op.UpdateDynamicInputDesc(
      "x", 0, create_desc_with_ori({1, 12, 6416}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 12, 6416}, ge::FORMAT_ND));
  op.SetAttr("post_nms_topn", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(YoloV5DetectionOutputTest, InfershapeYoloV5DetectionOutput_003) {
  ge::op::YoloV5DetectionOutput op;
  op.create_dynamic_input_x(1);
  op.UpdateDynamicInputDesc(
      "x", 0, create_desc_with_ori({1, 12, 6416}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 12, 6416}, ge::FORMAT_ND));
  op.SetAttr("out_box_dim", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(YoloV5DetectionOutputTest, InfershapeYoloV5DetectionOutput_004) {
  ge::op::YoloV5DetectionOutput op;
  op.create_dynamic_input_x(1);
  op.UpdateDynamicInputDesc(
      "x", 0, create_desc_with_ori({1, 12, 6416}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 12, 6416}, ge::FORMAT_ND));
  op.SetAttr("out_box_dim", 3);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("box_out");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {1, 6, 512};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}