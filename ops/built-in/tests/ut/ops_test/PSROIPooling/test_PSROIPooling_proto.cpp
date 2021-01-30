#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

class PSROIPoolingProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "PSROIPooling Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "PSROIPooling Proto Test TearDown" << std::endl;
  }
};

TEST_F(PSROIPoolingProtoTest, roi_pooling_infershape_test_1) {
  ge::op::PSROIPooling op;

  auto tensor_desc = create_desc_with_ori({1, 2*7*7, 14, 14}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 2*7*7, 14, 14}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("rois", create_desc({1, 5, 16}, ge::DT_FLOAT16));
  op.SetAttr("output_dim", 2);
  op.SetAttr("group_size", 7);
  op.SetAttr("spatial_scale", 0.0625f);

  auto status1 = op.VerifyAllAttr(true);
  EXPECT_EQ(status1, ge::GRAPH_SUCCESS);
  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(PSROIPoolingProtoTest, roi_pooling_infershape_test_2) {
  ge::op::PSROIPooling op;

  auto tensor_desc = create_desc_with_ori({1, 2*129*129, 14, 14}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 2*129*129, 14, 14}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("rois", create_desc({1, 5, 16}, ge::DT_FLOAT16));
  op.SetAttr("output_dim", 2);
  op.SetAttr("group_size", 129);
  op.SetAttr("spatial_scale", 0.0625f);

  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(PSROIPoolingProtoTest, roi_pooling_infershape_test_3) {
  ge::op::PSROIPooling op;

  auto tensor_desc = create_desc_with_ori({1, 2*7*7, 14, 14}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 2*7*7, 14, 14}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("rois", create_desc({1, 5, 16}, ge::DT_FLOAT16));
  op.SetAttr("output_dim", 3);
  op.SetAttr("group_size", 7);
  op.SetAttr("spatial_scale", 0.0625f);

  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(PSROIPoolingProtoTest, roi_pooling_verify_test_1) {
  ge::op::PSROIPooling op;

  auto tensor_desc = create_desc({1, 2*7*7, 14, 14}, ge::DT_FLOAT16);
  tensor_desc.SetFormat(ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("rois", create_desc({1, 5}, ge::DT_FLOAT16));
  op.SetAttr("output_dim", 2);
  op.SetAttr("group_size", 7);
  op.SetAttr("spatial_scale", 0.0625f);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(PSROIPoolingProtoTest, roi_pooling_verify_test_2) {
  ge::op::PSROIPooling op;

  auto tensor_desc = create_desc({1, 2*7*7, 14}, ge::DT_FLOAT16);
  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("rois", create_desc({1, 5, 16}, ge::DT_FLOAT16));
  op.SetAttr("output_dim", 2);
  op.SetAttr("group_size", 7);
  op.SetAttr("spatial_scale", 0.0625f);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(PSROIPoolingProtoTest, roi_pooling_verify_test_3) {
  ge::op::PSROIPooling op;

  auto tensor_desc = create_desc({1, 2*7*7, 14, 14}, ge::DT_FLOAT16);
  tensor_desc.SetFormat(ge::FORMAT_NHWC);
  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("rois", create_desc({1, 5, 16}, ge::DT_FLOAT16));
  op.SetAttr("output_dim", 2);
  op.SetAttr("group_size", 7);
  op.SetAttr("spatial_scale", 0.0625f);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}
