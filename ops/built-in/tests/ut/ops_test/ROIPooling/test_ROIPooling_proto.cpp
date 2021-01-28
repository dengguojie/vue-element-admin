#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

class ROIPoolingProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ROIPooling Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ROIPooling Proto Test TearDown" << std::endl;
  }
};

TEST_F(ROIPoolingProtoTest, roi_pooling_infershape_test_1) {
  ge::op::ROIPooling op;

  op.UpdateInputDesc("x", create_desc({1, 128, 24, 78}, ge::DT_FLOAT16));
  op.UpdateInputDesc("rois", create_desc({1, 5, 16}, ge::DT_FLOAT16));
  op.SetAttr("pooled_h", 6);
  op.SetAttr("pooled_w", 6);
  op.SetAttr("spatial_scale_h", 0.0625f);
  op.SetAttr("spatial_scale_w", 0.0625f);

  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(ROIPoolingProtoTest, roi_pooling_verify_test_1) {
  ge::op::ROIPooling op;

  op.UpdateInputDesc("x", create_desc({1, 128, 24, 78}, ge::DT_FLOAT16));
  op.UpdateInputDesc("rois", create_desc({1, 5, 17}, ge::DT_FLOAT16));
  op.SetAttr("pooled_h", 6);
  op.SetAttr("pooled_w", 6);
  op.SetAttr("spatial_scale_h", 0.0625f);
  op.SetAttr("spatial_scale_w", 0.0625f);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}
