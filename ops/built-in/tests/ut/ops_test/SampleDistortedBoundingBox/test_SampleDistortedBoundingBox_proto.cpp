#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "image_ops.h"

class sampledistortedboudingbox : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SampleDistortedBoundingBox SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SampleDistortedBoundingBox TearDown" << std::endl;
  }
};

TEST_F(sampledistortedboudingbox, sampledistortedboudingbox_infershape_test) {
  ge::op::SampleDistortedBoundingBox op;
  std::vector<float> ratio_list = {0.75, 1.33};
  std::vector<float> area_list = {0.05, 1.0};
  op.UpdateInputDesc("image_size", create_desc({3}, ge::DT_INT32));
  op.UpdateInputDesc("bounding_boxes", create_desc({1,1,4}, ge::DT_FLOAT));
  op.SetAttr("seed", 0);
  op.SetAttr("seed2", 0);
  op.SetAttr("min_object_covered", 0.1f);
  op.SetAttr("aspect_ratio_range", ratio_list);
  op.SetAttr("area_range", area_list);
  op.SetAttr("max_attempts", 100);
  op.SetAttr("use_image_if_no_bounding_boxes", false);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("begin");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {3};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  auto output_desc1 = op.GetOutputDesc("size");
  EXPECT_EQ(output_desc1.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape1 = {3};
  EXPECT_EQ(output_desc1.GetShape().GetDims(), expected_output_shape1);

  auto output_desc2 = op.GetOutputDesc("bboxes");
  EXPECT_EQ(output_desc2.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape2 = {1,1,4};
  EXPECT_EQ(output_desc2.GetShape().GetDims(), expected_output_shape2);
}

TEST_F(sampledistortedboudingbox, infershape_00) {
  ge::op::SampleDistortedBoundingBox op;
  op.UpdateInputDesc("image_size", create_desc({}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(sampledistortedboudingbox, infershape_01) {
  ge::op::SampleDistortedBoundingBox op;
  op.UpdateInputDesc("image_size", create_desc({3}, ge::DT_INT32));
  op.UpdateInputDesc("bounding_boxes", create_desc({1,1}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(sampledistortedboudingbox, infershape_02) {
  ge::op::SampleDistortedBoundingBox op;
  op.UpdateInputDesc("image_size", create_desc({3}, ge::DT_INT32));
  op.UpdateInputDesc("bounding_boxes", create_desc({1,1,2}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(sampledistortedboudingbox, infershape_03) {
  ge::op::SampleDistortedBoundingBox op;
  op.UpdateInputDesc("image_size", create_desc({1}, ge::DT_INT32));
  op.UpdateInputDesc("bounding_boxes", create_desc({1,1,4}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(sampledistortedboudingbox, infershape_04) {
  ge::op::SampleDistortedBoundingBox op;
  op.UpdateInputDesc("image_size", create_desc({3}, ge::DT_INT32));
  op.UpdateInputDesc("bounding_boxes", create_desc({1,1,4}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
