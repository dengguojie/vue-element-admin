#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "image_ops.h"

class SampleDistortedBoundingBoxExt2Test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SampleDistortedBoundingBoxExt2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SampleDistortedBoundingBoxExt2 TearDown" << std::endl;
  }
};

TEST_F(SampleDistortedBoundingBoxExt2Test, infershape_00) {
  ge::op::SampleDistortedBoundingBoxExt2 op;
  op.UpdateInputDesc("image_size", create_desc({}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SampleDistortedBoundingBoxExt2Test, infershape_01) {
  ge::op::SampleDistortedBoundingBoxExt2 op;
  op.UpdateInputDesc("image_size", create_desc({3}, ge::DT_INT32));
  op.UpdateInputDesc("bounding_boxes", create_desc({1,1}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SampleDistortedBoundingBoxExt2Test, infershape_02) {
  ge::op::SampleDistortedBoundingBoxExt2 op;
  op.UpdateInputDesc("image_size", create_desc({3}, ge::DT_INT32));
  op.UpdateInputDesc("bounding_boxes", create_desc({1,1,4}, ge::DT_FLOAT));
  op.UpdateInputDesc("min_object_covered", create_desc({1,1,4}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SampleDistortedBoundingBoxExt2Test, infershape_03) {
  ge::op::SampleDistortedBoundingBoxExt2 op;
  op.UpdateInputDesc("image_size", create_desc({3}, ge::DT_INT32));
  op.UpdateInputDesc("bounding_boxes", create_desc({1,1,2}, ge::DT_FLOAT));
  op.UpdateInputDesc("min_object_covered", create_desc({}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SampleDistortedBoundingBoxExt2Test, infershape_04) {
  ge::op::SampleDistortedBoundingBoxExt2 op;
  op.UpdateInputDesc("image_size", create_desc({3}, ge::DT_INT32));
  op.UpdateInputDesc("bounding_boxes", create_desc({1,1,4}, ge::DT_FLOAT));
  op.UpdateInputDesc("min_object_covered", create_desc({}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(SampleDistortedBoundingBoxExt2Test, infershape_05) {
  ge::op::SampleDistortedBoundingBoxExt2 op;
  op.UpdateInputDesc("image_size", create_desc({-1}, ge::DT_INT32));
  op.UpdateInputDesc("bounding_boxes", create_desc({1,1,4}, ge::DT_FLOAT));
  op.UpdateInputDesc("min_object_covered", create_desc({}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(SampleDistortedBoundingBoxExt2Test, infershape_06) {
  ge::op::SampleDistortedBoundingBoxExt2 op;
  op.UpdateInputDesc("image_size", create_desc({3}, ge::DT_INT32));
  op.UpdateInputDesc("bounding_boxes", create_desc({1,1,-1}, ge::DT_FLOAT));
  op.UpdateInputDesc("min_object_covered", create_desc({}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}