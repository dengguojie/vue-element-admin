#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "stateless_random_ops.h"

class StatelessSampleDistortedBoundingBox : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StatelessSampleDistortedBoundingBox SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StatelessSampleDistortedBoundingBox TearDown" << std::endl;
  }
};

TEST_F(StatelessSampleDistortedBoundingBox, StatelessSampleDistortedBoundingBox_test01) {
  ge::op::StatelessSampleDistortedBoundingBox op;
  op.UpdateInputDesc("image_size", create_desc({3}, ge::DT_INT32));
  op.UpdateInputDesc("bounding_boxes", create_desc({2, 3, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("min_object_covered", create_desc({}, ge::DT_FLOAT));
  op.UpdateInputDesc("seed", create_desc({2}, ge::DT_INT32));
  
  auto ret = op.InferShapeAndType();
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  ge::TensorDesc begin = op.GetOutputDescByName("begin");
  ge::TensorDesc size = op.GetOutputDesc("size");
  ge::TensorDesc bboxes = op.GetOutputDesc("bboxes");

  EXPECT_EQ(begin.GetShape().GetDimNum(), 1);
  EXPECT_EQ(size.GetShape().GetDimNum(), 1);
  EXPECT_EQ(bboxes.GetShape().GetDimNum(), 3);

  EXPECT_EQ(begin.GetShape().GetDim(0), 3);
  EXPECT_EQ(size.GetShape().GetDim(0), 3);
  EXPECT_EQ(bboxes.GetShape().GetDim(0), 1);
  EXPECT_EQ(bboxes.GetShape().GetDim(1), 1);
  EXPECT_EQ(bboxes.GetShape().GetDim(2), 4);

  EXPECT_EQ(begin.GetDataType(), ge::DT_INT32);
  EXPECT_EQ(size.GetDataType(), ge::DT_INT32);
  EXPECT_EQ(bboxes.GetDataType(), ge::DT_FLOAT);
}

TEST_F(StatelessSampleDistortedBoundingBox, StatelessSampleDistortedBoundingBox_test02) {
  ge::op::StatelessSampleDistortedBoundingBox op;
  op.UpdateInputDesc("image_size", create_desc({3}, ge::DT_UINT8));
  op.UpdateInputDesc("bounding_boxes", create_desc({2, 3, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("min_object_covered", create_desc({}, ge::DT_FLOAT));
  op.UpdateInputDesc("seed", create_desc({2}, ge::DT_INT64));
  
  auto ret = op.InferShapeAndType();
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  ge::TensorDesc begin = op.GetOutputDescByName("begin");
  ge::TensorDesc size = op.GetOutputDesc("size");
  ge::TensorDesc bboxes = op.GetOutputDesc("bboxes");

  EXPECT_EQ(begin.GetShape().GetDimNum(), 1);
  EXPECT_EQ(size.GetShape().GetDimNum(), 1);
  EXPECT_EQ(bboxes.GetShape().GetDimNum(), 3);

  EXPECT_EQ(begin.GetShape().GetDim(0), 3);
  EXPECT_EQ(size.GetShape().GetDim(0), 3);
  EXPECT_EQ(bboxes.GetShape().GetDim(0), 1);
  EXPECT_EQ(bboxes.GetShape().GetDim(1), 1);
  EXPECT_EQ(bboxes.GetShape().GetDim(2), 4);

  EXPECT_EQ(begin.GetDataType(), ge::DT_UINT8);
  EXPECT_EQ(size.GetDataType(), ge::DT_UINT8);
  EXPECT_EQ(bboxes.GetDataType(), ge::DT_FLOAT);
}

TEST_F(StatelessSampleDistortedBoundingBox, StatelessSampleDistortedBoundingBox_infer_failed_1) {
  ge::op::StatelessSampleDistortedBoundingBox op;
  op.UpdateInputDesc("image_size", create_desc({1, 3}, ge::DT_UINT8));
  op.UpdateInputDesc("bounding_boxes", create_desc({2, 3, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("min_object_covered", create_desc({}, ge::DT_FLOAT));
  op.UpdateInputDesc("seed", create_desc({2}, ge::DT_INT64));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StatelessSampleDistortedBoundingBox, StatelessSampleDistortedBoundingBox_infer_failed_2) {
  ge::op::StatelessSampleDistortedBoundingBox op;
  op.UpdateInputDesc("image_size", create_desc({2}, ge::DT_UINT8));
  op.UpdateInputDesc("bounding_boxes", create_desc({2, 3, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("min_object_covered", create_desc({}, ge::DT_FLOAT));
  op.UpdateInputDesc("seed", create_desc({2}, ge::DT_INT64));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StatelessSampleDistortedBoundingBox, StatelessSampleDistortedBoundingBox_infer_failed_3) {
  ge::op::StatelessSampleDistortedBoundingBox op;
  op.UpdateInputDesc("image_size", create_desc({3}, ge::DT_UINT8));
  op.UpdateInputDesc("bounding_boxes", create_desc({2, 3, 5}, ge::DT_FLOAT));
  op.UpdateInputDesc("min_object_covered", create_desc({}, ge::DT_FLOAT));
  op.UpdateInputDesc("seed", create_desc({2}, ge::DT_INT64));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StatelessSampleDistortedBoundingBox, StatelessSampleDistortedBoundingBox_infer_failed_4) {
  ge::op::StatelessSampleDistortedBoundingBox op;
  op.UpdateInputDesc("image_size", create_desc({3}, ge::DT_UINT8));
  op.UpdateInputDesc("bounding_boxes", create_desc({3, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("min_object_covered", create_desc({}, ge::DT_FLOAT));
  op.UpdateInputDesc("seed", create_desc({2}, ge::DT_INT64));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StatelessSampleDistortedBoundingBox, StatelessSampleDistortedBoundingBox_infer_failed_5) {
  ge::op::StatelessSampleDistortedBoundingBox op;
  op.UpdateInputDesc("image_size", create_desc({3}, ge::DT_UINT8));
  op.UpdateInputDesc("bounding_boxes", create_desc({2, 3, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("min_object_covered", create_desc({1}, ge::DT_FLOAT));
  op.UpdateInputDesc("seed", create_desc({2}, ge::DT_INT64));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StatelessSampleDistortedBoundingBox, StatelessSampleDistortedBoundingBox_infer_failed_6) {
  ge::op::StatelessSampleDistortedBoundingBox op;
  op.UpdateInputDesc("image_size", create_desc({3}, ge::DT_UINT8));
  op.UpdateInputDesc("bounding_boxes", create_desc({2, 3, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("min_object_covered", create_desc({}, ge::DT_FLOAT));
  op.UpdateInputDesc("seed", create_desc({3}, ge::DT_INT64));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StatelessSampleDistortedBoundingBox, StatelessSampleDistortedBoundingBox_infer_failed_7) {
  ge::op::StatelessSampleDistortedBoundingBox op;
  op.UpdateInputDesc("image_size", create_desc({3}, ge::DT_UINT8));
  op.UpdateInputDesc("bounding_boxes", create_desc({2, 3, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("min_object_covered", create_desc({}, ge::DT_FLOAT));
  op.UpdateInputDesc("seed", create_desc({1, 2}, ge::DT_INT64));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(StatelessSampleDistortedBoundingBox, StatelessSampleDistortedBoundingBox_verify_failed_1) {
  ge::op::StatelessSampleDistortedBoundingBox op;
  op.UpdateInputDesc("image_size", create_desc({3}, ge::DT_UINT32));
  op.UpdateInputDesc("bounding_boxes", create_desc({2, 3, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("min_object_covered", create_desc({}, ge::DT_FLOAT));
  op.UpdateInputDesc("seed", create_desc({2}, ge::DT_INT64));
  
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(StatelessSampleDistortedBoundingBox, StatelessSampleDistortedBoundingBox_verify_failed_2) {
  ge::op::StatelessSampleDistortedBoundingBox op;
  op.UpdateInputDesc("image_size", create_desc({3}, ge::DT_UINT8));
  op.UpdateInputDesc("bounding_boxes", create_desc({2, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("min_object_covered", create_desc({}, ge::DT_FLOAT));
  op.UpdateInputDesc("seed", create_desc({2}, ge::DT_INT64));
  
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(StatelessSampleDistortedBoundingBox, StatelessSampleDistortedBoundingBox_verify_failed_3) {
  ge::op::StatelessSampleDistortedBoundingBox op;
  op.UpdateInputDesc("image_size", create_desc({3}, ge::DT_UINT8));
  op.UpdateInputDesc("bounding_boxes", create_desc({2, 3, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("min_object_covered", create_desc({}, ge::DT_FLOAT16));
  op.UpdateInputDesc("seed", create_desc({2}, ge::DT_INT64));
  
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(StatelessSampleDistortedBoundingBox, StatelessSampleDistortedBoundingBox_verify_failed_4) {
  ge::op::StatelessSampleDistortedBoundingBox op;
  op.UpdateInputDesc("image_size", create_desc({3}, ge::DT_UINT8));
  op.UpdateInputDesc("bounding_boxes", create_desc({2, 3, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("min_object_covered", create_desc({}, ge::DT_FLOAT));
  op.UpdateInputDesc("seed", create_desc({2}, ge::DT_UINT64));
  
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}