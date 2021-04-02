#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "image_ops.h"

class NonMaxSuppressionV2Test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "NonMaxSuppressionV2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "NonMaxSuppressionV2 TearDown" << std::endl;
  }
};

TEST_F(NonMaxSuppressionV2Test, non_max_suppressio_test_case_1) {
  ge::op::NonMaxSuppressionV2 op;
  op.UpdateInputDesc("boxes", create_desc_with_ori({2, 6, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 6, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores", create_desc_with_ori({2}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 1, 6}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(NonMaxSuppressionV2Test, non_max_suppressio_test_case_2) {
  ge::op::NonMaxSuppressionV2 op;
  op.UpdateInputDesc("boxes", create_desc_with_ori({2, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 6}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores", create_desc_with_ori({2, 1, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 1, 6}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(NonMaxSuppressionV2Test, non_max_suppressio_test_case_3) {
  ge::op::NonMaxSuppressionV2 op;
  op.UpdateInputDesc("boxes", create_desc_with_ori({2, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 6}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores", create_desc_with_ori({2}, ge::DT_FLOAT, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("max_output_size", create_desc_with_ori({2}, ge::DT_INT32, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(NonMaxSuppressionV2Test, non_max_suppressio_test_case_4) {
  ge::op::NonMaxSuppressionV2 op;
  op.UpdateInputDesc("boxes", create_desc_with_ori({2, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 6}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores", create_desc_with_ori({2}, ge::DT_FLOAT, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("max_output_size", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(NonMaxSuppressionV2Test, non_max_suppressio_test_case_5) {
  ge::op::NonMaxSuppressionV2 op;
  op.UpdateInputDesc("boxes", create_desc_with_ori({2, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 6}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores", create_desc_with_ori({3}, ge::DT_FLOAT, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("max_output_size", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("iou_threshold", create_desc_with_ori({1}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(NonMaxSuppressionV2Test, non_max_suppressio_test_case_6) {
  ge::op::NonMaxSuppressionV2 op;
  op.UpdateInputDesc("boxes", create_desc_with_ori({2, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 6}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores", create_desc_with_ori({3}, ge::DT_FLOAT, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("max_output_size", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("iou_threshold", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(NonMaxSuppressionV2Test, non_max_suppressio_test_case_7) {
  ge::op::NonMaxSuppressionV2 op;
  op.UpdateInputDesc("boxes", create_desc_with_ori({2, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 6}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores", create_desc_with_ori({2}, ge::DT_FLOAT, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("max_output_size", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("iou_threshold", create_desc_with_ori({}, ge::DT_FLOAT, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}