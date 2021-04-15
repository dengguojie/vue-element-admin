#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "ragged_math_ops.h"

class RaggedRangeTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RaggedRange SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RaggedRange TearDown" << std::endl;
  }
};

TEST_F(RaggedRangeTest, non_max_suppressio_test_case_1) {
  ge::op::RaggedRange op;
  op.UpdateInputDesc("starts", create_desc_with_ori({2, 6}, ge::DT_INT32, ge::FORMAT_ND, {2, 6}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RaggedRangeTest, non_max_suppressio_test_case_2) {
  ge::op::RaggedRange op;
  op.UpdateInputDesc("starts", create_desc_with_ori({2}, ge::DT_INT32, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("limits", create_desc_with_ori({2, 6}, ge::DT_INT32, ge::FORMAT_ND, {2, 6}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RaggedRangeTest, non_max_suppressio_test_case_3) {
  ge::op::RaggedRange op;
  op.UpdateInputDesc("starts", create_desc_with_ori({2}, ge::DT_INT32, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("limits", create_desc_with_ori({2}, ge::DT_INT32, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("deltas", create_desc_with_ori({2,6}, ge::DT_INT32, ge::FORMAT_ND, {2,6}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RaggedRangeTest, non_max_suppressio_test_case_4) {
  ge::op::RaggedRange op;
  op.UpdateInputDesc("starts", create_desc_with_ori({2}, ge::DT_INT32, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("limits", create_desc_with_ori({2}, ge::DT_INT32, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("deltas", create_desc_with_ori({2}, ge::DT_INT32, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RaggedRangeTest, non_max_suppressio_test_case_5) {
  ge::op::RaggedRange op;
  op.UpdateInputDesc("starts", create_desc_with_ori({-1}, ge::DT_INT32, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("limits", create_desc_with_ori({2}, ge::DT_INT32, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("deltas", create_desc_with_ori({2}, ge::DT_INT32, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RaggedRangeTest, non_max_suppressio_test_case_6) {
  ge::op::RaggedRange op;
  op.UpdateInputDesc("starts", create_desc_with_ori({-1}, ge::DT_INT32, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("limits", create_desc_with_ori({-1}, ge::DT_INT32, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("deltas", create_desc_with_ori({2}, ge::DT_INT32, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
TEST_F(RaggedRangeTest, non_max_suppressio_test_case_7) {
  ge::op::RaggedRange op;
  op.UpdateInputDesc("starts", create_desc_with_ori({-1}, ge::DT_INT32, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("limits", create_desc_with_ori({-1}, ge::DT_INT32, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("deltas", create_desc_with_ori({-1}, ge::DT_INT32, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RaggedRangeTest, non_max_suppressio_test_case_8) {
  ge::op::RaggedRange op;
  op.UpdateInputDesc("starts", create_desc_with_ori({-1}, ge::DT_INT32, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("limits", create_desc_with_ori({-1}, ge::DT_INT32, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("deltas", create_desc_with_ori({-1}, ge::DT_INT32, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.SetAttr("Tsplits", ge::DT_INT32);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
