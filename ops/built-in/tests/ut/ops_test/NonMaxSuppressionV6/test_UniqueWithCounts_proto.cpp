#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "sparse_ops.h"

class UniqueWithCountsTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "UniqueWithCounts SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "UniqueWithCounts TearDown" << std::endl;
  }
};

TEST_F(UniqueWithCountsTest, non_max_suppressio_test_case_1) {
  ge::op::UniqueWithCounts op;
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(UniqueWithCountsTest, non_max_suppressio_test_case_2) {
  ge::op::UniqueWithCounts op;
  op.UpdateInputDesc("x", create_desc_with_ori({-1}, ge::DT_INT32, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(UniqueWithCountsTest, non_max_suppressio_test_case_5) {
  ge::op::UniqueWithCounts op;
  op.UpdateInputDesc("x", create_desc_with_ori({-1}, ge::DT_INT32, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.SetAttr("out_idx", ge::DT_INT32);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(UniqueWithCountsTest, non_max_suppressio_test_case_3) {
  ge::op::UniqueWithCountsExt2 op;
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(UniqueWithCountsTest, non_max_suppressio_test_case_4) {
  ge::op::SparseSplit op;
  op.UpdateInputDesc("shape", create_desc_with_ori({-1}, ge::DT_INT64, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  op.UpdateInputDesc("values", create_desc_with_ori({-1}, ge::DT_INT64, ge::FORMAT_ND, {}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(UniqueWithCountsTest, non_max_suppressio_test_case_6) {
  ge::op::SparseReduceMax op;
  op.SetAttr("keep_dims", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}