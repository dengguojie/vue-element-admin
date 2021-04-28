#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "selection_ops.h"

class upper_bound : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "upper_bound SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "upper_bound TearDown" << std::endl;
  }
};

TEST_F(upper_bound, upper_bound_infershape_test_1) {
  ge::op::UpperBound op;
  op.UpdateInputDesc("sorted_x", create_desc({1, 2}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({1, 2}, ge::DT_INT64));
  op.SetAttr("out_type", ge::DT_INT64);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(upper_bound, upper_bound_infershape_test_2) {
  ge::op::UpperBound op;
  op.UpdateInputDesc("sorted_x", create_desc({1, 2}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({1, 2}, ge::DT_INT64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(upper_bound, upper_bound_infershape_test_3) {
  ge::op::UpperBound op;
  op.UpdateInputDesc("sorted_x", create_desc({1, 2, 3}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({1, 2}, ge::DT_INT64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(upper_bound, upper_bound_infershape_test_4) {
  ge::op::UpperBound op;
  op.UpdateInputDesc("sorted_x", create_desc({1, 2}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({1, 2, 3}, ge::DT_INT64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}