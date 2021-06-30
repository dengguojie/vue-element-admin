#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "random_ops.h"

class BernoulliTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "BernoulliTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BernoulliTest TearDown" << std::endl;
  }
};


// normal cases
TEST_F(BernoulliTest, bernoulli_infershape_test) {
  ge::op::Bernoulli op;
  op.UpdateInputDesc("x", create_desc_with_ori({33, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 5}, ge::FORMAT_ND));
  op.UpdateInputDesc("p", create_desc_with_ori({1}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {33, 5};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

// exception cases
TEST_F(BernoulliTest, bernoulli_infershape_invalid_test) {
  ge::op::Bernoulli op;
  op.UpdateInputDesc("x", create_desc_with_ori({33, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 5}, ge::FORMAT_ND));
  op.UpdateInputDesc("p", create_desc_with_ori({6}, ge::DT_FLOAT, ge::FORMAT_ND, {6}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
