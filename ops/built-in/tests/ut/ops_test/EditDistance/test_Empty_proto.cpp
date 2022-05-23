#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"

class EmptyUT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "EmptyUT Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "EmptyUT Proto Test TearDown" << std::endl;
  }
};

TEST_F(EmptyUT, Empty_success) {
  ge::op::Empty op("empty");
  op.UpdateInputDesc("shape", create_desc({1, 3, 2, 5}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output = op.GetOutputDesc("y");
  EXPECT_EQ(output.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> output_shape = {-1};
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);
}

TEST_F(EmptyUT, Empty_success_unknow_rank){
  ge::op::Empty op("empty");
  op.UpdateInputDesc("shape", create_desc({-2}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output = op.GetOutputDesc("y");
  std::vector<int64_t> output_shape = {-1};
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);
}

TEST_F(EmptyUT, Empty_success_unknow_dim){
  ge::op::Empty op("empty");
  op.UpdateInputDesc("shape", create_desc({-1}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output = op.GetOutputDesc("y");
  std::vector<int64_t> output_shape = {-1};
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);
}
