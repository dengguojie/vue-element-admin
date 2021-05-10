#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"
#include "array_ops.h"

class NthElementTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "NthElementTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "NthElementTest TearDown" << std::endl;
  }
};

TEST_F(NthElementTest, NthElementTest_infer_shape_x_failed) {
  ge::op::NthElement op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc1);
  op.UpdateInputDesc("n", tensor_desc1);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(NthElementTest, NthElementTest_infer_shape_n_failed) {
  ge::op::NthElement op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc1);
  op.UpdateInputDesc("n", tensor_desc1);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(NthElementTest, NthElementTest_infer_shape_n_failed2) {
  ge::op::NthElement op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc1);

  ge::TensorDesc const_desc1(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value1[1] = {4};
  auto const_op1 = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc1, (uint8_t *)const_value1, 1 * sizeof(int32_t)));
  op.set_input_n(const_op1);
  op.UpdateInputDesc("n", const_desc1);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

