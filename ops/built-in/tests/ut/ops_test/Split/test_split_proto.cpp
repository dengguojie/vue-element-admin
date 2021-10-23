#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "split_combination_ops.h"
#include "array_ops.h"


class SplitTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SplitTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SplitTest TearDown" << std::endl;
  }
};

TEST_F(SplitTest, split_test_infershape_diff_test_1) {
  ge::op::Split op;

  op.UpdateInputDesc("split_dim", create_desc_shape_range({1}, ge::DT_INT32, ge::FORMAT_ND, {1}, ge::FORMAT_ND,{{1,1}}));
  op.UpdateInputDesc("x", create_desc_shape_range({-1, 100}, ge::DT_INT32, ge::FORMAT_ND, {-1, 100}, ge::FORMAT_ND,{{2,100},{100,100}}));
  op.SetAttr("num_split", 2);

  op.InferShapeAndType();
}

TEST_F(SplitTest, split_test_infershape_diff_test_2) {
  ge::op::Split op;

  op.UpdateInputDesc("x", create_desc_shape_range({-1, 100}, ge::DT_INT32, ge::FORMAT_ND, {-1, 100}, ge::FORMAT_ND,{{2,100},{100,100}}));
  op.SetAttr("num_split", 2);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {0};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_split_dim(const0);

  op.InferShapeAndType();
}
