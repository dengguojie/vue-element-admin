#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "sparse_ops.h"

class sparse_reduce_max : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "sparse_reduce_max SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "sparse_reduce_max TearDown" << std::endl;
  }
};

TEST_F(sparse_reduce_max, sparse_reduce_max_infer_shape_01) {
  ge::op::SparseReduceMax op;
  op.UpdateInputDesc("x_indices", create_desc({3,2}, ge::DT_INT64));
  op.UpdateInputDesc("x_values", create_desc({3}, ge::DT_FLOAT));
  op.UpdateInputDesc("x_shape", create_desc({-2}, ge::DT_INT64));
  op.UpdateInputDesc("reduction_axes", create_desc({-2}, ge::DT_INT32));
  op.SetAttr("keep_dims", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
TEST_F(sparse_reduce_max, sparse_reduce_max_infer_shape_02) {
  ge::op::SparseReduceMax op;
  op.UpdateInputDesc("x_indices", create_desc({3,2}, ge::DT_INT64));
  op.UpdateInputDesc("x_values", create_desc({3}, ge::DT_FLOAT));
  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT64);
  constDesc.SetSize(2 * sizeof(int64_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[2] = {3,4};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int64_t));
  auto shape_dim = ge::op::Constant().set_attr_value(constTensor);
  
  op.set_input_x_shape(shape_dim);
  auto desc = op.GetInputDesc("x_shape");
  desc.SetDataType(ge::DT_INT64);
  op.UpdateInputDesc("x_shape", desc);

  ge::Tensor constTensor1;
  ge::TensorDesc constDesc1(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
  constDesc1.SetSize(1 * sizeof(int64_t));
  constTensor1.SetTensorDesc(constDesc);
  int32_t constData1[1] = {1};
  constTensor1.SetData((uint8_t*)constData1, 1 * sizeof(int32_t));
  auto reduction_dim = ge::op::Constant().set_attr_value(constTensor1);
  
  op.set_input_reduction_axes(reduction_dim);
  desc = op.GetInputDesc("reduction_axes");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("reduction_axes", desc);

  op.SetAttr("keep_dims", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
}