#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "math_ops.h"

class SparseSegmentSumGrad : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SparseSegmentSumGrad SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SparseSegmentSumGrad TearDown" << std::endl;
  }
};

TEST_F(SparseSegmentSumGrad, sparse_segment_sum_grad_infershape_diff_test_1) {
  ge::op::SparseSegmentSumGrad op;
  op.UpdateInputDesc("grad", create_desc_with_ori({1, 2, 3}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 2, 3}, ge::FORMAT_ND));
  op.UpdateInputDesc("indices", create_desc_with_ori({5}, ge::DT_FLOAT, ge::FORMAT_ND, {5}, ge::FORMAT_ND));
  op.UpdateInputDesc("segment_ids", create_desc_with_ori({5}, ge::DT_FLOAT, ge::FORMAT_ND, {5}, ge::FORMAT_ND));

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {6};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_output_dim0(const0);

  ge::TensorDesc tensor_output_dim0 = op.GetInputDescByName("output_dim0");
  tensor_output_dim0.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("output_dim0", tensor_output_dim0);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("output");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {6, 2, 3};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(SparseSegmentSumGrad, sparse_segment_sum_grad_infershape_diff_test_2) {
  ge::op::SparseSegmentSumGrad op;
  op.UpdateInputDesc("grad", create_desc_with_ori({6, 2, 3}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 2, 3}, ge::FORMAT_ND));
  op.UpdateInputDesc("indices", create_desc_with_ori({5}, ge::DT_FLOAT, ge::FORMAT_ND, {5}, ge::FORMAT_ND));
  op.UpdateInputDesc("segment_ids", create_desc_with_ori({5}, ge::DT_FLOAT, ge::FORMAT_ND, {5}, ge::FORMAT_ND));

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {1};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_output_dim0(const0);

  ge::TensorDesc tensor_output_dim0 = op.GetInputDescByName("output_dim0");
  tensor_output_dim0.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("output_dim0", tensor_output_dim0);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("output");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {1, 2, 3};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(SparseSegmentSumGrad, sparse_segment_sum_grad_infershape_diff_test_3) {
  ge::op::SparseSegmentSumGrad op;
  op.UpdateInputDesc("grad", create_desc_with_ori({1024, 80}, ge::DT_FLOAT, ge::FORMAT_ND, {1024, 80}, ge::FORMAT_ND));
  op.UpdateInputDesc("indices", create_desc_with_ori({512}, ge::DT_FLOAT, ge::FORMAT_ND, {512}, ge::FORMAT_ND));
  op.UpdateInputDesc("segment_ids", create_desc_with_ori({512}, ge::DT_FLOAT, ge::FORMAT_ND, {512}, ge::FORMAT_ND));

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {6};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_output_dim0(const0);

  ge::TensorDesc tensor_output_dim0 = op.GetInputDescByName("output_dim0");
  tensor_output_dim0.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("output_dim0", tensor_output_dim0);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("output");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {6, 80};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(SparseSegmentSumGrad, sparse_segment_sum_grad_infershape_diff_test_4) {
  ge::op::SparseSegmentSumGrad op;
  op.UpdateInputDesc("grad", create_desc_with_ori({46, 19994}, ge::DT_FLOAT, ge::FORMAT_ND, {46, 19994}, ge::FORMAT_ND));
  op.UpdateInputDesc("indices", create_desc_with_ori({50}, ge::DT_FLOAT, ge::FORMAT_ND, {50}, ge::FORMAT_ND));
  op.UpdateInputDesc("segment_ids", create_desc_with_ori({50}, ge::DT_FLOAT, ge::FORMAT_ND, {50}, ge::FORMAT_ND));

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {300};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_output_dim0(const0);

  ge::TensorDesc tensor_output_dim0 = op.GetInputDescByName("output_dim0");
  tensor_output_dim0.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("output_dim0", tensor_output_dim0);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDescByName("output");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {300, 19994};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}