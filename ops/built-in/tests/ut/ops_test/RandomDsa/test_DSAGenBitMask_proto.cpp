#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "randomdsa_ops.h"
#include "array_ops.h"
#include "utils/attr_utils.h"
#include "utils/op_desc_utils.h"

class dsaGenBitMask : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DSAGenBitMask Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DSAGenBitMask Proto Test TearDown" << std::endl;
  }
};

TEST_F(dsaGenBitMask, dsaGenBitMask_infershape_diff_test){
  ge::op::DSAGenBitMask op;
  op.UpdateInputDesc("count", create_desc({3}, ge::DT_INT64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(dsaGenBitMask, dsaGenBitMask_infershape_diff_test_0){
  ge::op::DSAGenBitMask op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 100}};
  auto tensor_desc = create_desc_shape_range({-1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {64},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("count", tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("out");
  std::vector<int64_t> expected_output_shape = {-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
}

TEST_F(dsaGenBitMask, dsaGenBitMask_infershape_diff_test_1){
  ge::op::DSAGenBitMask op;

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape({4}), ge::FORMAT_NHWC, ge::DT_INT64);
  constDesc.SetSize(4 * sizeof(int64_t));
  constTensor.SetTensorDesc(constDesc);
  int64_t constData[4] = {2, 3, 4, 5};
  constTensor.SetData((uint8_t*)constData, 4 * sizeof(int64_t));
  auto output_size = ge::op::Constant().set_attr_value(constTensor);

  op.set_input_count(output_size);
  auto desc = op.GetInputDesc("count");
  desc.SetDataType(ge::DT_INT64);
  op.UpdateInputDesc("count", desc);

  auto probDesc = op.GetInputDesc("dropout");
  probDesc.SetDataType(ge::DT_FLOAT);
  probDesc.SetShape(ge::Shape());
  op.UpdateInputDesc("dropout", probDesc);  

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(dsaGenBitMask, dsaGenBitMask_infershape_unconstData_test){
  ge::op::DSAGenBitMask op;
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto input_sizes_desc = op_desc->MutableInputDesc("count");
  std::vector<std::pair<int64_t, int64_t>> value_range = {{1, 2}, {32, 32}, {4, 5}, {4, 5}};
  input_sizes_desc->SetValueRange(value_range);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}


TEST_F(dsaGenBitMask, dsaGenBitMask_infershape_prob_rank_err_1){
  ge::op::DSAGenBitMask op;
  op.UpdateInputDesc("count", create_desc({3,4}, ge::DT_INT64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(dsaGenBitMask, dsaGenBitMask_003) {
  ge::op::DSAGenBitMask op;
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{30, 30}};
  auto input_desc = create_desc_shape_range({2}, ge::DT_INT64, ge::FORMAT_NCHW,
                                            {2}, ge::FORMAT_NCHW, range_x1);
  op.UpdateInputDesc("count", input_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("out");
  std::vector<int64_t> expect_output_shape = {2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expect_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  std::vector<std::pair<int64_t,int64_t>> expected_range = {};
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

TEST_F(dsaGenBitMask, dsaGenBitMask_004) {
  ge::op::DSAGenBitMask op;
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{30, 30}};
  auto input_desc = create_desc_shape_range({-1}, ge::DT_INT64, ge::FORMAT_NCHW,
                                            {-1}, ge::FORMAT_NCHW, range_x1);
  op.UpdateInputDesc("count", input_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("out");
  std::vector<int64_t> expect_output_shape = {-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expect_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{30,30}};
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

