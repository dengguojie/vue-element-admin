#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "selection_ops.h"

class gather_v2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "gather_v2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "gather_v2 TearDown" << std::endl;
  }
};

TEST_F(gather_v2, gather_v2_infershape_diff_test_1) {
  ge::op::GatherV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({6, 7}, ge::DT_INT32, ge::FORMAT_ND, {6, 7}, ge::FORMAT_ND, {{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({9, 10, 2}, ge::DT_INT32, ge::FORMAT_ND, {9, 10, 2}, ge::FORMAT_ND,{{9,9},{10,10},{2,2}}));
  /*ge::op::Constant axis;
  //int32_t value = 1;
  axis.SetAttr("value", std::vector<int32_t>{1});
  op.set_input_axis(axis);*/
  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {1};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_axis(const0);


  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(2);
  op.UpdateInputDesc("x", tensor_x);

  ge::TensorDesc tensor_indices = op.GetInputDesc("indices");
  tensor_indices.SetRealDimCnt(3);
  op.UpdateInputDesc("indices", tensor_indices);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {6, 9, 10, 2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{6,6},{9,9},{10,10},{2,2}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
  //delete []constData;
}

// TODO fix me
//TEST_F(gather_v2, gather_v2_infershape_diff_test_2) {
//  ge::op::GatherV2 op;
//  op.UpdateInputDesc("x", create_desc_with_ori({2, 2, 2}, ge::DT_INT32, ge::FORMAT_ND, {2, 2, 2}, ge::FORMAT_ND));
//  op.UpdateInputDesc("indices", create_desc_with_ori({2, 2, 2}, ge::DT_INT32, ge::FORMAT_ND, {2, 2, 2}, ge::FORMAT_ND));
//
//  ge::Tensor constTensor;
//  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
//  constDesc.SetSize(1 * sizeof(int32_t));
//  constTensor.SetTensorDesc(constDesc);
//  int32_t constData[1] = {2};
//  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
//  auto const0 = ge::op::Constant().set_attr_value(constTensor);
//  op.set_input_axis(const0);
//
//  ge::TensorDesc tensor_x = op.GetInputDesc("x");
//  tensor_x.SetRealDimCnt(3);
//  op.UpdateInputDesc("x", tensor_x);
//
//  ge::TensorDesc tensor_indices = op.GetInputDesc("indices");
//  tensor_indices.SetRealDimCnt(3);
//  op.UpdateInputDesc("indices", tensor_indices);
//
//  auto ret = op.InferShapeAndType();
//  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
//  auto output_desc = op.GetOutputDesc("y");
//  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
//  std::vector<int64_t> expected_output_shape = {2, 2, 2, 2, 2};
//  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
//  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {};
//  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
//  output_desc.GetShapeRange(output_shape_range);
//  EXPECT_EQ(output_shape_range, expected_output_shape_range);
//}

TEST_F(gather_v2, gather_v2_infershape_diff_test_5) {
  ge::op::GatherV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {-2}, ge::FORMAT_ND, {{1,3},{4,5},{9,10}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({10, 2}, ge::DT_INT32, ge::FORMAT_ND, {10, 2}, ge::FORMAT_ND, {{10,10},{2,2}}));

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {2};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_axis(const0);

  /*ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(3);
  op.UpdateInputDesc("x", tensor_x);*/

  ge::TensorDesc tensor_indices = op.GetInputDesc("indices");
  tensor_indices.SetRealDimCnt(3);
  op.UpdateInputDesc("indices", tensor_indices);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(gather_v2, gather_v2_infershape_diff_test_6) {
  ge::op::GatherV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 7}, ge::FORMAT_ND,{{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {-2}, ge::FORMAT_ND,{{1,100},{2,10}}));

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {0};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_axis(const0);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  /*ge::TensorDesc tensor_indices = op.GetInputDesc("indices");
  tensor_indices.SetRealDimCnt(3);
  op.UpdateInputDesc("indices", tensor_indices);*/

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(gather_v2, gather_v2_infershape_diff_test_7) {
  ge::op::GatherV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 7}, ge::FORMAT_ND,{{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({-1, 2}, ge::DT_INT32, ge::FORMAT_ND, {-1, 2}, ge::FORMAT_ND, {{3,4},{2,2}}));

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {2};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_axis(const0);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  ge::TensorDesc tensor_indices = op.GetInputDesc("indices");
  tensor_indices.SetRealDimCnt(2);
  op.UpdateInputDesc("indices", tensor_indices);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {3,4,-1,2,6,7};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{3,3},{4,4},{3,4},{2,2},{6,6},{7,7}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(gather_v2, gather_v2_infershape_diff_test_8) {
  ge::op::GatherV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, -1, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, -1, 5, 6, 7}, ge::FORMAT_ND,{{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({3, -1}, ge::DT_INT32, ge::FORMAT_ND, {3, -1}, ge::FORMAT_ND,{{3,3},{1,10}}));

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {0};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_axis(const0);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  ge::TensorDesc tensor_indices = op.GetInputDesc("indices");
  tensor_indices.SetRealDimCnt(2);
  op.UpdateInputDesc("indices", tensor_indices);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {3,-1,-1,5,6,7};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{3,3},{1,10},{4,4},{5,5},{6,6},{7,7}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}
// TODO fix me run failed
//TEST_F(gather_v2, gather_v2_infershape_diff_test_9) {
//  ge::op::GatherV2 op;
//  op.UpdateInputDesc("x", create_desc_shape_range({3, -1, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, -1, 5, 6, 7}, ge::FORMAT_ND,{{3,3},{4,4},{5,5},{6,6},{7,7}}));
//  op.UpdateInputDesc("indices", create_desc_shape_range({3, -1}, ge::DT_INT32, ge::FORMAT_ND, {3, -1}, ge::FORMAT_ND,{{3,3},{1,10}}));
//
//  auto data0 = ge::op::Data().set_attr_index(0);
//  op.set_input_axis(data0);
//
//  ge::TensorDesc tensor_x = op.GetInputDesc("x");
//  tensor_x.SetRealDimCnt(5);
//  op.UpdateInputDesc("x", tensor_x);
//
//  ge::TensorDesc tensor_indices = op.GetInputDesc("indices");
//  tensor_indices.SetRealDimCnt(2);
//  op.UpdateInputDesc("indices", tensor_indices);
//
//  auto ret = op.InferShapeAndType();
//  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
//  auto output_desc = op.GetOutputDesc("y");
//  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
//  std::vector<int64_t> expected_output_shape = {-1,-1,-1,-1,-1,-1};
//  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
//  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1,-1},{3,3},{1,10},{1,-1}};
//  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
//  output_desc.GetShapeRange(output_shape_range);
//  EXPECT_EQ(output_shape_range, expected_output_shape_range);
//}

TEST_F(gather_v2, gather_v2_infershape_diff_test_10) {
  ge::op::GatherV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 7}, ge::FORMAT_ND,{{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {-2}, ge::FORMAT_ND,{{1,100},{2,10}}));

  auto data0 = ge::op::Data().set_attr_index(0);
  op.set_input_axis(data0);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  /*ge::TensorDesc tensor_indices = op.GetInputDesc("indices");
  tensor_indices.SetRealDimCnt(3);
  op.UpdateInputDesc("indices", tensor_indices);*/

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(gather_v2, gather_v2_infershape_diff_test_11) {
  ge::op::GatherV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 7}, ge::FORMAT_ND,{{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {-1}, ge::FORMAT_ND,{{1,-1}}));

  auto data0 = ge::op::Data().set_attr_index(0);
  op.set_input_axis(data0);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  /*ge::TensorDesc tensor_indices = op.GetInputDesc("indices");
  tensor_indices.SetRealDimCnt(3);
  op.UpdateInputDesc("indices", tensor_indices);*/

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1,-1},{1,-1},{1,-1},{1,-1},{1,-1}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(gather_v2, gather_v2_infershape_with_batch_dims_1) {
  ge::op::GatherV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND,
                                                  {3, 4, 5, 6, 7}, ge::FORMAT_ND,
                                                  {{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("indices", create_desc_shape_range({3,4,32}, ge::DT_INT32, ge::FORMAT_ND,
                                                        {3,4,32}, ge::FORMAT_ND,{{3,3}, {4,4},{32,32}}));

  auto data0 = ge::op::Data().set_attr_index(2);
  op.set_input_axis(data0);

  op.SetAttr("batch_dims", 2);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  /*ge::TensorDesc tensor_indices = op.GetInputDesc("indices");
  tensor_indices.SetRealDimCnt(3);
  op.UpdateInputDesc("indices", tensor_indices);*/

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{{3,32},{3,32},{3,32},{3,32},{3,32}}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}
