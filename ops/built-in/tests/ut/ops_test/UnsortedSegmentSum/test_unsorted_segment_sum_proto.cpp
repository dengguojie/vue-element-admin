#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "selection_ops.h"

class unsorted_segment_sum : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "unsorted_segment_sum SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "unsorted_segment_sum TearDown" << std::endl;
  }
};

TEST_F(unsorted_segment_sum, unsorted_segment_sum_infershape_diff_test_1) {
  ge::op::UnsortedSegmentSum op;
  op.UpdateInputDesc("x", create_desc_shape_range({9, 10, 2, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {9, 10, 2, 6, 7}, ge::FORMAT_ND,{{9,9},{10,10},{2,2},{6,6},{7,7}}));
  op.UpdateInputDesc("segment_ids", create_desc_shape_range({9, 10, 2}, ge::DT_INT32, ge::FORMAT_ND, {9, 10, 2}, ge::FORMAT_ND,{{9,9},{10,10},{2,2}}));


  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {1};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_num_segments(const0);


  ge::TensorDesc tensor_num_segments = op.GetInputDesc("num_segments");
  tensor_num_segments.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("num_segments", tensor_num_segments);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {1, 6, 7};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1,1},{6,6},{7,7}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(unsorted_segment_sum, unsorted_segment_sum_infershape_diff_test_2) {
  ge::op::UnsortedSegmentSum op;
  op.UpdateInputDesc("x", create_desc_shape_range({2, 2, 2, 1}, ge::DT_INT32, ge::FORMAT_ND, {2, 2, 2, 1}, ge::FORMAT_ND,{{2,2},{2,2},{2,2},{1,1}}));
  op.UpdateInputDesc("segment_ids", create_desc_shape_range({2}, ge::DT_INT32, ge::FORMAT_ND, {2}, ge::FORMAT_ND,{{2,2}}));

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {3};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_num_segments(const0);

  ge::TensorDesc tensor_num_segments = op.GetInputDesc("num_segments");
  tensor_num_segments.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("num_segments", tensor_num_segments);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {3, 2, 2, 1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{3,3}, {2,2}, {2,2}, {1,1}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(unsorted_segment_sum, unsorted_segment_sum_infershape_diff_test_3) {
  ge::op::UnsortedSegmentSum op;
  op.UpdateInputDesc("x", create_desc_shape_range({2}, ge::DT_INT32, ge::FORMAT_ND, {2}, ge::FORMAT_ND,{{2,2}}));
  op.UpdateInputDesc("segment_ids", create_desc_shape_range({2}, ge::DT_INT32, ge::FORMAT_ND, {2}, ge::FORMAT_ND,{{2,2}}));

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {3};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_num_segments(const0);

  ge::TensorDesc tensor_num_segments = op.GetInputDesc("num_segments");
  tensor_num_segments.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("num_segments", tensor_num_segments);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {3};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{3,3}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(unsorted_segment_sum, unsorted_segment_sum_infershape_diff_test_5) {
  ge::op::UnsortedSegmentSum op;
  op.UpdateInputDesc("x", create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {-2}, ge::FORMAT_ND, {{10,10},{2,2},{9,10}}));
  op.UpdateInputDesc("segment_ids", create_desc_shape_range({10, 2}, ge::DT_INT32, ge::FORMAT_ND, {10, 2}, ge::FORMAT_ND, {{10,10},{2,2}}));

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {2};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_num_segments(const0);



  ge::TensorDesc tensor_num_segments = op.GetInputDesc("num_segments");
  tensor_num_segments.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("num_segments", tensor_num_segments);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(unsorted_segment_sum, unsorted_segment_sum_infershape_diff_test_6) {
  ge::op::UnsortedSegmentSum op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 7}, ge::FORMAT_ND,{{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("segment_ids", create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {-2}, ge::FORMAT_ND,{{1,100},{2,10}}));

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {1};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_num_segments(const0);

  ge::TensorDesc tensor_num_segments = op.GetInputDesc("num_segments");
  tensor_num_segments.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("num_segments", tensor_num_segments);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(unsorted_segment_sum, unsorted_segment_sum_infershape_diff_test_7) {
  ge::op::UnsortedSegmentSum op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 7}, ge::FORMAT_ND,{{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("segment_ids", create_desc_shape_range({-1, 4}, ge::DT_INT32, ge::FORMAT_ND, {-1, 4}, ge::FORMAT_ND, {{3,4},{4,4}}));

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {2};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_num_segments(const0);

  ge::TensorDesc tensor_num_segments = op.GetInputDesc("num_segments");
  tensor_num_segments.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("num_segments", tensor_num_segments);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {2,5,6,7};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{2,2},{5,5},{6,6},{7,7}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(unsorted_segment_sum, unsorted_segment_sum_infershape_diff_test_8) {
  ge::op::UnsortedSegmentSum op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, -1, 5, -1, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, -1, 5, -1, 7}, ge::FORMAT_ND,{{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("segment_ids", create_desc_shape_range({3, -1}, ge::DT_INT32, ge::FORMAT_ND, {3, -1}, ge::FORMAT_ND,{{3,3},{1,10}}));

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {1};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_num_segments(const0);

  ge::TensorDesc tensor_num_segments = op.GetInputDesc("num_segments");
  tensor_num_segments.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("num_segments", tensor_num_segments);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {1,5,-1,7};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1,1},{5,5},{6,6},{7,7}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(unsorted_segment_sum, unsorted_segment_sum_infershape_diff_test_9) {
  ge::op::UnsortedSegmentSum op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, -1, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, -1, 5, 6, 7}, ge::FORMAT_ND,{{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("segment_ids", create_desc_shape_range({3, -1}, ge::DT_INT32, ge::FORMAT_ND, {3, -1}, ge::FORMAT_ND,{{3,3},{1,10}}));

  auto data0 = ge::op::Data().set_attr_index(0);
  op.set_input_num_segments(data0);

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(5);
  op.UpdateInputDesc("x", tensor_x);

  ge::TensorDesc tensor_num_segments = op.GetInputDesc("num_segments");
  tensor_num_segments.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("num_segments", tensor_num_segments);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-1,5,6,7};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{0,-1},{5,5},{6,6},{7,7}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(unsorted_segment_sum, unsorted_segment_sum_infershape_diff_test_10) {
  ge::op::UnsortedSegmentSum op;
  op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, 7}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, 7}, ge::FORMAT_ND,{{3,3},{4,4},{5,5},{6,6},{7,7}}));
  op.UpdateInputDesc("segment_ids", create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {-2}, ge::FORMAT_ND,{{1,100},{2,10}}));

  auto data0 = ge::op::Data().set_attr_index(0);
  op.set_input_num_segments(data0);

  ge::TensorDesc tensor_num_segments = op.GetInputDesc("num_segments");
  tensor_num_segments.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("num_segments", tensor_num_segments);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
