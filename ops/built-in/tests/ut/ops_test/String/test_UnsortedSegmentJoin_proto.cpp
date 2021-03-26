#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "string_ops.h"
#include "array_ops.h"

class UnsortedSegmentJoin : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "UnsortedSegmentJoin SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "UnsortedSegmentJoin TearDown" << std::endl;
  }
};

TEST_F(UnsortedSegmentJoin, UnsortedSegmentJoin_infershape_diff_test){
  ge::op::UnsortedSegmentJoin op;
  op.UpdateInputDesc("input", create_desc({2,3}, ge::DT_STRING));
  op.UpdateInputDesc("segment_ids", create_desc({2}, ge::DT_STRING));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("output");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_STRING);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(UnsortedSegmentJoin, UnsortedSegmentJoin_infershape_diff_test1){
  ge::op::UnsortedSegmentJoin op;
  op.UpdateInputDesc("input", create_desc({2,3}, ge::DT_STRING));
  op.UpdateInputDesc("segment_ids", create_desc({2}, ge::DT_INT32));

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {2};
  constTensor.SetData((uint8_t*)constData, 1* sizeof(int32_t));
  auto num_segments = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_num_segments(num_segments);
  auto desc = op.GetInputDesc("num_segments");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("num_segments", desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}