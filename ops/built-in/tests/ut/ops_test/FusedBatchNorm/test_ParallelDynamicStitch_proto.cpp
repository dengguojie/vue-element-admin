#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"
#include "array_ops.h"

class ParallelDynamicStitch : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ParallelDynamicStitch SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ParallelDynamicStitch TearDown" << std::endl;
  }
};

TEST_F(ParallelDynamicStitch, ParallelDynamicStitch_infer_shape_1) {
  ge::op::ParallelDynamicStitch op; 
  op.SetAttr("N", 2);  
  op.UpdateInputDesc("indices0",  create_desc({3}, ge::DT_INT32));
  op.UpdateInputDesc("indices1",  create_desc({2}, ge::DT_INT64));                                                                                                                                                                               
  op.UpdateInputDesc("x0", create_desc({3}, ge::DT_FLOAT));
  op.UpdateInputDesc("x1", create_desc({2},ge::DT_FLOAT));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {0};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(ParallelDynamicStitch, ParallelDynamicStitch_infer_shape_2) {
  ge::op::ParallelDynamicStitch op; 
  op.SetAttr("N", 1);  
  // op.UpdateInputDesc("indices0",  create_desc({3}, ge::DT_INT32));                                                                                                                                                                              
  op.UpdateInputDesc("x0", create_desc({3}, ge::DT_FLOAT));

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape({3}), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(3 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);

  int64_t constData[3] = {2, 3, 4};
  constTensor.SetData((uint8_t*)constData, 3 * sizeof(int32_t));
  auto const_0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_dynamic_input_indices(0,const_0);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ParallelDynamicStitch, ParallelDynamicStitch_infer_shape_3) {
  ge::op::ParallelDynamicStitch op; 
  op.SetAttr("N", 0);  
  // op.UpdateInputDesc("indices0",  create_desc({3}, ge::DT_INT32));                                                                                                                                                                              
  op.UpdateInputDesc("x0", create_desc({3}, ge::DT_FLOAT));

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape({3}), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(3 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);

  int64_t constData[3] = {2, 3, 4};
  constTensor.SetData((uint8_t*)constData, 3 * sizeof(int32_t));
  auto const_0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_dynamic_input_indices(0,const_0);
  
  auto ret = op.InferShapeAndType();
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
}
