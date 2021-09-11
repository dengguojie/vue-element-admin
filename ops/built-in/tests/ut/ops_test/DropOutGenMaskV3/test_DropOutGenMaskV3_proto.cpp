#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "random_ops.h"
#include "array_ops.h"

class dropOutGenMaskV3 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "dropOutGenMaskV3 Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "dropOutGenMaskV3 Proto Test TearDown" << std::endl;
  }
};

TEST_F(dropOutGenMaskV3, dropOutGenMaskV3_infershape_diff_test){
  ge::op::DropOutGenMaskV3 op;
  op.UpdateInputDesc("shape", create_desc({3}, ge::DT_INT32));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(dropOutGenMaskV3, dropOutGenMaskV3_infershape_diff_test_1){
  ge::op::DropOutGenMaskV3 op;

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape({4}), ge::FORMAT_NHWC, ge::DT_INT32);
  constDesc.SetSize(4 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[4] = {2, 3, 4, 5};
  constTensor.SetData((uint8_t*)constData, 4 * sizeof(int32_t));
  auto output_size = ge::op::Constant().set_attr_value(constTensor);

  op.set_input_shape(output_size);
  auto desc = op.GetInputDesc("shape");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("shape", desc);

  auto probDesc = op.GetInputDesc("prob");
  probDesc.SetDataType(ge::DT_FLOAT);
  probDesc.SetShape(ge::Shape());
  op.UpdateInputDesc("prob", probDesc);
  

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(dropOutGenMaskV3, dropOutGenMaskV3_infershape_align_test_1){
  ge::op::DropOutGenMaskV3 op;

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape({4}), ge::FORMAT_NHWC, ge::DT_INT32);
  constDesc.SetSize(4 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[4] = {2, 2, 16, 32};
  constTensor.SetData((uint8_t*)constData, 4 * sizeof(int32_t));
  auto output_size = ge::op::Constant().set_attr_value(constTensor);

  op.set_input_shape(output_size);
  auto desc = op.GetInputDesc("shape");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("shape", desc);

  auto probDesc = op.GetInputDesc("prob");
  probDesc.SetDataType(ge::DT_FLOAT);
  probDesc.SetShape(ge::Shape());
  op.UpdateInputDesc("prob", probDesc);
  

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(dropOutGenMaskV3, dropOutGenMaskV3_infershape_invalid_dim_test_1){
  ge::op::DropOutGenMaskV3 op;

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape({4}), ge::FORMAT_NHWC, ge::DT_INT32);
  constDesc.SetSize(4 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[4] = {2, 2, -16, 32};
  constTensor.SetData((uint8_t*)constData, 4 * sizeof(int32_t));
  auto output_size = ge::op::Constant().set_attr_value(constTensor);

  op.set_input_shape(output_size);
  auto desc = op.GetInputDesc("shape");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("shape", desc);

  auto probDesc = op.GetInputDesc("prob");
  probDesc.SetDataType(ge::DT_FLOAT);
  probDesc.SetShape(ge::Shape());
  op.UpdateInputDesc("prob", probDesc);
  

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(dropOutGenMaskV3, dropOutGenMaskV3_infershape_prob_rank_err_1){
  ge::op::DropOutGenMaskV3 op;
  auto probDesc = op.GetInputDesc("prob");
  probDesc.SetDataType(ge::DT_FLOAT);
  probDesc.SetShape(ge::Shape({1}));
  probDesc.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("prob", probDesc);
  

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
