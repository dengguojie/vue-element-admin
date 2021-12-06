#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "random_ops.h"
#include "array_ops.h"
#include "utils/attr_utils.h"
#include "utils/op_desc_utils.h"

class dropOutGenMask : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "dropOutGenMask Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "dropOutGenMask Proto Test TearDown" << std::endl;
  }
};

TEST_F(dropOutGenMask, dropOutGenMask_infershape_diff_test){
  ge::op::DropOutGenMask op;
  op.UpdateInputDesc("x", create_desc({3}, ge::DT_INT32));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(dropOutGenMask, dropOutGenMask_infershape_diff_test_1){
  ge::op::DropOutGenMask op;

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape({4}), ge::FORMAT_NHWC, ge::DT_INT64);
  constDesc.SetSize(4 * sizeof(int64_t));
  constTensor.SetTensorDesc(constDesc);
  int64_t constData[4] = {2, 3, 4, 5};
  constTensor.SetData((uint8_t*)constData, 4 * sizeof(int64_t));
  auto output_size = ge::op::Constant().set_attr_value(constTensor);

  op.set_input_shape(output_size);
  auto desc = op.GetInputDesc("shape");
  desc.SetDataType(ge::DT_INT64);
  op.UpdateInputDesc("shape", desc);

  auto probDesc = op.GetInputDesc("prob");
  probDesc.SetDataType(ge::DT_FLOAT);
  probDesc.SetShape(ge::Shape());
  op.UpdateInputDesc("prob", probDesc);  

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(dropOutGenMask, dropOutGenMask_infershape_prob_rank_err_1){
  ge::op::DropOutGenMask op;
  auto probDesc = op.GetInputDesc("prob");
  probDesc.SetDataType(ge::DT_FLOAT);
  probDesc.SetShape(ge::Shape({1}));
  probDesc.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("prob", probDesc);
  

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(dropOutGenMask, dropOutGenMask_infershape_unconstData_test){
  ge::op::DropOutGenMask op;

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto input_sizes_desc = op_desc->MutableInputDesc("shape");
  std::vector<std::pair<int64_t, int64_t>> value_range = {{1, 2}, {32, 32}, {4, 5}, {4, 5}};
  input_sizes_desc->SetValueRange(value_range);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
