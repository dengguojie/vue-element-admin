#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "nn_detect_ops.h"

class NonMaxSuppressionV6Test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "NonMaxSuppressionV6 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "NonMaxSuppressionV6 TearDown" << std::endl;
  }
};

TEST_F(NonMaxSuppressionV6Test, non_max_suppression_v6_test_case_1) {


  ge::op::NonMaxSuppressionV6 op;
  op.UpdateInputDesc("boxes", create_desc_with_ori({2, 6, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 6, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores", create_desc_with_ori({2, 1, 6}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 1, 6}, ge::FORMAT_ND));

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {5};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto max_output_size = ge::op::Constant().set_attr_value(constTensor);

  op.set_input_max_output_size(max_output_size);
  auto desc = op.GetInputDesc("max_output_size");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("max_output_size", desc);


  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_var_desc = op.GetOutputDesc("selected_indices");
  std::vector<int64_t> expected_var_output_shape = {10, 3};
  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_INT32);
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);


}

