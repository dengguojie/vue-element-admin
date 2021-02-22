#include <gtest/gtest.h>

#include <iostream>

#include "array_ops.h"
#include "op_proto_test_util.h"
#include "transformation_ops.h"

class AffineGridTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AffineGridTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AffineGridTest TearDown" << std::endl;
  }
};

TEST_F(AffineGridTest, affine_grid_test_case_1) {
  ge::op::AffineGrid op;

  op.UpdateInputDesc(
      "theta", create_desc_with_ori({2, 2, 3}, ge::DT_FLOAT16, ge::FORMAT_ND,
                                    {2, 2, 3}, ge::FORMAT_ND));

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(4 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[4] = {2, 3, 4, 5};
  constTensor.SetData((uint8_t*)constData, 4 * sizeof(int32_t));
  auto output_size = ge::op::Constant().set_attr_value(constTensor);

  op.set_input_output_size(output_size);
  auto desc = op.GetInputDesc("output_size");
  desc.SetDataType(ge::DT_INT32);
  op.UpdateInputDesc("output_size", desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_var_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_var_output_shape = {2, 20, 2};
  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
}
