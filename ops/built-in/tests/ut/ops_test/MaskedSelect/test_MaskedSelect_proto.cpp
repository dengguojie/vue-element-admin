#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "selection_ops.h"

class MaskedSelectUT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MaskedSelectUT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MaskedSelectUT TearDown" << std::endl;
  }
};

TEST_F(MaskedSelectUT, masked_select_test_1) {
  ge::op::MaskedSelect op;

  ge::TensorDesc tensor_desc;
  ge::Shape shape({3, 4, 2});
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetShape(shape);
  tensor_desc.SetOriginShape(shape);
  op.UpdateInputDesc("x", tensor_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}