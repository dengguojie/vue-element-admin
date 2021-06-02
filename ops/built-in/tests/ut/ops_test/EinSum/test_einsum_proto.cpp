#include <gtest/gtest.h>

#include <iostream>

#include "array_ops.h"
#include "op_proto_test_util.h"
#include "matrix_calculation_ops.h"

class EinsumTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "EinsumTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "EinsumTest TearDown" << std::endl;
  }
};

TEST_F(EinsumTest, einsum_test_case_1) {
    ge::op::Einsum op;
    auto tensor_desc_0 = create_desc_with_ori({3, 2, 5}, ge::DT_FLOAT, ge::FORMAT_ND,
                                    {3, 2, 5}, ge::FORMAT_ND);
    auto tensor_desc_1 = create_desc_with_ori({3, 5, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
                                    {3, 5, 4}, ge::FORMAT_ND);
    op.create_dynamic_input_x(2);
    op.UpdateDynamicInputDesc("x", 0, tensor_desc_0);
    op.UpdateDynamicInputDesc("x", 1, tensor_desc_1);
    
    std::string equation = "bij,bjk->bik";
    op.SetAttr("equation", equation);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {3, 2, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
