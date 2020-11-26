#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class fused_mul_add_n : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "fused_mul_add_n SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "fused_mul_add_n TearDown" << std::endl;
    }
};

TEST_F(fused_mul_add_n, fused_mul_add_n_case) {
    ge::op::FusedMulAddN op;

    op.UpdateInputDesc("x1", create_desc({4, 4, 16, 16}, ge::DT_FLOAT));
    op.UpdateInputDesc("x2", create_desc({4, 4, 16, 16}, ge::DT_FLOAT));
    op.UpdateInputDesc("x3", create_desc({1,}, ge::DT_FLOAT));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {4, 4, 16, 16};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}