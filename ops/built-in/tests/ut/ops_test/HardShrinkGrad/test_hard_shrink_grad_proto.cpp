#include <gtest/gtest.h>
#include <vector>
#include "nonlinear_fuc_ops.h"

class HardShrinkGradTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "hard_shrink_grad test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "hard_shrink_grad test TearDown" << std::endl;
    }
};

TEST_F(HardShrinkGradTest, hard_shrink_grad_test_case_1) {
    ge::op::HardShrinkGrad hard_shrink_grad_op;
    ge::TensorDesc input_grad;
    ge::Shape input_grad_shape({100, 100});
    input_grad.SetDataType(ge::DT_FLOAT16);
    input_grad.SetShape(input_grad_shape);
    input_grad.SetOriginShape(input_grad_shape);

    ge::TensorDesc input_x;
    ge::Shape input_x_shape({100,100});
    input_x.SetDataType(ge::DT_FLOAT16);
    input_x.SetShape(input_x_shape);
    input_x.SetOriginShape(input_x_shape);

    hard_shrink_grad_op.UpdateInputDesc("gradients", input_grad);
    hard_shrink_grad_op.UpdateInputDesc("features", input_x);

    auto ret = hard_shrink_grad_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = hard_shrink_grad_op.GetOutputDesc("backprops");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {100,100};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
