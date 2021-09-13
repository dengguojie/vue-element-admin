#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"

class SoftmaxGradTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "SoftmaxGradTest Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "SoftmaxGradTest Proto Test TearDown" << std::endl;
    }
};

TEST_F(SoftmaxGradTest, softmax_grad_tsest_1) {
    ge::op::SoftmaxGrad softmax_grad_op;
    ge::TensorDesc tensor_x_desc;
    ge::Shape x_shape({16, 16});
    tensor_x_desc.SetDataType(ge::DT_FLOAT16);
    tensor_x_desc.SetShape(x_shape);
    tensor_x_desc.SetOriginShape(x_shape);
    tensor_x_desc.SetFormat(ge::FORMAT_ND);
    // update attr
    softmax_grad_op.SetAttr("axes", {-1});
    // update input
    softmax_grad_op.UpdateInputDesc("softmax", tensor_x_desc);
    softmax_grad_op.UpdateInputDesc("grad_softmax", tensor_x_desc);
    // infer
    auto ret = softmax_grad_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    // compare
    auto output_desc = softmax_grad_op.GetOutputDesc("grad_x");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {16, 16};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}