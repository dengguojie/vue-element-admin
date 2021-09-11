#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "math_ops.h"

class CdistGradTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "cdist_grad test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "cdist_grad test TearDown" << std::endl;
    }
};

TEST_F(CdistGradTest, cdist_grad_test_case_1) {
    ge::op::CdistGrad cdist_grad_op;
    ge::TensorDesc input1_desc;
    ge::TensorDesc input2_desc;
    ge::TensorDesc grad_desc;

    ge::Shape input1_shape({2, 3, 5, 4});
    ge::Shape input2_shape({2, 3, 5, 4});
    ge::Shape grad_shape({2, 3, 5, 4});

    input1_desc.SetDataType(ge::DT_FLOAT16);
    input1_desc.SetShape(input1_shape);
    input1_desc.SetOriginShape(input1_shape);
    input2_desc.SetDataType(ge::DT_FLOAT16);
    input2_desc.SetShape(input2_shape);
    input2_desc.SetOriginShape(input2_shape);
    grad_desc.SetDataType(ge::DT_FLOAT16);
    grad_desc.SetShape(grad_shape);
    grad_desc.SetOriginShape(grad_shape);

    cdist_grad_op.UpdateInputDesc("x1", input1_desc);
    cdist_grad_op.UpdateInputDesc("x2", input2_desc);
    cdist_grad_op.UpdateInputDesc("cdist", grad_desc);
    cdist_grad_op.UpdateInputDesc("grad", grad_desc);

    auto ret = cdist_grad_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = cdist_grad_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}