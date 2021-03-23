#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"

class MaxPoolGradWithArgmaxV1ProtoTest : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "max_pool_grad_with_argmax_v1 test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "max_pool_grad_with_argmax_v1 test TearDown" << std::endl;
    }
};

TEST_F(MaxPoolGradWithArgmaxV1ProtoTest, max_pool_grad_with_argmax_v1_test_case_1) {
    // [TODO] define your op here
    ge::op::MaxPoolGradWithArgmaxV1 op;
    ge::TensorDesc x;
    ge::Shape shape_x({32, 640, 20, 20});
    x.SetDataType(ge::DT_FLOAT16);
    x.SetShape(shape_x);
    op.UpdateInputDesc("x", x);

    ge::TensorDesc grad;
    ge::Shape shape_grad({32, 640, 20, 20});
    grad.SetDataType(ge::DT_FLOAT16);
    grad.SetShape(shape_grad);
    op.UpdateInputDesc("grad", grad);

    ge::TensorDesc argmax;
    ge::Shape shape_argmax({32, 640, 169, 26});
    argmax.SetDataType(ge::DT_UINT16);
    argmax.SetShape(shape_argmax);
    op.UpdateInputDesc("argmax", argmax);

    op.SetAttr("ksize", (1, 13, 13, 1));
    op.SetAttr("strides", (1, 1, 1, 1));
    op.SetAttr("pads", (1, 6, 6, 1));
    op.SetAttr("dtype", 3);
    op.SetAttr("dilation", (1, 1, 1, 1));
    op.SetAttr("ceil_mode", false);

    // [TODO] call InferShapeAndType function here
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // [TODO] compare dtype and shape of op output
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {32, 640, 20, 20};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
