#include <gtest/gtest.h>
#include <vector>
#include "image_ops.h"
#include "op_proto_test_util.h"

using namespace std;
class ResizeGradDTest : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "resize_grad_d test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "resize_grad_d test TearDown" << std::endl;
    }
};

TEST_F(ResizeGradDTest, resize_grad_d_test_case_1) {

    ge::op::ResizeGradD resize_grad_d_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({1, 2, 3, 4});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    resize_grad_d_op.UpdateInputDesc("grads", tensorDesc);
    //set attr
    std::vector<int> list_shape_value({1, 2, 3, 4});
    resize_grad_d_op.SetAttr("original_size", list_shape_value);

    std::vector<int> list_roi({0});
    resize_grad_d_op.SetAttr("roi", list_roi);

    std::vector<float> list_scales = {0.0,0.0};
    resize_grad_d_op.SetAttr("scales", list_scales);

    std::string mode = "cubic";
    resize_grad_d_op.SetAttr("mode", mode);

    auto ret = resize_grad_d_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = resize_grad_d_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {1,2,3,4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(ResizeGradDTest, resize_grad_d_test_case_2) {

    ge::op::ResizeGradD resize_grad_d_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({1, 1, 1, 2});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);
    resize_grad_d_op.UpdateInputDesc("grads", tensorDesc);
    //set attr
    std::vector<int> list_shape_value = {1, 1, 1};
    resize_grad_d_op.SetAttr("original_size", list_shape_value);

    std::vector<int> list_roi = {0};
    resize_grad_d_op.SetAttr("roi", list_roi);

    std::vector<float> list_scales = {2.0};
    resize_grad_d_op.SetAttr("scales", list_scales);

    std::string mode = "linear";
    resize_grad_d_op.SetAttr("mode", mode);

    auto ret = resize_grad_d_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = resize_grad_d_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {1,1,1,1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}