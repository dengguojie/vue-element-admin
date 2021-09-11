#include <gtest/gtest.h>
#include <vector>
#include "image_ops.h"
#include "op_proto_test_util.h"

class ResizeDTest : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "resize_d test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "resize_d test TearDown" << std::endl;
    }
};

TEST_F(ResizeDTest, resize_d_test_case_1) {

    ge::op::ResizeD resize_d_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({1, 1, 2, 2});
    tensorDesc.SetDataType(ge::DT_FLOAT);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);


    resize_d_op.UpdateInputDesc("x", tensorDesc);

    std::vector<int64_t> attr_value ={2, 2};
    resize_d_op.SetAttr("sizes", attr_value);

    std::string mode = "cubic";
    resize_d_op.SetAttr("mode", mode);


    auto ret = resize_d_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);


    auto output_desc = resize_d_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {1, 1, 2, 2};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(ResizeDTest, resize_d_test_case_2) {

    ge::op::ResizeD resize_d_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({4, 1, 1, 2});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);


    resize_d_op.UpdateInputDesc("x", tensorDesc);

    std::vector<int64_t> attr_value = {4,};
    resize_d_op.SetAttr("sizes", attr_value);

    std::string mode = "linear";
    resize_d_op.SetAttr("mode", mode);


    auto ret = resize_d_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);


    auto output_desc = resize_d_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {4, 1, 1, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

