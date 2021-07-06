#include <gtest/gtest.h>
#include <vector>
#include "image_ops.h"
#include "op_proto_test_util.h"
#include "array_ops.h"

class ResizeTest : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "ResizeTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "ResizeTest TearDown" << std::endl;
    }
};

TEST_F(ResizeTest, ResizeTest_err1) {
    ge::op::Resize resize_op;
    resize_op.SetAttr("const_input", true);
    auto x_desc = create_desc_shape_range({-1,}, ge::DT_FLOAT, ge::FORMAT_ND, {1,1,2,2}, ge::FORMAT_ND, {{1, 1},{2,2},{3,3},{4,4},});
    resize_op.UpdateInputDesc("x", x_desc);
    resize_op.UpdateInputDesc("roi", create_desc({1}, ge::DT_FLOAT));
    resize_op.UpdateInputDesc("scales", create_desc({1}, ge::DT_FLOAT));
    resize_op.UpdateInputDesc("sizes", create_desc({1,1,2,2}, ge::DT_INT64));

    auto ret = resize_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ResizeTest, ResizeTest_err2) {

    ge::op::Resize resize_op;
    resize_op.SetAttr("const_input", true);
    resize_op.UpdateInputDesc("x", create_desc({1,1,2,2}, ge::DT_FLOAT));
    resize_op.UpdateInputDesc("roi", create_desc({1}, ge::DT_FLOAT));
    resize_op.UpdateInputDesc("scales", create_desc({1}, ge::DT_FLOAT));
    resize_op.UpdateInputDesc("sizes", create_desc({1}, ge::DT_INT64));

    auto ret = resize_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ResizeTest, ResizeTest_err3) {

    ge::op::Resize resize_op;
    resize_op.SetAttr("const_input", true);
    resize_op.UpdateInputDesc("x", create_desc({1,1,2,2}, ge::DT_FLOAT));
    resize_op.UpdateInputDesc("roi", create_desc({1,1,2,2}, ge::DT_FLOAT));
    resize_op.UpdateInputDesc("scales", create_desc({1,1,2,2}, ge::DT_INT64));

    auto ret = resize_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ResizeTest, ResizeTest_err4) {

    ge::op::Resize resize_op;
    resize_op.SetAttr("const_input", true);
    resize_op.UpdateInputDesc("x", create_desc({1,1,2,2}, ge::DT_FLOAT));
    resize_op.UpdateInputDesc("roi", create_desc({1,1,2,2}, ge::DT_FLOAT));
    resize_op.UpdateInputDesc("scales", create_desc({1}, ge::DT_FLOAT));

    auto ret = resize_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ResizeTest, ResizeTest_err5) {

    ge::op::Resize resize_op;
    resize_op.SetAttr("const_input", true);
    resize_op.UpdateInputDesc("x", create_desc({1,1,2,2,2}, ge::DT_FLOAT));
    resize_op.UpdateInputDesc("roi", create_desc({1,1,2,2,2}, ge::DT_FLOAT));
    resize_op.UpdateInputDesc("scales", create_desc({1,1,2,2,2}, ge::DT_FLOAT));

    auto ret = resize_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ResizeTest, ResizeTest_err6) {

    std::vector<int64_t> v_scales = {1, 1, 2, 2};
    ge::TensorDesc tensorDesc;
    std::vector<int64_t> dims = {4};
    ge::Shape shape(dims);
    tensorDesc.SetShape(shape);
    tensorDesc.SetDataType(ge::DT_INT64);
    ge::Tensor tensor(tensorDesc, reinterpret_cast<uint8_t*>(v_scales.data()), v_scales.size() * sizeof(int));
    auto scales = ge::op::Const("scales").set_attr_value(tensor);

    ge::op::Resize resize_op;
    resize_op.SetAttr("const_input", true);
    resize_op.UpdateInputDesc("x", create_desc({1,1,2,2}, ge::DT_FLOAT));
    resize_op.UpdateInputDesc("roi", create_desc({1,1,2,2}, ge::DT_FLOAT));
    resize_op.set_input_scales(scales);

    auto ret = resize_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ResizeTest, ResizeTest_err7) {

    std::vector<int64_t> v_sizes = {1,1,2,2,3};
    ge::TensorDesc tensorDesc;
    std::vector<int64_t> dims = {5};
    ge::Shape shape(dims);
    tensorDesc.SetShape(shape);
    tensorDesc.SetDataType(ge::DT_INT64);
    ge::Tensor tensor(tensorDesc, reinterpret_cast<uint8_t*>(v_sizes.data()), v_sizes.size() * sizeof(int));
    auto sizes = ge::op::Const("sizes").set_attr_value(tensor);

    ge::op::Resize resize_op;
    resize_op.SetAttr("const_input", true);
    resize_op.UpdateInputDesc("x", create_desc({1,1,2,2}, ge::DT_FLOAT));
    resize_op.UpdateInputDesc("roi", create_desc({1,1,2,2}, ge::DT_FLOAT));
    resize_op.UpdateInputDesc("scales", create_desc({1,1,2,2}, ge::DT_FLOAT));
    resize_op.set_input_sizes(sizes);

    auto ret = resize_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ResizeTest, ResizeTest_err8) {

    std::vector<int64_t> v_sizes = {1, 1, 2, 2};
    ge::TensorDesc tensorDesc;
    std::vector<int64_t> dims = {4};
    ge::Shape shape(dims);
    tensorDesc.SetShape(shape);
    tensorDesc.SetDataType(ge::DT_INT64);
    ge::Tensor tensor(tensorDesc, reinterpret_cast<uint8_t*>(v_sizes.data()), v_sizes.size() * sizeof(int));
    auto sizes = ge::op::Const("sizes").set_attr_value(tensor);

    ge::op::Resize resize_op;
    resize_op.SetAttr("const_input", true);
    resize_op.UpdateInputDesc("x", create_desc({1}, ge::DT_FLOAT));
    resize_op.UpdateInputDesc("roi", create_desc({1,1,2,2}, ge::DT_FLOAT));
    resize_op.UpdateInputDesc("scales", create_desc({1,1,2,2}, ge::DT_FLOAT));
    resize_op.set_input_sizes(sizes);

    auto ret = resize_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ResizeTest, ResizeTest_err9) {

    std::vector<int64_t> v_scales = {1, 1, 2, 2};
    ge::TensorDesc tensorDesc;
    std::vector<int64_t> dims = {4};
    ge::Shape shape(dims);
    tensorDesc.SetShape(shape);
    tensorDesc.SetDataType(ge::DT_INT64);
    ge::Tensor tensor(tensorDesc, reinterpret_cast<uint8_t*>(v_scales.data()), v_scales.size() * sizeof(int));
    auto scales = ge::op::Const("scales").set_attr_value(tensor);

    ge::op::Resize resize_op;
    resize_op.SetAttr("const_input", true);
    resize_op.UpdateInputDesc("x", create_desc({1,1,2,2}, ge::DT_FLOAT));
    resize_op.UpdateInputDesc("roi", create_desc({1,1,2,2}, ge::DT_FLOAT));
    resize_op.set_input_scales(scales);

    auto ret = resize_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ResizeTest, ResizeTest_err10) {

    std::vector<int64_t> v_scales = {1};
    ge::TensorDesc tensorDesc;
    std::vector<int64_t> dims = {1};
    ge::Shape shape(dims);
    tensorDesc.SetShape(shape);
    tensorDesc.SetDataType(ge::DT_INT64);
    ge::Tensor tensor(tensorDesc, reinterpret_cast<uint8_t*>(v_scales.data()), v_scales.size() * sizeof(int));
    auto scales = ge::op::Const("scales").set_attr_value(tensor);

    ge::op::Resize resize_op;
    resize_op.SetAttr("const_input", true);
    resize_op.UpdateInputDesc("x", create_desc({1}, ge::DT_INT64));
    resize_op.UpdateInputDesc("roi", create_desc({1,1,2,2}, ge::DT_FLOAT));
    resize_op.set_input_scales(scales);

    auto ret = resize_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
