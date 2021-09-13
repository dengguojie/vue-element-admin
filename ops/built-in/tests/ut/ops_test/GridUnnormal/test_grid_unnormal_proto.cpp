#include <gtest/gtest.h>

#include <iostream>
#include <numeric>

#include "array_ops.h"
#include "image_ops.h"
#include "op_proto_test_util.h"

class grid_unnormal : public testing::Test {
   protected:
    static void SetUpTestCase() { std::cout << "grid_unnormal SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "grid_unnormal TearDown" << std::endl; }
};

TEST_F(grid_unnormal, grid_unnormal_test1) {
    ge::op::GridUnnormal op;
    std::vector<int64_t> test_shape = {1, 2, 3, 2};
    constexpr int size = 1 * 2 * 3 * 2;
    uint16_t data[size];
    for (int i = 0; i < size; i++) {
        data[i] = 5;
    }

    ge::DataType grid_type = ge::DT_FLOAT16;
    ge::Format grid_format = ge::FORMAT_ND;
    op.UpdateInputDesc("grid", create_desc_with_ori({1, 2, 3, 2}, grid_type, grid_format, {1, 2, 3, 2}, grid_format));

    ge::Tensor assist_tensor;
    ge::TensorDesc assist_desc(ge::Shape(test_shape), grid_format, grid_type);
    assist_desc.SetOriginShape(ge::Shape(test_shape));
    assist_tensor.SetTensorDesc(assist_desc);
    assist_tensor.SetData((uint8_t*)data, size * sizeof(uint16_t));
    auto assist_const_node = ge::op::Constant().set_attr_value(assist_tensor);
    op.set_input_assist(assist_const_node);
    op.UpdateInputDesc("assist", assist_desc);

    op.SetAttr("align_corners", false);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto diff_desc = op.GetOutputDescByName("diff");
    EXPECT_EQ(diff_desc.GetDataType(), grid_type);
    EXPECT_EQ(diff_desc.GetShape().GetDims(), test_shape);
    auto position_desc = op.GetOutputDescByName("position");
    EXPECT_EQ(position_desc.GetDataType(), ge::DT_INT32);
    EXPECT_EQ(position_desc.GetShape().GetDims(), test_shape);
}

TEST_F(grid_unnormal, grid_unnormal_test2) {
    ge::op::GridUnnormal op;
    std::vector<int64_t> test_shape = {7, 8, 4, 2};
    constexpr int size = 7 * 8 * 4 * 2;
    float data[size];
    for (int i = 0; i < size; i++) {
        data[i] = 5;
    }
    ge::DataType grid_type = ge::DT_FLOAT;
    ge::Format grid_format = ge::FORMAT_ND;
    op.UpdateInputDesc("grid", create_desc_with_ori({7, 8, 4, 2}, grid_type, grid_format, {7, 8, 4, 2}, grid_format));

    ge::Tensor assist_tensor;
    ge::TensorDesc assist_desc(ge::Shape(test_shape), grid_format, grid_type);
    assist_desc.SetOriginShape(ge::Shape(test_shape));
    assist_tensor.SetTensorDesc(assist_desc);
    assist_tensor.SetData((uint8_t*)data, size * sizeof(float));
    auto assist_const_node = ge::op::Constant().set_attr_value(assist_tensor);
    op.set_input_assist(assist_const_node);
    op.UpdateInputDesc("assist", assist_desc);

    op.SetAttr("align_corners", true);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto diff_desc = op.GetOutputDescByName("diff");
    EXPECT_EQ(diff_desc.GetDataType(), grid_type);
    EXPECT_EQ(diff_desc.GetShape().GetDims(), test_shape);
    auto position_desc = op.GetOutputDescByName("position");
    EXPECT_EQ(position_desc.GetDataType(), ge::DT_INT32);
    EXPECT_EQ(position_desc.GetShape().GetDims(), test_shape);
}

TEST_F(grid_unnormal, grid_unnormal_test3) {
    ge::op::GridUnnormal op;
    std::vector<int64_t> test_shape = {1, 2, 3, 4};
    constexpr int size = 1 * 2 * 3 * 4;
    float data[size];
    for (int i = 0; i < size; i++) {
        data[i] = 5;
    }
    ge::DataType grid_type = ge::DT_FLOAT;
    ge::Format grid_format = ge::FORMAT_ND;
    op.UpdateInputDesc("grid", create_desc_with_ori({1, 2, 3, 4}, grid_type, grid_format, {1, 2, 3, 4}, grid_format));

    ge::Tensor assist_tensor;
    ge::TensorDesc assist_desc(ge::Shape(test_shape), grid_format, grid_type);
    assist_tensor.SetTensorDesc(assist_desc);
    assist_tensor.SetData((uint8_t*)data, size * sizeof(float));
    auto assist_const_node = ge::op::Constant().set_attr_value(assist_tensor);
    op.set_input_assist(assist_const_node);
    op.UpdateInputDesc("assist", assist_desc);

    op.SetAttr("align_corners", false);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(grid_unnormal, grid_unnormal_test4) {
    ge::op::GridUnnormal op;
    std::vector<int64_t> test_shape = {1, 2, 3};
    constexpr int size = 1 * 2 * 3;
    float data[size];
    for (int i = 0; i < size; i++) {
        data[i] = 5;
    }
    ge::DataType grid_type = ge::DT_FLOAT;
    ge::Format grid_format = ge::FORMAT_ND;
    op.UpdateInputDesc("grid", create_desc_with_ori({1, 2, 3}, grid_type, grid_format, {1, 2, 3}, grid_format));

    ge::Tensor assist_tensor;
    ge::TensorDesc assist_desc(ge::Shape(test_shape), grid_format, grid_type);
    assist_tensor.SetTensorDesc(assist_desc);
    assist_tensor.SetData((uint8_t*)data, size * sizeof(float));
    auto assist_const_node = ge::op::Constant().set_attr_value(assist_tensor);
    op.set_input_assist(assist_const_node);
    op.UpdateInputDesc("assist", assist_desc);

    op.SetAttr("align_corners", false);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}