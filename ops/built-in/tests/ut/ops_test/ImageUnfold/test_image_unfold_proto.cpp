#include <gtest/gtest.h>

#include <iostream>
#include <numeric>

#include "array_ops.h"
#include "image_ops.h"
#include "op_proto_test_util.h"

class image_unfold : public testing::Test {
   protected:
    static void SetUpTestCase() { std::cout << "image_unfold SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "image_unfold TearDown" << std::endl; }
};

TEST_F(image_unfold, image_unfold_test1) {
    ge::op::ImageUnfold op;
    std::vector<int64_t> x_shape = {1, 2, 3, 3};
    std::vector<int64_t> postion_shape = {1, 3, 3, 2};

    ge::DataType grid_type = ge::DT_FLOAT16;
    ge::Format grid_format = ge::FORMAT_ND;
    op.UpdateInputDesc("position", create_desc_with_ori({1, 3, 3, 2}, ge::DT_INT32, grid_format, {1, 3, 3, 2}, grid_format));
    op.UpdateInputDesc("x", create_desc_with_ori({1, 2, 3, 3}, grid_type, grid_format, {1, 2, 3, 3}, grid_format));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto diff_desc = op.GetOutputDescByName("y");
    EXPECT_EQ(diff_desc.GetDataType(), grid_type);
    EXPECT_EQ(diff_desc.GetShape().GetDims(), x_shape);
}

TEST_F(image_unfold, image_unfold_test2) {
    ge::op::ImageUnfold op;
    std::vector<int64_t> test_shape = {1, 2, 3};

    ge::DataType grid_type = ge::DT_FLOAT16;
    ge::Format grid_format = ge::FORMAT_ND;
    op.UpdateInputDesc("postion", create_desc_with_ori({1, 2, 3}, grid_type, grid_format, {1, 2, 3}, grid_format));
    op.UpdateInputDesc("x", create_desc_with_ori({1, 2, 3}, grid_type, grid_format, {1, 2, 3}, grid_format));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}