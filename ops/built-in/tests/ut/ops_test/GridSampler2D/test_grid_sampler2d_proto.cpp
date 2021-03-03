#include <gtest/gtest.h>

#include <iostream>
#include <numeric>

#include "array_ops.h"
#include "image_ops.h"
#include "op_proto_test_util.h"

class grid_sampler2d : public testing::Test {
   protected:
    static void SetUpTestCase() { std::cout << "grid_sampler2d SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "grid_sampler2d TearDown" << std::endl; }
};

TEST_F(grid_sampler2d, grid_sampler2d_test1) {
    ge::op::GridSampler2D op;
    std::vector<int64_t> out_shape = {1, 2, 2, 3};

    ge::DataType grid_type = ge::DT_FLOAT16;
    ge::Format grid_format = ge::FORMAT_ND;
    op.UpdateInputDesc("grid", create_desc_with_ori({1, 2, 3, 2}, grid_type, grid_format, {1, 2, 3, 2}, grid_format));
    op.UpdateInputDesc("x", create_desc_with_ori({1, 2, 3, 2}, grid_type, grid_format, {1, 2, 3, 2}, grid_format));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto diff_desc = op.GetOutputDescByName("y");
    EXPECT_EQ(diff_desc.GetDataType(), grid_type);
    EXPECT_EQ(diff_desc.GetShape().GetDims(), out_shape);
}

TEST_F(grid_sampler2d, grid_sampler2d_test2) {
    ge::op::GridSampler2D op;
    std::vector<int64_t> test_shape = {1, 2, 3};

    ge::DataType grid_type = ge::DT_FLOAT16;
    ge::Format grid_format = ge::FORMAT_ND;
    op.UpdateInputDesc("grid", create_desc_with_ori({1, 2, 3}, grid_type, grid_format, {1, 2, 3}, grid_format));
    op.UpdateInputDesc("x", create_desc_with_ori({1, 2, 3}, grid_type, grid_format, {1, 2, 3}, grid_format));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}