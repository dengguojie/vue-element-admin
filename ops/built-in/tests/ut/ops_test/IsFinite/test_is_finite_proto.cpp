#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "reduce_ops.h"
#include "math_ops.h"

// ----------------IsFinite--------------
class is_finite : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "is_finite SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "is_finite TearDown" << std::endl;
    }
};

TEST_F(is_finite, is_finite_infershape_test1) {
ge::op::IsFinite op;
op.UpdateInputDesc("x", create_desc_with_ori({3, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 4}, ge::FORMAT_ND));

auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

auto output_desc = op.GetOutputDesc("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_BOOL);

std::vector<int64_t> expected_output_y_shape = {3, 4};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_y_shape);
}

TEST_F(is_finite, is_finite_infershape_test2) {
ge::op::IsFinite op;
op.UpdateInputDesc("x", create_desc_with_ori({5, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {5, -1}, ge::FORMAT_ND));

auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

auto output_desc = op.GetOutputDesc("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_BOOL);

std::vector<int64_t> expected_output_y_shape = {5, -1};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_y_shape);
}

TEST_F(is_finite, is_finite_infershape_test3) {
ge::op::IsFinite op;
op.UpdateInputDesc("x", create_desc_with_ori({-2}, ge::DT_FLOAT, ge::FORMAT_ND, {-2}, ge::FORMAT_ND));

auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

auto output_desc = op.GetOutputDesc("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_BOOL);

std::vector<int64_t> expected_output_y_shape = {-2};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_y_shape);
}

TEST_F(is_finite, is_finite_infershape_test4) {
ge::op::IsFinite op;
op.UpdateInputDesc("x", create_desc_with_ori({
4, 5, 6, 7, 8, 1, 5, 9}, ge::DT_FLOAT16, ge::FORMAT_ND, {
4, 5, 6, 7, 8, 1, 5, 9}, ge::FORMAT_ND));

auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

auto output_desc = op.GetOutputDesc("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_BOOL);

std::vector<int64_t> expected_output_y_shape = {4, 5, 6, 7, 8, 1, 5, 9};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_y_shape);
}