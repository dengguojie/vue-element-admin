#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

// ----------------Cast--------------
class cast : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "cast SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "cast TearDown" << std::endl;
    }
};

TEST_F(cast, cast_infershape_test1) {
    ge::op::Cast op;
    op.UpdateInputDesc("x", create_desc_with_ori({3, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 4}, ge::FORMAT_ND));
    op.SetAttr("dst_type", 9);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT64);

    std::vector<int64_t> expected_output_y_shape = {3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_y_shape);
}

TEST_F(cast, cast_infershape_test2) {
    ge::op::Cast op;
    op.UpdateInputDesc("x", create_desc_with_ori({5, -1}, ge::DT_INT64, ge::FORMAT_ND, {5, -1}, ge::FORMAT_ND));
    op.SetAttr("dst_type", 0);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_y_shape = {5, -1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_y_shape);
}

TEST_F(cast, cast_infershape_test3) {
    ge::op::Cast op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 3, 4}, ge::DT_INT64, ge::FORMAT_ND, {4, 3, 4}, ge::FORMAT_ND));
    op.SetAttr("dst_type", 3);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);

    std::vector<int64_t> expected_output_y_shape = {4, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_y_shape);
}

TEST_F(cast, cast_infershape_test4) {
    ge::op::Cast op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 3, 4, 10}, ge::DT_INT32, ge::FORMAT_ND, {4, 3, 4, 10}, ge::FORMAT_ND));
    op.SetAttr("dst_type", 9);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT64);

    std::vector<int64_t> expected_output_y_shape = {4, 3, 4, 10};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_y_shape);
}

TEST_F(cast, cast_infershape_test5) {
    ge::op::Cast op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 3, 4, 10, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 3, 4, 10, 2}, ge::FORMAT_ND));
    op.SetAttr("dst_type", 27);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_BF16);

    std::vector<int64_t> expected_output_y_shape = {4, 3, 4, 10, 2};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_y_shape);
}

TEST_F(cast, cast_infershape_test6) {
    ge::op::Cast op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 3, 4, 10, 2, 1}, ge::DT_BF16, ge::FORMAT_ND, {4, 3, 4, 10, 2, 1}, ge::FORMAT_ND));
    op.SetAttr("dst_type", 0);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_y_shape = {4, 3, 4, 10, 2, 1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_y_shape);
}

TEST_F(cast, cast_infershape_test7) {
    ge::op::Cast op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 3, 4, 10, 2, 1, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 3, 4, 10, 2, 1, 5}, ge::FORMAT_ND));
    op.SetAttr("dst_type", 27);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_BF16);

    std::vector<int64_t> expected_output_y_shape = {4, 3, 4, 10, 2, 1, 5};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_y_shape);
}

TEST_F(cast, cast_infershape_test8) {
    ge::op::Cast op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 3, 4, 10, 2, 1, 5}, ge::DT_BF16, ge::FORMAT_ND, {4, 3, 4, 10, 2, 1, 5}, ge::FORMAT_ND));
    op.SetAttr("dst_type", 1);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);

    std::vector<int64_t> expected_output_y_shape = {4, 3, 4, 10, 2, 1, 5};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_y_shape);
}

TEST_F(cast, cast_infershape_test9) {
    ge::op::Cast op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 3, 4, 10, 2, 1, 5}, ge::DT_BF16, ge::FORMAT_ND, {4, 3, 4, 10, 2, 1, 5}, ge::FORMAT_ND));
    op.SetAttr("dst_type", 3);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);

    std::vector<int64_t> expected_output_y_shape = {4, 3, 4, 10, 2, 1, 5};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_y_shape);
}
