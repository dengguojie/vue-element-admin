#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "matrix_calculation_ops.h"

class fill_diagonal_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "fill_diagonal SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "fill_diagonal TearDown" << std::endl;
    }
};

TEST_F(fill_diagonal_test, fill_diagonal_infershape_test_1) {
    ge::op::FillDiagonal op;
    op.UpdateInputDesc("x", create_desc_with_ori({7, 3}, ge::DT_FLOAT, ge::FORMAT_ND, {7, 3}, ge::FORMAT_ND));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {7, 3};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(fill_diagonal_test, fill_diagonal_infershape_test_2) {
    ge::op::FillDiagonal op;
    op.UpdateInputDesc("x", create_desc_with_ori({7, 3}, ge::DT_INT32, ge::FORMAT_ND, {7, 3}, ge::FORMAT_ND));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
    std::vector<int64_t> expected_output_shape = {7, 3};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
