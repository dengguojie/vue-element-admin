#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "matrix_calculation_ops.h"

class scatter_scalar_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "scatter_scalar SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "scatter_scalar TearDown" << std::endl;
    }
};

TEST_F(scatter_scalar_test, scatter_scalar_infershape_test_1) {
    ge::op::ScatterScalar op;
    op.UpdateInputDesc("index", create_desc_with_ori({2, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 4}, ge::FORMAT_ND));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {2, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(scatter_scalar_test, scatter_scalar_infershape_test_2) {
    ge::op::ScatterScalar op;
    op.UpdateInputDesc("index", create_desc_with_ori({2, 4}, ge::DT_INT32, ge::FORMAT_ND, {2, 4}, ge::FORMAT_ND));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
    std::vector<int64_t> expected_output_shape = {2, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}