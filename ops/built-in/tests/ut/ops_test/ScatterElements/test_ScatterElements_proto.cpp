#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "matrix_calculation_ops.h"

class scatter_elements_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "scatter_elements SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "scatter_elements TearDown" << std::endl;
    }
};


// normal cases
TEST_F(scatter_elements_test, scatter_elements_infershape_test) {
    ge::op::ScatterElements op;
    op.UpdateInputDesc("data", create_desc_with_ori({33, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 5}, ge::FORMAT_ND));
    op.UpdateInputDesc("indices", create_desc_with_ori({33, 25, 1}, ge::DT_INT32, ge::FORMAT_ND, {33, 25, 1}, ge::FORMAT_ND));
    op.UpdateInputDesc("updates", create_desc_with_ori({33, 25, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 25, 5}, ge::FORMAT_ND));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {33, 5};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(scatter_elements_test, scatter_elements_verify_test) {
    ge::op::ScatterElements op;
    op.UpdateInputDesc("data", create_desc_with_ori({33, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 5}, ge::FORMAT_ND));
    op.UpdateInputDesc("indices", create_desc_with_ori({33, 25, 1}, ge::DT_INT32, ge::FORMAT_ND, {33, 25, 1}, ge::FORMAT_ND));
    op.UpdateInputDesc("updates", create_desc_with_ori({33, 25, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 25, 5}, ge::FORMAT_ND));
    auto ret = op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

