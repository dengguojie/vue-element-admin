#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "matrix_calculation_ops.h"

class scatter_tensor_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "scatter_tensor SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "scatter_tensor TearDown" << std::endl;
    }
};

TEST_F(scatter_tensor_test, scatter_tensor_infershape_test_1) {
    ge::op::ScatterTensor op;
    op.UpdateInputDesc("index", create_desc_with_ori({2, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("src", create_desc_with_ori({2, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3}, ge::FORMAT_ND));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {2, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(scatter_tensor_test, scatter_tensor_infershape_test_2) {
    ge::op::ScatterTensor op;
    op.UpdateInputDesc("index", create_desc_with_ori({2, 4}, ge::DT_INT32, ge::FORMAT_ND, {2, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("src", create_desc_with_ori({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {2, 3}, ge::FORMAT_ND));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
    std::vector<int64_t> expected_output_shape = {2, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(scatter_tensor_test, scatter_tensor_verify_test) {
    ge::op::ScatterTensor op;
    op.UpdateInputDesc("index", create_desc_with_ori({2, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("src", create_desc_with_ori({2, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3}, ge::FORMAT_ND));
    auto ret = op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
