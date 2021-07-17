#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "matrix_calculation_ops.h"

class index_add_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "index_add SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "index_add TearDown" << std::endl;
    }
};

TEST_F(index_add_test, index_add_infershape_test) {
    ge::op::IndexAdd op;
    op.UpdateInputDesc("var", create_desc_with_ori({33, 25, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 25, 5}, ge::FORMAT_ND));
    op.UpdateInputDesc("indices", create_desc_with_ori({33, 25, 1}, ge::DT_INT32, ge::FORMAT_ND, {33, 25, 1}, ge::FORMAT_ND));
    op.UpdateInputDesc("updates", create_desc_with_ori({33, 25, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 25, 5}, ge::FORMAT_ND));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDescByName("var_out");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {33, 25, 5};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(index_add_test, index_add_verify_test_1) {
    ge::op::IndexAdd op;
    op.UpdateInputDesc("var", create_desc_with_ori({33, 25, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 25, 5}, ge::FORMAT_ND));
    op.UpdateInputDesc("indices", create_desc_with_ori({33, 25, 1}, ge::DT_INT32, ge::FORMAT_ND, {33, 25, 1}, ge::FORMAT_ND));
    op.UpdateInputDesc("updates", create_desc_with_ori({33, 25, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 25, 5}, ge::FORMAT_ND));
    auto ret = op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(index_add_test, index_add_verify_test_2) {
    ge::op::IndexAdd op;
    op.UpdateInputDesc("var", create_desc_with_ori({33, 25, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {33, 25, 5}, ge::FORMAT_ND));
    op.UpdateInputDesc("indices", create_desc_with_ori({33, 25, 1}, ge::DT_INT32, ge::FORMAT_ND, {33, 25, 1}, ge::FORMAT_ND));
    op.UpdateInputDesc("updates", create_desc_with_ori({33, 25, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 25, 5}, ge::FORMAT_ND));
    auto ret = op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}