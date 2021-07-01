#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "matrix_calculation_ops.h"


class index_put_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "index_put SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "index_put TearDown" << std::endl;
    }
};

TEST_F(index_put_test, index_put_infershape_test) {
    ge::op::IndexPut op;
    op.UpdateInputDesc("x1", create_desc_with_ori({33, 25, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 25, 5}, ge::FORMAT_ND));
    op.UpdateInputDesc("indices", create_desc_with_ori({33, 25, 1}, ge::DT_INT32, ge::FORMAT_ND, {33, 25, 1}, ge::FORMAT_ND));
    op.UpdateInputDesc("x2", create_desc_with_ori({33, 25, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 25, 5}, ge::FORMAT_ND));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {33, 25, 5};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(index_put_test, index_put_verify_test_1) {
    ge::op::IndexPut op;
    op.UpdateInputDesc("x1", create_desc_with_ori({33, 25, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 25, 5}, ge::FORMAT_ND));
    op.UpdateInputDesc("indices", create_desc_with_ori({33, 25, 1}, ge::DT_INT32, ge::FORMAT_ND, {33, 25, 1}, ge::FORMAT_ND));
    op.UpdateInputDesc("x2", create_desc_with_ori({33, 25, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 25, 5}, ge::FORMAT_ND));
    auto ret = op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(index_put_test, index_put_verify_test_2) {
    ge::op::IndexPut op;
    op.UpdateInputDesc("x1", create_desc_with_ori({33, 25, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {33, 25, 5}, ge::FORMAT_ND));
    op.UpdateInputDesc("indices", create_desc_with_ori({33, 25, 1}, ge::DT_INT32, ge::FORMAT_ND, {33, 25, 1}, ge::FORMAT_ND));
    op.UpdateInputDesc("x2", create_desc_with_ori({33, 25, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 25, 5}, ge::FORMAT_ND));
    auto ret = op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}