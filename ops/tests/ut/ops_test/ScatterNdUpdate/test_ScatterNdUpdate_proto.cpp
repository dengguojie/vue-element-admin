#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "matrix_calculation_ops.h"


class scatter_nd_update : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "scatter_nd_update SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "scatter_nd_update TearDown" << std::endl;
    }
};

// TODO fix me run failed
// normal cases
//TEST_F(scatter_nd_update, scatter_nd_update_infershape_test) {
//    ge::op::ScatterNdUpdate op;
//    op.UpdateInputDesc("x", create_desc_with_ori({33, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 5}, ge::FORMAT_ND));
//    op.UpdateInputDesc("indices", create_desc_with_ori({33, 25, 1}, ge::DT_INT32, ge::FORMAT_ND, {33, 25, 1}, ge::FORMAT_ND));
//    op.UpdateInputDesc("updates", create_desc_with_ori({33, 25, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 25, 5}, ge::FORMAT_ND));
//    auto ret = op.InferShapeAndType();
//    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
//    auto output_desc = op.GetOutputDesc("y");
//    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
//    std::vector<int64_t> expected_output_shape = {33, 5};
//    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
//}

TEST_F(scatter_nd_update, scatter_nd_update_verify_test) {
    ge::op::ScatterNdUpdate op;
    op.UpdateInputDesc("x", create_desc_with_ori({33, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 5}, ge::FORMAT_ND));
    op.UpdateInputDesc("indices", create_desc_with_ori({33, 25, 1}, ge::DT_INT32, ge::FORMAT_ND, {33, 25, 1}, ge::FORMAT_ND));
    op.UpdateInputDesc("updates", create_desc_with_ori({33, 25, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 25, 5}, ge::FORMAT_ND));
    auto ret = op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// TODO fix me run failed
// exception cases
//TEST_F(scatter_nd_update, scatter_nd_update_verify_invalid_test) {
//    ge::op::ScatterNdUpdate op;
//    op.UpdateInputDesc("x", create_desc_with_ori({33, 5}, ge::DT_INT32, ge::FORMAT_ND, {33, 5}, ge::FORMAT_ND));
//    op.UpdateInputDesc("indices", create_desc_with_ori({33, 25, 1}, ge::DT_INT32, ge::FORMAT_ND, {33, 25, 1}, ge::FORMAT_ND));
//    op.UpdateInputDesc("updates", create_desc_with_ori({33, 25, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 25, 5}, ge::FORMAT_ND));
//    auto ret = op.VerifyAllAttr(true);
//    EXPECT_EQ(ret, ge::GRAPH_FAILED);
//}
