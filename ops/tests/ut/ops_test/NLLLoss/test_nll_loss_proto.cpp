#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "reduce_ops.h"
#include "math_ops.h"

// ----------------NLLLoss--------------
class nll_loss : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "nll_loss SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "nll_loss TearDown" << std::endl;
    }
};

TEST_F(nll_loss, nll_loss_infershape_test) {
    ge::op::NLLLoss op;
    op.UpdateInputDesc("x", create_desc_with_ori({3, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("target", create_desc_with_ori({3, 4}, ge::DT_INT32, ge::FORMAT_ND, {3, }, ge::FORMAT_ND));
    op.UpdateInputDesc("weight", create_desc_with_ori({4, }, ge::DT_FLOAT, ge::FORMAT_ND, {3, }, ge::FORMAT_ND));
    op.SetAttr("reduction", "mean");
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    auto output1_desc = op.GetOutputDesc("total_weight");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(output1_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_y_shape = {};
    std::vector<int64_t> expected_output_total_weight_shape = {};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_y_shape);
    EXPECT_EQ(output1_desc.GetShape().GetDims(), expected_output_total_weight_shape);
}
