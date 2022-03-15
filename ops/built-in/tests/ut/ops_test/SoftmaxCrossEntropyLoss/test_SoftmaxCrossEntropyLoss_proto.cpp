#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"

class SoftmaxCrossEntropyLoss : public testing::Test{
    protected:
        static void SetUpTestCase(){
            std::cout << "SoftmaxCrossEntropyLoss Proto Test SetUp" << std::endl;
        }

        static void TearDownTestCase(){
            std::cout << "SoftmaxCrossEntropyLoss Proto Test TearDown" << std::endl;
        }
};

// static no broadcast case
TEST_F(SoftmaxCrossEntropyLoss, SoftmaxCrossEntropyLoss_infershape_test1) {
    ge::op::SoftmaxCrossEntropyLoss op;
    op.UpdateInputDesc("scores", create_desc({3,5,6}, ge::DT_FLOAT));
    op.UpdateInputDesc("labels", create_desc({3,6}, ge::DT_FLOAT));
    op.UpdateInputDesc("weights", create_desc({5,}, ge::DT_FLOAT));
 
    op.SetAttr("ignore_index", 0);
    op.SetAttr("reduction", "none");
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_loss_desc = op.GetOutputDesc("loss");
    auto output_log_prop_desc = op.GetOutputDesc("log_prop");

    EXPECT_EQ(output_loss_desc.GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(output_log_prop_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape1 = {3,6};
    std::vector<int64_t> expected_output_shape2 = {3,5,6};
    EXPECT_EQ(output_loss_desc.GetShape().GetDims(), expected_output_shape1);
    EXPECT_EQ(output_log_prop_desc.GetShape().GetDims(), expected_output_shape2);
}

TEST_F(SoftmaxCrossEntropyLoss, SoftmaxCrossEntropyLoss_infershape_test2) {
    ge::op::SoftmaxCrossEntropyLoss op;
    op.UpdateInputDesc("scores", create_desc({3,5,6}, ge::DT_FLOAT));
    op.UpdateInputDesc("labels", create_desc({3,6}, ge::DT_FLOAT));
    op.UpdateInputDesc("weights", create_desc({5,}, ge::DT_FLOAT));
 
    op.SetAttr("ignore_index", 0);
    op.SetAttr("reduction", "mean");
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_loss_desc = op.GetOutputDesc("loss");
    auto output_log_prop_desc = op.GetOutputDesc("log_prop");

    EXPECT_EQ(output_loss_desc.GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(output_log_prop_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape1 = {};
    std::vector<int64_t> expected_output_shape2 = {3,5,6};
    EXPECT_EQ(output_loss_desc.GetShape().GetDims(), expected_output_shape1);
    EXPECT_EQ(output_log_prop_desc.GetShape().GetDims(), expected_output_shape2);
}