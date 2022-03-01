#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"

class SoftmaxCrossEntropyWithLogits : public testing::Test{
    protected:
        static void SetUpTestCase(){
            std::cout << "SoftmaxCrossEntropyWithLogits Proto Test SetUp" << std::endl;
        }

        static void TearDownTestCase(){
            std::cout << "SoftmaxCrossEntropyWithLogits Proto Test TearDown" << std::endl;
        }
};

// static no broadcast case
TEST_F(SoftmaxCrossEntropyWithLogits, SoftmaxCrossEntropyWithLogits_infershape_test1) {
    ge::op::SoftmaxCrossEntropyWithLogits op;
    op.UpdateInputDesc("features", create_desc({98, 8}, ge::DT_FLOAT));
    op.UpdateInputDesc("labels", create_desc({98, 8}, ge::DT_FLOAT));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_loss_desc = op.GetOutputDesc("loss");
    auto output_backprop_desc = op.GetOutputDesc("backprop");

    EXPECT_EQ(output_loss_desc.GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(output_backprop_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape1 = {98};
    std::vector<int64_t> expected_output_shape2 = {98, 8};
    EXPECT_EQ(output_loss_desc.GetShape().GetDims(), expected_output_shape1);
    EXPECT_EQ(output_backprop_desc.GetShape().GetDims(), expected_output_shape2);
}

// static broadcast case1
TEST_F(SoftmaxCrossEntropyWithLogits, SoftmaxCrossEntropyWithLogits_infershape_static_broadcast_test1) {
    ge::op::SoftmaxCrossEntropyWithLogits op;
    op.UpdateInputDesc("features", create_desc({8}, ge::DT_FLOAT));
    op.UpdateInputDesc("labels", create_desc({98, 8}, ge::DT_FLOAT));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_loss_desc = op.GetOutputDesc("loss");
    auto output_backprop_desc = op.GetOutputDesc("backprop");

    EXPECT_EQ(output_loss_desc.GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(output_backprop_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape1 = {98};
    std::vector<int64_t> expected_output_shape2 = {98, 8};
    EXPECT_EQ(output_loss_desc.GetShape().GetDims(), expected_output_shape1);
    EXPECT_EQ(output_backprop_desc.GetShape().GetDims(), expected_output_shape2);
}

// static broadcast case2
TEST_F(SoftmaxCrossEntropyWithLogits, SoftmaxCrossEntropyWithLogits_infershape_static_broadcast_test2) {
    ge::op::SoftmaxCrossEntropyWithLogits op;
    op.UpdateInputDesc("features", create_desc({98, 8}, ge::DT_FLOAT));
    op.UpdateInputDesc("labels", create_desc({8}, ge::DT_FLOAT));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_loss_desc = op.GetOutputDesc("loss");
    auto output_backprop_desc = op.GetOutputDesc("backprop");

    EXPECT_EQ(output_loss_desc.GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(output_backprop_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape1 = {98};
    std::vector<int64_t> expected_output_shape2 = {98, 8};
    EXPECT_EQ(output_loss_desc.GetShape().GetDims(), expected_output_shape1);
    EXPECT_EQ(output_backprop_desc.GetShape().GetDims(), expected_output_shape2);
}

// dynamic case
TEST_F(SoftmaxCrossEntropyWithLogits, SoftmaxCrossEntropyWithLogits_infershape_unknown_shape_test1) {
    ge::op::SoftmaxCrossEntropyWithLogits op;
    op.UpdateInputDesc("features", create_desc({-1, -1}, ge::DT_FLOAT));
    op.UpdateInputDesc("labels", create_desc({-1, -1}, ge::DT_FLOAT));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_loss_desc = op.GetOutputDesc("loss");
    auto output_backprop_desc = op.GetOutputDesc("backprop");

    EXPECT_EQ(output_loss_desc.GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(output_backprop_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape1 = {-1};
    std::vector<int64_t> expected_output_shape2 = {-1, -1};
    EXPECT_EQ(output_loss_desc.GetShape().GetDims(), expected_output_shape1);
    EXPECT_EQ(output_backprop_desc.GetShape().GetDims(), expected_output_shape2);
}


// -2 case
TEST_F(SoftmaxCrossEntropyWithLogits, SoftmaxCrossEntropyWithLogits_infershape_unknown_len_test1) {
    ge::op::SoftmaxCrossEntropyWithLogits op;
    op.UpdateInputDesc("features", create_desc({-1, -1}, ge::DT_FLOAT));
    op.UpdateInputDesc("labels", create_desc({-2}, ge::DT_FLOAT));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// -2 case
TEST_F(SoftmaxCrossEntropyWithLogits, SoftmaxCrossEntropyWithLogits_infershape_unknown_len_test2) {
    ge::op::SoftmaxCrossEntropyWithLogits op;
    op.UpdateInputDesc("features", create_desc({-2}, ge::DT_FLOAT));
    op.UpdateInputDesc("labels", create_desc({-2}, ge::DT_FLOAT));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}


// failed case 1
TEST_F(SoftmaxCrossEntropyWithLogits, SoftmaxCrossEntropyWithLogits_infershape_failed_test1) {
    ge::op::SoftmaxCrossEntropyWithLogits op;
    op.UpdateInputDesc("features", create_desc({8}, ge::DT_FLOAT));
    op.UpdateInputDesc("labels", create_desc({8}, ge::DT_FLOAT));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}


// failed case 2
TEST_F(SoftmaxCrossEntropyWithLogits, SoftmaxCrossEntropyWithLogits_infershape_failed_test2) {
    ge::op::SoftmaxCrossEntropyWithLogits op;
    op.UpdateInputDesc("features", create_desc({98, 8}, ge::DT_FLOAT));
    op.UpdateInputDesc("labels", create_desc({7, 8}, ge::DT_FLOAT));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}


// failed case 3
TEST_F(SoftmaxCrossEntropyWithLogits, SoftmaxCrossEntropyWithLogits_infershape_failed_test3) {
    ge::op::SoftmaxCrossEntropyWithLogits op;
    op.UpdateInputDesc("features", create_desc({98, 8}, ge::DT_FLOAT));
    op.UpdateInputDesc("labels", create_desc({98, 7}, ge::DT_FLOAT));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

