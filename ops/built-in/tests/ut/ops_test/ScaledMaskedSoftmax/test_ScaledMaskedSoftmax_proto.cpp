#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "math_ops.h"

class ScaledMaskedSoftmaxTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "scaled_masked_softmax test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "scaled_masked_softmax test TearDown" << std::endl;
    }
};

TEST_F(ScaledMaskedSoftmaxTest, scaled_masked_softmax_test_1) {
    ge::op::ScaledMaskedSoftmax scaled_masked_softmax_op;
    scaled_masked_softmax_op.UpdateInputDesc("x", create_desc({16,6,8,8,16,16}, ge::DT_FLOAT16));
    scaled_masked_softmax_op.UpdateInputDesc("mask", create_desc({16,6,8,8,16,16}, ge::DT_BOOL));

    auto ret = scaled_masked_softmax_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc_y = scaled_masked_softmax_op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc_y.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {16,6,8,8,16,16};
    EXPECT_EQ(output_desc_y.GetShape().GetDims(), expected_output_shape);
}

TEST_F(ScaledMaskedSoftmaxTest, scaled_masked_softmax_test_2) {
    ge::op::ScaledMaskedSoftmax scaled_masked_softmax_op;
    scaled_masked_softmax_op.UpdateInputDesc("x", create_desc({16,6,32,32,16,16}, ge::DT_FLOAT16));
    scaled_masked_softmax_op.UpdateInputDesc("mask", create_desc({16,6,32,32,16,16}, ge::DT_BOOL));

    auto ret = scaled_masked_softmax_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc_y = scaled_masked_softmax_op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc_y.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {16,6,32,32,16,16};
    EXPECT_EQ(output_desc_y.GetShape().GetDims(), expected_output_shape);
}

TEST_F(ScaledMaskedSoftmaxTest, scaled_masked_softmax_test_3) {
    ge::op::ScaledMaskedSoftmax scaled_masked_softmax_op;
    scaled_masked_softmax_op.UpdateInputDesc("x", create_desc({16,6,32,8,16,16}, ge::DT_FLOAT16));
    scaled_masked_softmax_op.UpdateInputDesc("mask", create_desc({16,6,32,8,16,16}, ge::DT_BOOL));

    auto ret = scaled_masked_softmax_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc_y = scaled_masked_softmax_op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc_y.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {16,6,32,8,16,16};
    EXPECT_EQ(output_desc_y.GetShape().GetDims(), expected_output_shape);
}

TEST_F(ScaledMaskedSoftmaxTest, scaled_masked_softmax_test_4) {
    ge::op::ScaledMaskedSoftmax scaled_masked_softmax_op;
    scaled_masked_softmax_op.UpdateInputDesc("x", create_desc({16,6,32,8,16,16}, ge::DT_FLOAT16));
    scaled_masked_softmax_op.UpdateInputDesc("mask", create_desc({16,1,32,8,16,16}, ge::DT_BOOL));

    auto ret = scaled_masked_softmax_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc_y = scaled_masked_softmax_op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc_y.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {16,6,32,8,16,16};
    EXPECT_EQ(output_desc_y.GetShape().GetDims(), expected_output_shape);
}

TEST_F(ScaledMaskedSoftmaxTest, scaled_masked_softmax_test_5) {
    ge::op::ScaledMaskedSoftmax scaled_masked_softmax_op;
    scaled_masked_softmax_op.UpdateInputDesc("x", create_desc({17,6,32,8,16,16}, ge::DT_FLOAT16));
    scaled_masked_softmax_op.UpdateInputDesc("mask", create_desc({17,1,32,8,16,16}, ge::DT_BOOL));

    auto ret = scaled_masked_softmax_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc_y = scaled_masked_softmax_op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc_y.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {17,6,32,8,16,16};
    EXPECT_EQ(output_desc_y.GetShape().GetDims(), expected_output_shape);
}
