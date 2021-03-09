#include <gtest/gtest.h>

#include <iostream>
#include <numeric>

#include "array_ops.h"
#include "nn_pooling_ops.h"
#include "op_proto_test_util.h"

class adaptive_avgpool2d_grad : public testing::Test {
   protected:
    static void SetUpTestCase() { std::cout << "AdaptiveAvgPool2dGrad SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AdaptiveAvgPool2dGrad TearDown" << std::endl; }
};

TEST_F(adaptive_avgpool2d_grad, input_normal_4d) {
    ge::op::AdaptiveAvgPool2dGrad op;
    ge::DataType x_type = ge::DT_FLOAT16;
    ge::Format x_format = ge::FORMAT_ND;
    op.UpdateInputDesc("input_grad", create_desc_with_ori({1, 2, 3, 2}, x_type, x_format, {1, 2, 3, 2}, x_format));
    std::vector<int64_t> out_shape = {1, 2, 4, 6};
    op.SetAttr("orig_input_shape", out_shape);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDescByName("output_grad");
    EXPECT_EQ(output_desc.GetDataType(), x_type);
    EXPECT_EQ(output_desc.GetShape().GetDims(), out_shape);
}

TEST_F(adaptive_avgpool2d_grad, input_normal_6d) {
    ge::op::AdaptiveAvgPool2dGrad op;
    ge::DataType x_type = ge::DT_FLOAT;
    ge::Format x_format = ge::FORMAT_ND;
    op.UpdateInputDesc("input_grad", create_desc_with_ori({1, 2, 4, 5, 3, 2}, x_type, x_format, {1, 2, 4, 5, 3, 2}, x_format));
    std::vector<int64_t> out_shape = {1, 2, 4, 5, 4, 6};
    op.SetAttr("orig_input_shape", out_shape);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDescByName("output_grad");
    EXPECT_EQ(output_desc.GetDataType(), x_type);
    EXPECT_EQ(output_desc.GetShape().GetDims(), out_shape);
}

TEST_F(adaptive_avgpool2d_grad, input_attr_not_match) {
    ge::op::AdaptiveAvgPool2dGrad op;
    ge::DataType x_type = ge::DT_FLOAT16;
    ge::Format x_format = ge::FORMAT_ND;
    op.UpdateInputDesc("input_grad", create_desc_with_ori({1, 2, 3, 2}, x_type, x_format, {1, 2, 3, 2}, x_format));

	auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}