#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nonlinear_fuc_ops.h"

using namespace ge;
using namespace op;

class threshold_v2_d : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "threshold_v2_d SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "threshold_v2_d Test TearDown" << std::endl;
    }
};

TEST_F(threshold_v2_d, cce_threshold_v2_d_fp16_test1) {
    op::ThresholdV2D threshold_v2_op;
    threshold_v2_op.UpdateInputDesc("x", create_desc({5, 6, 7, 8, 9}, ge::DT_FLOAT16));
    threshold_v2_op.SetAttr("threshold", 3.5f);
    threshold_v2_op.SetAttr("value", 5.6f);
    auto ret = threshold_v2_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = threshold_v2_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {5, 6, 7, 8, 9};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(threshold_v2_d, cce_threshold_v2_d_fp32_test2) {
    ge::op::ThresholdV2D threshold_v2_op;
    threshold_v2_op.UpdateInputDesc("x", create_desc({10, 11, 12, 13}, ge::DT_FLOAT));
    threshold_v2_op.SetAttr("threshold", 3.5f);
    threshold_v2_op.SetAttr("value", 5.6f);
    auto ret = threshold_v2_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = threshold_v2_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {10, 11, 12, 13};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(threshold_v2_d, cce_threshold_v2_d_int32_test3) {
    ge::op::ThresholdV2D threshold_v2_op;
    threshold_v2_op.UpdateInputDesc("x", create_desc({10, 11, 13}, ge::DT_INT32));
    threshold_v2_op.SetAttr("threshold", 3.5f);
    threshold_v2_op.SetAttr("value", 5.6f);
    auto ret = threshold_v2_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = threshold_v2_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
    std::vector<int64_t> expected_output_shape = {10, 11, 13};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(threshold_v2_d, cce_threshold_v2_d_int8_test4) {
    ge::op::ThresholdV2D threshold_v2_op;
    threshold_v2_op.UpdateInputDesc("x", create_desc({10, 11, 13}, ge::DT_INT8));
    threshold_v2_op.SetAttr("threshold", 3.5f);
    threshold_v2_op.SetAttr("value", 5.6f);
    auto ret = threshold_v2_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = threshold_v2_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT8);
    std::vector<int64_t> expected_output_shape = {10, 11, 13};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(threshold_v2_d, cce_threshold_v2_d_uint8_test5) {
    ge::op::ThresholdV2D threshold_v2_op;
    threshold_v2_op.UpdateInputDesc("x", create_desc({10, 11, 13}, ge::DT_UINT8));
    threshold_v2_op.SetAttr("threshold", 3.5f);
    threshold_v2_op.SetAttr("value", 5.6f);
    auto ret = threshold_v2_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = threshold_v2_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_UINT8);
    std::vector<int64_t> expected_output_shape = {10, 11, 13};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}