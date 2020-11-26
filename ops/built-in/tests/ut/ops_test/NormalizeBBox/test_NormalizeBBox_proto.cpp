#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"


class normalize_bbox : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "normalize_bbox SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "normalize_bbox TearDown" << std::endl;
    }
};

TEST_F(normalize_bbox, normalize_bbox_infershape_test_false_fp16) {
    ge::op::NormalizeBBox op;
    op.UpdateInputDesc("boxes", create_desc_with_ori({33, 5, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {33, 5, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("shape_hw", create_desc_with_ori({33, 3}, ge::DT_INT32, ge::FORMAT_ND, {33, 3}, ge::FORMAT_ND));
    op.SetAttr("reversed_box", false);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {33, 5, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(normalize_bbox, normalize_bbox_infershape_test_false_fp32) {
    ge::op::NormalizeBBox op;
    op.UpdateInputDesc("boxes", create_desc_with_ori({33, 28, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {33, 28, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("shape_hw", create_desc_with_ori({33, 3}, ge::DT_INT32, ge::FORMAT_ND, {33, 3}, ge::FORMAT_ND));
    op.SetAttr("reversed_box", false);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {33, 28, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(normalize_bbox, normalize_bbox_infershape_test_true_fp16) {
    ge::op::NormalizeBBox op;
    op.UpdateInputDesc("boxes", create_desc_with_ori({10, 4, 45}, ge::DT_FLOAT16, ge::FORMAT_ND, {10, 4, 45}, ge::FORMAT_ND));
    op.UpdateInputDesc("shape_hw", create_desc_with_ori({10, 3}, ge::DT_INT32, ge::FORMAT_ND, {10, 3}, ge::FORMAT_ND));
    op.SetAttr("reversed_box", true);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {10, 4, 45};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(normalize_bbox, normalize_bbox_infershape_test_true_fp32) {
    ge::op::NormalizeBBox op;
    op.UpdateInputDesc("boxes", create_desc_with_ori({18, 4, 20}, ge::DT_FLOAT, ge::FORMAT_ND, {18, 4, 20}, ge::FORMAT_ND));
    op.UpdateInputDesc("shape_hw", create_desc_with_ori({18, 3}, ge::DT_INT32, ge::FORMAT_ND, {18, 3}, ge::FORMAT_ND));
    op.SetAttr("reversed_box", true);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {18, 4, 20};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
