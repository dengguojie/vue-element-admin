#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "rpn_ops.h"


class NMSWithMaskTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "nms_with_mask test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "nms_with_mask test TearDown" << std::endl;
    }
};

TEST_F(NMSWithMaskTest, nms_with_mask_test_case_1) {
    ge::op::NMSWithMask op;
    ge::TensorDesc tensordesc_box_scores;
    ge::Shape box_scores_shape({16, 5});
    tensordesc_box_scores.SetDataType(ge::DT_FLOAT16);
    tensordesc_box_scores.SetShape(box_scores_shape);
    tensordesc_box_scores.SetOriginShape(box_scores_shape);

    op.UpdateInputDesc("box_scores", tensordesc_box_scores);
    float iou_threshold = 0.7;
    op.SetAttr("iou_threshold", iou_threshold);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_boxes_desc = op.GetOutputDesc("selected_boxes");
    EXPECT_EQ(output_boxes_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_boxes_shape = {16, 5};
    EXPECT_EQ(output_boxes_desc.GetShape().GetDims(), expected_boxes_shape);
    auto output_idx_desc = op.GetOutputDesc("selected_idx");
    EXPECT_EQ(output_idx_desc.GetDataType(), ge::DT_INT32);
    std::vector<int64_t> expected_idx_shape = {16};
    EXPECT_EQ(output_idx_desc.GetShape().GetDims(), expected_idx_shape);
    auto output_mask_desc = op.GetOutputDesc("selected_mask");
    EXPECT_EQ(output_mask_desc.GetDataType(), ge::DT_BOOL);
    std::vector<int64_t> expected_mask_shape = {16};
    EXPECT_EQ(output_mask_desc.GetShape().GetDims(), expected_mask_shape);
}

TEST_F(NMSWithMaskTest, nms_with_mask_test_case_2) {
    ge::op::NMSWithMask op;
    ge::TensorDesc tensordesc_box_scores;
    ge::Shape box_scores_shape({720, 5});
    tensordesc_box_scores.SetDataType(ge::DT_FLOAT16);
    tensordesc_box_scores.SetShape(box_scores_shape);
    tensordesc_box_scores.SetOriginShape(box_scores_shape);

    op.UpdateInputDesc("box_scores", tensordesc_box_scores);
    float iou_threshold = 0.7;
    op.SetAttr("iou_threshold", iou_threshold);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_boxes_desc = op.GetOutputDesc("selected_boxes");
    EXPECT_EQ(output_boxes_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_boxes_shape = {720, 5};
    EXPECT_EQ(output_boxes_desc.GetShape().GetDims(), expected_boxes_shape);
    auto output_idx_desc = op.GetOutputDesc("selected_idx");
    EXPECT_EQ(output_idx_desc.GetDataType(), ge::DT_INT32);
    std::vector<int64_t> expected_idx_shape = {720};
    EXPECT_EQ(output_idx_desc.GetShape().GetDims(), expected_idx_shape);
    auto output_mask_desc = op.GetOutputDesc("selected_mask");
    EXPECT_EQ(output_mask_desc.GetDataType(), ge::DT_BOOL);
    std::vector<int64_t> expected_mask_shape = {720};
    EXPECT_EQ(output_mask_desc.GetShape().GetDims(), expected_mask_shape);
}

// failed case
TEST_F(NMSWithMaskTest, nms_with_mask_test_case_3) {
    ge::op::NMSWithMask op;
    ge::TensorDesc tensordesc_box_scores;
    ge::Shape box_scores_shape({720, 5});
    tensordesc_box_scores.SetDataType(ge::DT_FLOAT16);
    tensordesc_box_scores.SetShape(box_scores_shape);

    op.UpdateInputDesc("box_scores", tensordesc_box_scores);
    float iou_threshold = 0;
    op.SetAttr("iou_threshold", iou_threshold);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
