#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

class YoloBoxesEncodeTest : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "yolo_boxes_encode test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "yolo_boxes_encode test TearDown" << std::endl;
    }
};

TEST_F(YoloBoxesEncodeTest, yolo_boxes_encode_test_case_1) {
    ge::op::YoloBoxesEncode op;
    op.UpdateInputDesc("anchor_boxes", create_desc_with_ori(
        {6400, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {6400, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("gt_bboxes", create_desc_with_ori(
        {6400, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {6400, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("stride", create_desc_with_ori(
        {6400}, ge::DT_INT32, ge::FORMAT_ND, {6400}, ge::FORMAT_ND));
    op.UpdateOutputDesc("encoded_bboxes", create_desc_with_ori(
        {6400, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {6400, 4}, ge::FORMAT_ND));

    op.SetAttr("performance_mode", "high_precision");

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("encoded_bboxes");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {6400, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(YoloBoxesEncodeTest, yolo_boxes_encode_test_case_2) {
    ge::op::YoloBoxesEncode op;
    op.UpdateInputDesc("anchor_boxes", create_desc_with_ori(
        {6400, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {6400, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("gt_bboxes", create_desc_with_ori(
        {6400, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {6400, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("stride", create_desc_with_ori(
        {6400}, ge::DT_INT32, ge::FORMAT_ND, {6400}, ge::FORMAT_ND));
    op.UpdateOutputDesc("encoded_bboxes", create_desc_with_ori(
        {6400, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {6400, 4}, ge::FORMAT_ND));

    op.SetAttr("performance_mode", "high_precision");

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("encoded_bboxes");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {6400, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(YoloBoxesEncodeTest, yolo_boxes_encode_test_case_3) {
    ge::op::YoloBoxesEncode op;
    op.UpdateInputDesc("anchor_boxes", create_desc_with_ori(
        {6400, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {6400, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("gt_bboxes", create_desc_with_ori(
        {6400, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {6400, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("stride", create_desc_with_ori(
        {6400}, ge::DT_INT32, ge::FORMAT_ND, {6400}, ge::FORMAT_ND));
    op.UpdateOutputDesc("encoded_bboxes", create_desc_with_ori(
        {6400, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {6400, 4}, ge::FORMAT_ND));

    op.SetAttr("performance_mode", "high_performance");

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("encoded_bboxes");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {6400, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(YoloBoxesEncodeTest, yolo_boxes_encode_test_case_4) {
    ge::op::YoloBoxesEncode op;
    op.UpdateInputDesc("anchor_boxes", create_desc_with_ori(
        {1, 6400, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {6400, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("gt_bboxes", create_desc_with_ori(
        {1, 6400, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {6400, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("stride", create_desc_with_ori(
        {6400}, ge::DT_INT32, ge::FORMAT_ND, {6400}, ge::FORMAT_ND));
    op.UpdateOutputDesc("encoded_bboxes", create_desc_with_ori(
        {6400, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {6400, 4}, ge::FORMAT_ND));

    op.SetAttr("performance_mode", "high_precision");

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(YoloBoxesEncodeTest, yolo_boxes_encode_verify_test_1) {
    ge::op::YoloBoxesEncode op;
    op.UpdateInputDesc("anchor_boxes", create_desc_with_ori(
        {6400, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {6400, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("gt_bboxes", create_desc_with_ori(
        {6400, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {6300, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("stride", create_desc_with_ori(
        {6400}, ge::DT_INT32, ge::FORMAT_ND, {6400}, ge::FORMAT_ND));
    op.UpdateOutputDesc("encoded_bboxes", create_desc_with_ori(
        {6400, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {6400, 4}, ge::FORMAT_ND));

    op.SetAttr("performance_mode", "high_precision");
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(YoloBoxesEncodeTest, yolo_boxes_encode_verify_test_2) {
    ge::op::YoloBoxesEncode op;
    op.UpdateInputDesc("anchor_boxes", create_desc_with_ori(
        {6500, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {6500, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("gt_bboxes", create_desc_with_ori(
        {6400, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {6500, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("stride", create_desc_with_ori(
        {6400}, ge::DT_INT32, ge::FORMAT_ND, {6300}, ge::FORMAT_ND));
    op.UpdateOutputDesc("encoded_bboxes", create_desc_with_ori(
        {6400, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {6400, 4}, ge::FORMAT_ND));

    op.SetAttr("performance_mode", "high_precision");
    auto status = op.VerifyAllAttr(true);

    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(YoloBoxesEncodeTest, yolo_boxes_encode_verify_test_3) {
    ge::op::YoloBoxesEncode op;
    op.UpdateInputDesc("anchor_boxes", create_desc_with_ori(
        {6400, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {6400, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("gt_bboxes", create_desc_with_ori(
        {6400, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {6400, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("stride", create_desc_with_ori(
        {5000}, ge::DT_INT32, ge::FORMAT_ND, {5000}, ge::FORMAT_ND));
    op.UpdateOutputDesc("encoded_bboxes", create_desc_with_ori(
        {6400, 3}, ge::DT_FLOAT, ge::FORMAT_ND, {6400, 3}, ge::FORMAT_ND));

    op.SetAttr("performance_mode", "high_precision");
    auto status = op.VerifyAllAttr(true);

    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(YoloBoxesEncodeTest, yolo_boxes_encode_verify_test_4) {
    ge::op::YoloBoxesEncode op;
    op.UpdateInputDesc("anchor_boxes", create_desc_with_ori(
        {6400, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {6400, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("gt_bboxes", create_desc_with_ori(
        {6400, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {6300, 4}, ge::FORMAT_ND));
    op.UpdateInputDesc("stride", create_desc_with_ori(
        {6400}, ge::DT_INT32, ge::FORMAT_ND, {6400}, ge::FORMAT_ND));
    op.UpdateOutputDesc("encoded_bboxes", create_desc_with_ori(
        {6400, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {6400, 4}, ge::FORMAT_ND));

    op.SetAttr("performance_mode", "high");
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}
