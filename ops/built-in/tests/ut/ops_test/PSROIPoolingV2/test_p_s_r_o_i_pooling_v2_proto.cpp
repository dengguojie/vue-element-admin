#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

class PSROIPoolingV2ProtoTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "PSROIPoolingV2 Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "PSROIPoolingV2 Proto Test TearDown" << std::endl;
    }
};

TEST_F(PSROIPoolingV2ProtoTest, p_s_r_o_i_pooling_v2_test_case_1) {
    ge::op::PSROIPoolingV2 op;

    ge::TensorDesc tensor_x_desc;
    ge::Shape x_shape({2, 16*7*7, 7, 7});
    tensor_x_desc.SetDataType(ge::DT_FLOAT16);
    tensor_x_desc.SetShape(x_shape);
    tensor_x_desc.SetOriginShape(x_shape);
    op.UpdateInputDesc("x", tensor_x_desc);

    ge::TensorDesc tensor_rois;
    ge::Shape roi_shape({2, 5, 16});
    tensor_rois.SetDataType(ge::DT_FLOAT16);
    tensor_rois.SetShape(roi_shape);
    tensor_rois.SetOriginShape(roi_shape);
    op.UpdateInputDesc("rois", tensor_rois);

    op.SetAttr("output_dim", 16);
    op.SetAttr("group_size", 7);
    op.SetAttr("spatial_scale", 0.0625f);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {32, 16, 7, 7};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(PSROIPoolingV2ProtoTest, p_s_r_o_i_pooling_v2_test_case_2) {
    ge::op::PSROIPoolingV2 op;

    ge::TensorDesc tensor_x_desc;
    ge::Shape shape({2, 2*7*7, 7, 7});
    tensor_x_desc.SetDataType(ge::DT_FLOAT);
    tensor_x_desc.SetShape(shape);
    tensor_x_desc.SetOriginShape(shape);
    op.UpdateInputDesc("x", tensor_x_desc);

    ge::TensorDesc tensor_rois;
    ge::Shape roi_shape({2, 5, 16});
    tensor_rois.SetDataType(ge::DT_FLOAT);
    tensor_rois.SetShape(roi_shape);
    tensor_rois.SetOriginShape(roi_shape);
    op.UpdateInputDesc("rois", tensor_rois);

    op.SetAttr("output_dim", 2);
    op.SetAttr("group_size", 7);
    op.SetAttr("spatial_scale", 0.0625f);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {32, 2, 7, 7};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(PSROIPoolingV2ProtoTest, p_s_r_o_i_pooling_v2_test_case_3) {
  ge::op::PSROIPoolingV2 op;
  op.UpdateInputDesc("x", create_desc({2, 5, 16}, ge::DT_FLOAT));
  op.SetAttr("group_size", 129);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(PSROIPoolingV2ProtoTest, p_s_r_o_i_pooling_v2_test_case_4) {
  ge::op::PSROIPoolingV2 op;
  op.UpdateInputDesc("x", create_desc({2, 49, 7, 7}, ge::DT_FLOAT));
  op.SetAttr("group_size", 7);
  op.SetAttr("output_dim", 2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(PSROIPoolingV2ProtoTest, p_s_r_o_i_pooling_v2_test_case_5) {
  ge::op::PSROIPoolingV2 op;
  op.UpdateInputDesc("x", create_desc_with_ori({2, 7, 7}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 7, 7}, ge::FORMAT_ND));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(PSROIPoolingV2ProtoTest, p_s_r_o_i_pooling_v2_test_case_6) {
  ge::op::PSROIPoolingV2 op;
  op.UpdateInputDesc("x", create_desc_with_ori({4, 2, 7, 7}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 2, 7, 7}, ge::FORMAT_ND));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(PSROIPoolingV2ProtoTest, p_s_r_o_i_pooling_v2_test_case_7) {
  ge::op::PSROIPoolingV2 op;
  op.UpdateInputDesc("x",
                     create_desc_with_ori({4, 2, 7, 7}, ge::DT_FLOAT, ge::FORMAT_NCHW, {4, 2, 7, 7}, ge::FORMAT_NCHW));
  op.UpdateInputDesc("rois", create_desc_with_ori({7, 7}, ge::DT_FLOAT, ge::FORMAT_NCHW, {7, 7}, ge::FORMAT_NCHW));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}