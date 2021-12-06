#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

class PSROIPoolingGradV2DProtoTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "PSROIPoolingV2D Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "PSROIPoolingV2D Proto Test TearDown" << std::endl;
    }
};

TEST_F(PSROIPoolingGradV2DProtoTest, p_s_r_o_i_pooling_grad_v2_d_test_case_1) {
    // [TODO] define your op here
    ge::op::PSROIPoolingGradV2D op;
    ge::TensorDesc tensor_x;
    ge::Shape x_shape({512, 22, 7, 7});
    tensor_x.SetDataType(ge::DT_FLOAT);
    tensor_x.SetShape(x_shape);
    tensor_x.SetOriginShape(x_shape);

    ge::TensorDesc tensor_rois;
    ge::Shape roi_shape({4, 5, 128});
    tensor_rois.SetDataType(ge::DT_FLOAT);
    tensor_rois.SetShape(roi_shape);
    tensor_rois.SetOriginShape(roi_shape);

    op.SetAttr("output_dim", 22);
    op.SetAttr("group_size", 7);
    op.SetAttr("spatial_scale", 0.0625f);
    op.SetAttr("input_size", {84, 84});

    // [TODO] update op input here
    op.UpdateInputDesc("x", tensor_x);
    op.UpdateInputDesc("rois", tensor_rois);

    // [TODO] call InferShapeAndType function here
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // [TODO] compare dtype and shape of op output
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {4, 1078, 84, 84};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(PSROIPoolingGradV2DProtoTest, p_s_r_o_i_pooling_grad_v2_d_test_case_2) {
    // [TODO] define your op here
    ge::op::PSROIPoolingGradV2D op;
    ge::TensorDesc tensor_x;
    ge::Shape x_shape({512, 22, 7, 7});
    tensor_x.SetDataType(ge::DT_FLOAT16);
    tensor_x.SetShape(x_shape);
    tensor_x.SetOriginShape(x_shape);

    ge::TensorDesc tensor_rois;
    ge::Shape roi_shape({4, 5, 128});
    tensor_rois.SetDataType(ge::DT_FLOAT16);
    tensor_rois.SetShape(roi_shape);
    tensor_rois.SetOriginShape(roi_shape);

    op.SetAttr("output_dim", 22);
    op.SetAttr("group_size", 7);
    op.SetAttr("spatial_scale", 0.0625f);
    op.SetAttr("input_size", {84, 84});

    // [TODO] update op input here
    op.UpdateInputDesc("x", tensor_x);
    op.UpdateInputDesc("rois", tensor_rois);

    // [TODO] call InferShapeAndType function here
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // [TODO] compare dtype and shape of op output
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {4, 1078, 84, 84};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(PSROIPoolingGradV2DProtoTest, p_s_r_o_i_pooling_grad_v2_d_test_case_3) {
    // [TODO] define your op here
    ge::op::PSROIPoolingGradV2D op;
    ge::TensorDesc tensor_x;
    ge::Shape x_shape({512, 22, 7, 7});
    tensor_x.SetDataType(ge::DT_FLOAT16);
    tensor_x.SetShape(x_shape);

    ge::TensorDesc tensor_rois;
    ge::Shape roi_shape({4, 5, 128});
    tensor_rois.SetDataType(ge::DT_FLOAT16);
    tensor_rois.SetShape(roi_shape);

    op.SetAttr("output_dim", 22);
    op.SetAttr("group_size", 256);
    op.SetAttr("spatial_scale", 0.0625f);
    op.SetAttr("input_size", {84, 84});

    // [TODO] update op input here
    op.UpdateInputDesc("x", tensor_x);
    op.UpdateInputDesc("rois", tensor_rois);

    // [TODO] call InferShapeAndType function here
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(PSROIPoolingGradV2DProtoTest, p_s_r_o_i_pooling_grad_v2_d_test_case_4) {
    // [TODO] define your op here
    ge::op::PSROIPoolingGradV2D op;
    ge::TensorDesc tensor_x;
    ge::Shape x_shape({512, 22, 7, 7});
    tensor_x.SetDataType(ge::DT_FLOAT16);
    tensor_x.SetShape(x_shape);

    ge::TensorDesc tensor_rois;
    ge::Shape roi_shape({4, 5, 128});
    tensor_rois.SetDataType(ge::DT_FLOAT16);
    tensor_rois.SetShape(roi_shape);

    op.SetAttr("output_dim", 22);
    op.SetAttr("group_size", 7);
    op.SetAttr("spatial_scale", 0.0625f);
    op.SetAttr("input_size", {84, 84, 84});

    // [TODO] update op input here
    op.UpdateInputDesc("x", tensor_x);
    op.UpdateInputDesc("rois", tensor_rois);

    // [TODO] call InferShapeAndType function here
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(PSROIPoolingGradV2DProtoTest, p_s_r_o_i_pooling_grad_v2_d_test_case_5) {
    // [TODO] define your op here
    ge::op::PSROIPoolingGradV2D op;
    ge::TensorDesc tensor_x;
    ge::Shape x_shape({512, 22, 7, 7});
    tensor_x.SetDataType(ge::DT_FLOAT16);
    tensor_x.SetShape(x_shape);

    ge::TensorDesc tensor_rois;
    ge::Shape roi_shape({4, 5, 128});
    tensor_rois.SetDataType(ge::DT_FLOAT16);
    tensor_rois.SetShape(roi_shape);

    op.SetAttr("output_dim", 23);
    op.SetAttr("group_size", 7);
    op.SetAttr("spatial_scale", 0.0625f);
    op.SetAttr("input_size", {84, 84});

    // [TODO] update op input here
    op.UpdateInputDesc("x", tensor_x);
    op.UpdateInputDesc("rois", tensor_rois);

    // [TODO] call InferShapeAndType function here
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(PSROIPoolingGradV2DProtoTest, p_s_r_o_i_pooling_grad_v2_d_verify_test_case_1) {
    ge::op::PSROIPoolingGradV2D op;
    ge::TensorDesc tensor_x;
    ge::Shape x_shape({1, 2*7*7, 14});
    tensor_x.SetDataType(ge::DT_FLOAT16);
    tensor_x.SetShape(x_shape);

    ge::TensorDesc tensor_rois;
    ge::Shape roi_shape({1, 5, 16});
    tensor_rois.SetDataType(ge::DT_FLOAT16);
    tensor_rois.SetShape(roi_shape);

    op.SetAttr("output_dim", 2);
    op.SetAttr("group_size", 7);
    op.SetAttr("spatial_scale", 0.0625f);
    op.SetAttr("input_size", {28, 28});

    // [TODO] update op input here
    op.UpdateInputDesc("x", tensor_x);
    op.UpdateInputDesc("rois", tensor_rois);

    // [TODO] call Verify function here
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(PSROIPoolingGradV2DProtoTest, p_s_r_o_i_pooling_grad_v2_d_verify_test_case_2) {
    ge::op::PSROIPoolingGradV2D op;
    ge::TensorDesc tensor_x;
    ge::Shape x_shape({16, 2, 7, 7});
    tensor_x.SetDataType(ge::DT_FLOAT16);
    tensor_x.SetShape(x_shape);

    ge::TensorDesc tensor_rois;
    ge::Shape roi_shape({1, 1, 5, 16});
    tensor_rois.SetDataType(ge::DT_FLOAT16);
    tensor_rois.SetShape(roi_shape);

    op.SetAttr("output_dim", 2);
    op.SetAttr("group_size", 7);
    op.SetAttr("spatial_scale", 0.0625f);
    op.SetAttr("input_size", {28, 28});

    // [TODO] update op input here
    op.UpdateInputDesc("x", tensor_x);
    op.UpdateInputDesc("rois", tensor_rois);

    // [TODO] call Verify function here
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(PSROIPoolingGradV2DProtoTest, p_s_r_o_i_pooling_grad_v2_d_verify_test_case_3) {
  ge::op::PSROIPoolingGradV2D op;
  op.UpdateInputDesc("x",
                     create_desc_with_ori({4, 2, 7, 7}, ge::DT_FLOAT, ge::FORMAT_NCHW, {4, 2, 7, 7}, ge::FORMAT_NCHW));
  op.UpdateInputDesc("rois", create_desc_with_ori({7, 7}, ge::DT_FLOAT, ge::FORMAT_NCHW, {7, 7}, ge::FORMAT_NCHW));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}