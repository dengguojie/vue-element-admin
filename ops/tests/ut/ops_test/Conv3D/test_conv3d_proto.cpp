#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_calculation_ops.h"
using namespace std;

class Conv3DProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv3D Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv3D Proto Test TearDown" << std::endl;
  }
};

// class ge::op::Conv3D
// Test Format_NCDHW_Padding_VALID
TEST_F(Conv3DProtoTest, conv3d_Format_NCDHW_Padding_VALID){
    ge::op::Conv3D op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 32, 3, 18, 18}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 32, 3, 18, 18}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {16, 32, 2, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {16, 32, 2, 3, 3}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 3, 18, 18}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 16, 3, 18, 18}, ge::FORMAT_NCDHW));

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

// Test Format_NDHWC_Padding_SAME
TEST_F(Conv3DProtoTest, conv3d_Format_NDHWC_Padding_SAME){
    ge::op::Conv3D op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 18, 18, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 32}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {16, 2, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 3, 3, 32}, ge::FORMAT_NDHWC));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 3, 18, 18, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 16}, ge::FORMAT_NDHWC));

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}
// Test filter_Format_DHWCN_Padding_0_0_0_0_0_0
TEST_F(Conv3DProtoTest, conv3d_Filter_Format_DHWCN_Padding_0_0_0_0_0_0){
    ge::op::Conv3D op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 18, 18, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 32}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 3, 3, 32, 16}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {2, 3, 3, 32, 16}, ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 3, 18, 18, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 16}, ge::FORMAT_NDHWC));

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

// Test X shape error
TEST_F(Conv3DProtoTest, conv3d_Format_NDHWC_Padding_SAME_X_shape_Failed){
    ge::op::Conv3D op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 18, 18}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
      {2, 3, 18, 18}, ge::FORMAT_NHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {16, 2, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 3, 3, 32}, ge::FORMAT_NDHWC));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 3, 18, 18, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 16}, ge::FORMAT_NDHWC));

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", std::string("SAME"));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// Test Filter shape error
TEST_F(Conv3DProtoTest, conv3d_Format_NDHWC_Padding_SAME_W_Shape_Failed){
    ge::op::Conv3D op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 18, 18, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 32}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {16, 2, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
      {16, 2, 3, 3}, ge::FORMAT_NHWC));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 3, 18, 18, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 16}, ge::FORMAT_NDHWC));

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", std::string("SAME"));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// Padding size Failed
TEST_F(Conv3DProtoTest, conv3d_Padding_Size_Failed){
    ge::op::Conv3D op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 18, 18, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 32}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 3, 3, 32, 16}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {2, 3, 3, 32, 16}, ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 3, 18, 18, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 16}, ge::FORMAT_NDHWC));

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);

}

// xDtype != wDtype
TEST_F(Conv3DProtoTest, conv3d_xDtype_NotEQ_wDtype_Failed){
    ge::op::Conv3D op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 18, 18, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 32}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 3, 3, 32, 16}, ge::DT_FLOAT, ge::FORMAT_DHWCN,
      {2, 3, 3, 32, 16}, ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 3, 18, 18, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 16}, ge::FORMAT_NDHWC));

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// stideList length Failed
TEST_F(Conv3DProtoTest, conv3d_Strides_Length_Failed){
    ge::op::Conv3D op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 18, 18, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 32}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 3, 3, 32, 16}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {2, 3, 3, 32, 16}, ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 3, 18, 18, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 16}, ge::FORMAT_NDHWC));

    op.SetAttr("strides", {1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
// Dilation List length Failed
TEST_F(Conv3DProtoTest, conv3d_Dilation_Length_Failed){
    ge::op::Conv3D op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 18, 18, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 32}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 3, 3, 32, 16}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {2, 3, 3, 32, 16}, ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 3, 18, 18, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 16}, ge::FORMAT_NDHWC));

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
// Negative Strides Failed
TEST_F(Conv3DProtoTest, conv3d_Negative_Strides_Failed){
    ge::op::Conv3D op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 18, 18, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 32}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 3, 3, 32, 16}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {2, 3, 3, 32, 16}, ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 3, 18, 18, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 16}, ge::FORMAT_NDHWC));

    op.SetAttr("strides", {1, 1, -1, -1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// Negative pad Failed
TEST_F(Conv3DProtoTest, conv3d_Negative_Padding_Failed){
    ge::op::Conv3D op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 18, 18, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 32}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 3, 3, 32, 16}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {2, 3, 3, 32, 16}, ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 3, 18, 18, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 16}, ge::FORMAT_NDHWC));

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, -1, -1, -1, -1, 0});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// Output_Y_Format_Failed
TEST_F(Conv3DProtoTest, conv3d_Output_Y_Format_Failed){
    ge::op::Conv3D op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 32, 3, 18, 18}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 32, 3, 18, 18}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {16, 32, 2, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {16, 32, 2, 3, 3}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 16, 16, 2}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {2, 16, 16, 16, 2}, ge::FORMAT_DHWCN));

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", std::string("VALID"));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// ic != lc * group
TEST_F(Conv3DProtoTest, conv3d_X_W_Channel_NotEQ_Failed){
    ge::op::Conv3D op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 32, 3, 18, 18}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 32, 3, 18, 18}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {16, 16, 2, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {16, 16, 2, 3, 3}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", std::string("VALID"));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// Get_Strides_Failed
TEST_F(Conv3DProtoTest, conv3d_Get_Strides_Failed){
    ge::op::Conv3D op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 32, 3, 18, 18}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 32, 3, 18, 18}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {16, 32, 2, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {16, 32, 2, 3, 3}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    op.SetAttr("pads", std::string("VALID"));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// Y_Format_Error_Failed
TEST_F(Conv3DProtoTest, conv3d_Y_Format_Error_Failed){
    ge::op::Conv3D op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 18, 18, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 32}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 3, 3, 32, 16}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {2, 3, 3, 32, 16}, ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {3, 18, 18, 16, 2}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {3, 18, 18, 16, 2}, ge::FORMAT_DHWCN));

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

// Dilation_h_Not_EQ_1_Failed
TEST_F(Conv3DProtoTest, conv3d_Dilation_h_Not_EQ_1_Failed){
    ge::op::Conv3D op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 18, 18, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 32}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 3, 3, 32, 16}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {2, 3, 3, 32, 16}, ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 3, 18, 18, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 3, 18, 18, 16}, ge::FORMAT_NDHWC));

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 2, 2, 2, 1});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}