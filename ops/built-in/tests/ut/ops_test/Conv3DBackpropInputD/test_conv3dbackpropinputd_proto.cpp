#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_calculation_ops.h"


class Conv3DBackpropInputDProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv3DBackpropInputD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv3DBackpropInputD Proto Test TearDown" << std::endl;
  }
};

// Base pass test case
TEST_F(Conv3DBackpropInputDProtoTest, conv3dbpD_Base){
    ge::op::Conv3DBackpropInputD op;
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 32, 3, 18, 18}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 32, 3, 18, 18}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 32, 2, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {16, 32, 2, 3, 3}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    
    op.SetAttr("input_size", {2, 16, 2, 16, 16});
    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

// Get_Strides_Failed Case
TEST_F(Conv3DBackpropInputDProtoTest, conv3dbpD_Get_Strides_Failed){
    ge::op::Conv3DBackpropInputD op;
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 32, 3, 18, 18}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 32, 3, 18, 18}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 32, 2, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {16, 32, 2, 3, 3}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    
    op.SetAttr("input_size", {2, 16, 2, 16, 16});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

// Strides_Size_Error_Failed
TEST_F(Conv3DBackpropInputDProtoTest, conv3dbpD_Strides_Size_Error_Failed){
    ge::op::Conv3DBackpropInputD op;
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 32, 3, 18, 18}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 32, 3, 18, 18}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 32, 2, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {16, 32, 2, 3, 3}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    
    op.SetAttr("input_size", {2, 16, 2, 16, 16});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

// Padding_Size_Error_Failed 
TEST_F(Conv3DBackpropInputDProtoTest, conv3dbpD_Padding_Size_Error_Failed){
    ge::op::Conv3DBackpropInputD op;
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 32, 3, 18, 18}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 32, 3, 18, 18}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 32, 2, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {16, 32, 2, 3, 3}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    
    op.SetAttr("input_size", {2, 16, 2, 16, 16});
    op.SetAttr("pads", {0, 0, 0, 0, 0});
    op.SetAttr("strides", {1, 1, 1, 1, 1});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

// Negative_Padding_Error_Failed
TEST_F(Conv3DBackpropInputDProtoTest, conv3dbpD_Negative_Padding_Error_Failed){
    ge::op::Conv3DBackpropInputD op;
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 32, 3, 18, 18}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 32, 3, 18, 18}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 32, 2, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {16, 32, 2, 3, 3}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    
    op.SetAttr("input_size", {2, 16, 2, 16, 16});
    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, -1, -1, -1, -1, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

// No_Padding_Failed
TEST_F(Conv3DBackpropInputDProtoTest, conv3dbpD_No_Padding_Failed){
    ge::op::Conv3DBackpropInputD op;
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 32, 3, 18, 18}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 32, 3, 18, 18}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 32, 2, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {16, 32, 2, 3, 3}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    
    op.SetAttr("input_size", {2, 16, 2, 16, 16});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

// FilterDtype_NotEQ_OutBackpropType_Failed 
TEST_F(Conv3DBackpropInputDProtoTest, conv3dbpD_FilterDtype_NotEQ_OutBackpropType_Failed){
    ge::op::Conv3DBackpropInputD op;
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 32, 3, 18, 18}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 32, 3, 18, 18}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 32, 2, 3, 3}, ge::DT_FLOAT, ge::FORMAT_NCDHW,
      {16, 32, 2, 3, 3}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    
    op.SetAttr("input_size", {2, 16, 2, 16, 16});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("strides", {1, 1, 1, 1, 1});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// Filter_Size_Error_Failed 
TEST_F(Conv3DBackpropInputDProtoTest, conv3dbpD_Filter_Size_Error_Failed){
    ge::op::Conv3DBackpropInputD op;
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 32, 18, 18}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
      {2, 32, 18, 18}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 32, 2, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {16, 32, 2, 3, 3}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    
    op.SetAttr("input_size", {2, 16, 2, 16, 16});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("strides", {1, 1, 1, 1, 1});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

// OutBackprop_Size_Error_Failed
TEST_F(Conv3DBackpropInputDProtoTest, conv3dbpD_OutBackprop_Size_Error_Failed){
    ge::op::Conv3DBackpropInputD op;
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 32, 3, 18, 18}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 32, 3, 18, 18}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 32, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
      {16, 32, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    
    op.SetAttr("input_size", {2, 16, 2, 16, 16});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("strides", {1, 1, 1, 1, 1});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

// Input_Size_Error_Failed
TEST_F(Conv3DBackpropInputDProtoTest, conv3dbpD_Input_Size_Error_Failed){
    ge::op::Conv3DBackpropInputD op;
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 32, 3, 18, 18}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 32, 3, 18, 18}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 32, 2, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {16, 32, 2, 3, 3}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    
    op.SetAttr("input_size", {2, 16, 16, 16});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("strides", {1, 1, 1, 1, 1});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

// No_Input_Size_Error_Failed
TEST_F(Conv3DBackpropInputDProtoTest, conv3dbpD_No_Input_Size_Error_Failed){
    ge::op::Conv3DBackpropInputD op;
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 32, 3, 18, 18}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 32, 3, 18, 18}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 32, 2, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {16, 32, 2, 3, 3}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

// Dilation_Size_Error_Failed
TEST_F(Conv3DBackpropInputDProtoTest, conv3dbpD_Dilation_Size_Error_Failed){
    ge::op::Conv3DBackpropInputD op;
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 32, 3, 18, 18}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 32, 3, 18, 18}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 32, 2, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {16, 32, 2, 3, 3}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    
    op.SetAttr("input_size", {2, 16, 2, 16, 16});
    op.SetAttr("dilations", {1, 1, 1});
    op.SetAttr("strides", {1, 1, 1, 1, 1});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}
