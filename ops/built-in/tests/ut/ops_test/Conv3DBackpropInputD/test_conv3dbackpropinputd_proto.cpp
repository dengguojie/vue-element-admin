#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_calculation_ops.h"
#include "graph/utils/type_utils.h"
#include "op_desc.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"


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

// infer data slice --- empty query
TEST_F(Conv3DBackpropInputDProtoTest, conv3dbpD_infer_data_empty_slice_Failed){
    ge::op::Conv3DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {2, 4, 14, 14, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 4, 14, 14, 16},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 2, 2, 32, 16}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {2, 2, 2, 32, 16},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 8, 28, 28, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 8, 28, 28, 32},ge::FORMAT_NDHWC));

    op.SetAttr("input_size", {2, 8, 28, 28, 32});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// infer data slice --- query list size larger than one
TEST_F(Conv3DBackpropInputDProtoTest, conv3dbpD_infer_data_slice_query_more_than_one){
    ge::op::Conv3DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {2, 4, 14, 14, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 4, 14, 14, 16},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 2, 2, 32, 16}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {2, 2, 2, 32, 16},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 8, 28, 28, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 8, 28, 28, 32},ge::FORMAT_NDHWC));

    op.SetAttr("input_size", {2, 8, 28, 28, 32});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});

    std::vector<std::vector<int64_t>> y_data_slice ={{0,1}, {0,1}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// infer data slice --- cut N
TEST_F(Conv3DBackpropInputDProtoTest, conv3dbpD_data_slice_cut_n_infer){
    ge::op::Conv3DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {2, 4, 14, 14, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 4, 14, 14, 16},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 2, 2, 32, 16}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {2, 2, 2, 32, 16},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 8, 28, 28, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 8, 28, 28, 32},ge::FORMAT_NDHWC));

    op.SetAttr("input_size", {2, 8, 28, 28, 32});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});

    std::vector<std::vector<int64_t>> y_data_slice ={{0, 1}, {}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_dedy = op_desc->MutableInputDesc("out_backprop");
    std::vector<std::vector<int64_t>> dedy_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_dedy, ge::ATTR_NAME_DATA_SLICE, dedy_data_slice);
    std::vector<std::vector<int64_t>> expect_dedy_data_slice = {{0, 1}, {}, {}, {}, {}, {}};
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(expect_dedy_data_slice, dedy_data_slice);
}

// infer data slice --- cut D
TEST_F(Conv3DBackpropInputDProtoTest, conv3dbpD_data_slice_cut_d_infer){
    ge::op::Conv3DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {2, 4, 14, 14, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 4, 14, 14, 16},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 2, 2, 32, 16}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {2, 2, 2, 32, 16},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 8, 28, 28, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 8, 28, 28, 32},ge::FORMAT_NDHWC));

    op.SetAttr("input_size", {2, 8, 28, 28, 32});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {0, 4}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_dedy = op_desc->MutableInputDesc("out_backprop");
    std::vector<std::vector<int64_t>> dedy_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_dedy, ge::ATTR_NAME_DATA_SLICE, dedy_data_slice);
    std::vector<std::vector<int64_t>> expect_dedy_data_slice = {{}, {0, 2}, {}, {}, {}, {}};
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(expect_dedy_data_slice, dedy_data_slice);
}

// infer data slice --- cut H
TEST_F(Conv3DBackpropInputDProtoTest, conv3dbpD_data_slice_cut_h_infer){
    ge::op::Conv3DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {2, 4, 14, 14, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 4, 14, 14, 16},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 2, 2, 32, 16}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {2, 2, 2, 32, 16},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 8, 28, 28, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 8, 28, 28, 32},ge::FORMAT_NDHWC));

    op.SetAttr("input_size", {2, 8, 28, 28, 32});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {0, 4}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_dedy = op_desc->MutableInputDesc("out_backprop");
    std::vector<std::vector<int64_t>> dedy_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_dedy, ge::ATTR_NAME_DATA_SLICE, dedy_data_slice);
    std::vector<std::vector<int64_t>> expect_dedy_data_slice = {{}, {}, {}, {0, 2}, {}, {}};
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(expect_dedy_data_slice, dedy_data_slice);
}

// infer data slice --- cut W
TEST_F(Conv3DBackpropInputDProtoTest, conv3dbpD_data_slice_cut_w_infer){
    ge::op::Conv3DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {2, 4, 14, 14, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 4, 14, 14, 16},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 2, 2, 32, 16}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {2, 2, 2, 32, 16},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 8, 28, 28, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 8, 28, 28, 32},ge::FORMAT_NDHWC));
    op.SetAttr("input_size", {2, 8, 28, 28, 32});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {}, {0, 4}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_dedy = op_desc->MutableInputDesc("out_backprop");
    std::vector<std::vector<int64_t>> dedy_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_dedy, ge::ATTR_NAME_DATA_SLICE, dedy_data_slice);
    std::vector<std::vector<int64_t>> expect_dedy_data_slice = {{}, {}, {}, {}, {0, 2}, {}};
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(expect_dedy_data_slice, dedy_data_slice);
}

// infer data slice --- cut Cout
TEST_F(Conv3DBackpropInputDProtoTest, conv3dbpD_data_slice_cut_cout_infer){
    ge::op::Conv3DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {2, 4, 14, 14, 64}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 4, 14, 14, 64},ge::FORMAT_NDHWC));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {2, 2, 2, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {2, 2, 2, 64, 64},ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 8, 28, 28, 64}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {2, 8, 28, 28, 64},ge::FORMAT_NDHWC));

    op.SetAttr("input_size", {2, 8, 28, 28, 64});
    op.SetAttr("strides", {1, 2, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {0, 1}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_w = op_desc->MutableInputDesc("filter");
    std::vector<std::vector<int64_t>> w_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_w, ge::ATTR_NAME_DATA_SLICE, w_data_slice);
    std::vector<std::vector<int64_t>> expect_w_data_slice = {{0, 4}, {}, {}, {}};
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(expect_w_data_slice, w_data_slice);
}
