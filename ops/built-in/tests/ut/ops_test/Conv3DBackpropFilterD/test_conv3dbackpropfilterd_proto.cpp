#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_calculation_ops.h"
#include "graph/utils/type_utils.h"
#include "op_desc.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"


class Conv3DBackpropFilterDProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv3DBackpropFilterD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv3DBackpropFilterD Proto Test TearDown" << std::endl;
  }
};

// Base pass test case 
TEST_F(Conv3DBackpropFilterDProtoTest, conv3dbp_Dw_Base){
    ge::op::Conv3DBackpropFilterD op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {64, 2, 2, 2, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {64, 2, 2, 2, 32}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 2, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 3, 3, 32}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    
    op.SetAttr("filter_size", {64, 2, 2, 2, 32});
    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // auto output_desc = op.GetOutputDesc("y");
    // EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
}

// Filter_Size_Error_Failed
TEST_F(Conv3DBackpropFilterDProtoTest, conv3dbp_Dw_Filter_Size_Error_Failed){
    ge::op::Conv3DBackpropFilterD op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {64, 2, 2, 2, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {64, 2, 2, 2, 32}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 2, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 3, 3, 32}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    
    op.SetAttr("filter_size", {64, 2, 2, 32});
    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
// Not_Filter_Failed
TEST_F(Conv3DBackpropFilterDProtoTest, conv3dbp_Dw_Not_Filter_Failed){
    ge::op::Conv3DBackpropFilterD op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {64, 2, 2, 2, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {64, 2, 2, 2, 32}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 2, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 3, 3, 32}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    
    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Conv3DBackpropFilterDProtoTest, conv3dbpfilter_infer_data_slice_pass) {
    ge::op::Conv3DBackpropFilterD op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {1, 3, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {1, 3, 3, 3, 32}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {1, 3, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {1, 3, 3, 3, 32}, ge::FORMAT_NDHWC));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {32, 1, 1, 1, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {32, 1, 1, 1, 32}, ge::FORMAT_NDHWC));
    op.SetAttr("filter_size", {32, 1, 1, 1, 32});

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {0, 1}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_dedy = op_desc->MutableInputDesc("out_backprop");
    std::vector<std::vector<int64_t>> dedy_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_dedy, ge::ATTR_NAME_DATA_SLICE, dedy_data_slice);
    std::vector<std::vector<int64_t>> expect_dedy_data_slice = {{}, {}, {0, 1}, {}, {}, {}};
    EXPECT_EQ(expect_dedy_data_slice, dedy_data_slice);
}

TEST_F(Conv3DBackpropFilterDProtoTest, conv3dbpfilter_infer_data_empty_slice_Failed) {
    ge::op::Conv3DBackpropFilterD op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {1, 3, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {1, 3, 3, 3, 32}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {1, 3, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {1, 3, 3, 3, 32}, ge::FORMAT_NDHWC));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {32, 1, 1, 1, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {32, 1, 1, 1, 32}, ge::FORMAT_NDHWC));
    op.SetAttr("filter_size", {32, 1, 1, 1, 32});

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// no data slice
TEST_F(Conv3DBackpropFilterDProtoTest, Conv3DBackpropFilterDSliceTest1) {
    ge::op::Conv3DBackpropFilterD op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {1, 3, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {1, 3, 3, 3, 32}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {1, 3, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {1, 3, 3, 3, 32}, ge::FORMAT_NDHWC));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {32, 1, 1, 1, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {32, 1, 1, 1, 32}, ge::FORMAT_NDHWC));
    op.SetAttr("filter_size", {32, 1, 1, 1, 32});

    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// can not supported split in Cin, H and W
TEST_F(Conv3DBackpropFilterDProtoTest, Conv3DBackpropFilterDSliceTest2) {
    ge::op::Conv3DBackpropFilterD op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {1, 3, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {1, 3, 3, 3, 32}, ge::FORMAT_NDHWC));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {1, 3, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {1, 3, 3, 3, 32}, ge::FORMAT_NDHWC));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {32, 1, 1, 1, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {32, 1, 1, 1, 32}, ge::FORMAT_NDHWC));
    op.SetAttr("filter_size", {32, 1, 1, 1, 32});

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {}, {6, 10}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// pads's size is illegal
TEST_F(Conv3DBackpropFilterDProtoTest, Conv3DBackpropFilterDVerifyTest1){
    ge::op::Conv3DBackpropFilterD op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {64, 2, 2, 2, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {64, 2, 2, 2, 32}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 2, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 3, 3, 32}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    
    op.SetAttr("filter_size", {64, 2, 2, 2, 32});
    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// get pads is illegal
TEST_F(Conv3DBackpropFilterDProtoTest, Conv3DBackpropFilterDVerifyTest2){
    ge::op::Conv3DBackpropFilterD op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {64, 2, 2, 2, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {64, 2, 2, 2, 32}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 2, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 3, 3, 32}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    
    op.SetAttr("filter_size", {64, 2, 2, 2, 32});
    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, -1, -1});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// get pads failed
TEST_F(Conv3DBackpropFilterDProtoTest, Conv3DBackpropFilterDVerifyTest3){
    ge::op::Conv3DBackpropFilterD op;
    op.UpdateInputDesc("x", create_desc_with_ori(
      {64, 2, 2, 2, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {64, 2, 2, 2, 32}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 2, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 3, 3, 32}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    
    op.SetAttr("filter_size", {64, 2, 2, 2, 32});
    op.SetAttr("strides", {1, 1, 1, 1, 1});
    // op.SetAttr("pads", {0, 0, 0, 0, -1, -1});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}