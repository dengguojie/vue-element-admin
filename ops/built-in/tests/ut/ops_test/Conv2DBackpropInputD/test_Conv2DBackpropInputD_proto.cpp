#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_calculation_ops.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "op_desc.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

// ---------------Conv2DBackpropInputD-------------------
class Conv2DBackpropInputDProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv2DBackpropInputD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv2DBackpropInputD Proto Test TearDown" << std::endl;
  }
};

// base ut
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDVerifyBaseTest) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

//check filter out of size 4
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDVerifyfilterTest2) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// check x size
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDVerifyInputTest1) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc({128, 512, 7, 7, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// no stride
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDVerifyStrideTest1) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    //   op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// stride out of size 4
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDVerifyStrideTest2) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// dilations out of size 4
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDVerifyDilationsTest2) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// no pad
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDVerifyPadsTest1) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    //   op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// pad size 3
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDVerifyPadsTest2) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// pad out of -1
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDVerifyPadsTest3) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, -1});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// no input size
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDInferInputSizesTest1) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    //   op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// input size not 4
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDVerifyInputSizesTest2) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// cut Cin
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDSplicDataTest0) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({4, 64, 10, 10},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.SetAttr("input_size", {64, 64, 12, 12});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);
    std::vector<std::vector<int64_t>> y_data_slice = {{}, {0, 2}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_w = op_desc->MutableInputDesc("filter");
    std::vector<std::vector<int64_t>> w_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_w, ge::ATTR_NAME_DATA_SLICE, w_data_slice);

    std::vector<std::vector<int64_t>> expect_w_data_slice = {{0, 26}, {}, {}, {}};
    EXPECT_EQ(expect_w_data_slice, w_data_slice);
}

// cut N
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDSplicDataTest1) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({4, 64, 10, 10},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.SetAttr("input_size", {64, 64, 12, 12});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);
    std::vector<std::vector<int64_t>> y_data_slice ={{6, 20}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("filter");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);

    std::vector<std::vector<int64_t>> expect_x_data_slice = {};
    EXPECT_EQ(expect_x_data_slice, expect_x_data_slice);
}

//cut stride
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDSplicDataTest2) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({4, 64, 10, 10},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.SetAttr("input_size", {64, 64, 12, 12});
    op.SetAttr("strides", {-1, -1, -1, -1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);
    std::vector<std::vector<int64_t>> y_data_slice ={{6, 20}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}
//cut pad
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDSplicDataTest3) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({4, 64, 10, 10},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.SetAttr("input_size", {64, 64, 12, 12});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);
    std::vector<std::vector<int64_t>> y_data_slice ={{6, 20}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

//cut outback
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDSplicDataTest4) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({4, 64, 10},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 64, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.SetAttr("input_size", {64, 64, 12, 12});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);
    std::vector<std::vector<int64_t>> y_data_slice ={{6, 20}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

//cut stride length
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDSplicDataTest5) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({4, 64, 10, 10},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.SetAttr("input_size", {64, 64, 12, 12});
    op.SetAttr("strides", {1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);
    std::vector<std::vector<int64_t>> y_data_slice ={{6, 20}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

//cut dilation
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDSplicDataTest6) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({4, 64, 10, 10},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.SetAttr("input_size", {64, 64, 12, 12});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);
    std::vector<std::vector<int64_t>> y_data_slice ={{6, 20}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

//cut input_sizes
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDSplicDataTest7) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({4, 64, 10, 10},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.SetAttr("input_size", {64, 64, 12});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);
    std::vector<std::vector<int64_t>> y_data_slice ={{6, 20}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

//no need infer input
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDSplicDataTest8) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({4, 64, 10, 10},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.SetAttr("input_size", {64, 64, 12, 12});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);
    std::vector<std::vector<int64_t>> y_data_slice ={{6, 20}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    // ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    // ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// Dim of shape 
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDSplicDataTest9) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({4, 64, 10, 10},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.SetAttr("input_size", {64, 64, 12, 12});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {6}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// infer input in Cin
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDSplicDataTest10) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({4, 64, 10, 10},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.SetAttr("input_size", {64, 64, 12, 12});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {6, 20}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// infer input in H
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDSplicDataTest11) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({4, 64, 10, 10},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.SetAttr("input_size", {64, 64, 12, 12});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {6, 20}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// infer input in W
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDSplicDataTest12) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({4, 64, 10, 10},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.SetAttr("input_size", {64, 64, 12, 12});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {6, 20}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// cannot support cut in block_C
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDSplicDataTest13) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({4, 64, 10, 10},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.SetAttr("input_size", {64, 64, 12, 12});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {}, {6, 20}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, 50331645);
}

// not need infer input
TEST_F(Conv2DBackpropInputDProtoTest, Conv2DBackpropInputDSplicDataTest14) {
    ge::op::Conv2DBackpropInputD op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({4, 64, 10, 10},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.SetAttr("input_size", {64, 64, 12, 12});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}