#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_calculation_ops.h"
#include "array_ops.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "op_desc.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

// ---------------Conv2DTransposeD-------------------
class Conv2DBackpropInputProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv2DBackpropInput Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv2DBackpropInput Proto Test TearDown" << std::endl;
  }
};

//cut stride
TEST_F(Conv2DBackpropInputProtoTest, Conv2DBackpropInputSplicDataTest1) {
    ge::op::Conv2DBackpropInput op;
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
TEST_F(Conv2DBackpropInputProtoTest, Conv2DBackpropInputSplicDataTest2) {
    ge::op::Conv2DBackpropInput op;
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
TEST_F(Conv2DBackpropInputProtoTest, Conv2DBackpropInputDSplicDataTest3) {
    ge::op::Conv2DBackpropInput op;
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
TEST_F(Conv2DBackpropInputProtoTest, Conv2DBackpropInputSplicDataTest4) {
    ge::op::Conv2DBackpropInput op;
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
TEST_F(Conv2DBackpropInputProtoTest, Conv2DBackpropInputSplicDataTest5) {
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

