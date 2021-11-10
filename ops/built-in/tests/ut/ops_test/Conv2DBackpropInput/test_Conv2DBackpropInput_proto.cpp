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
#include "external/graph/ge_error_codes.h"

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
TEST_F(Conv2DBackpropInputProtoTest, Conv2DBackpropInputSplicDataTest3) {
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

//change data_format
TEST_F(Conv2DBackpropInputProtoTest, Conv2DBackpropInputSplicDataTest6) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({4, 64, 10, 10},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    auto input_size_shape = ge::Shape({4});
    ge::TensorDesc desc_input_size(input_size_shape, ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    ge::Tensor input_size_tensor(desc_input_size);
    uint8_t input_size_len = input_size_shape.GetShapeSize() * sizeof(int64_t);
    int64_t data[] = {64, 64, 12, 12};
    input_size_tensor.SetData(reinterpret_cast<uint8_t*>(data), input_size_len);
    auto input_size = ge::op::Const("input_size").set_attr_value(input_size_tensor);
    op.set_input_input_size(input_size);
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","HWCN");
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

//cut pads
TEST_F(Conv2DBackpropInputProtoTest, Conv2DBackpropInputSplicDataTest7) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({4, 64, 10, 10},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    auto input_size_shape = ge::Shape({4});
    ge::TensorDesc desc_input_size(input_size_shape, ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    ge::Tensor input_size_tensor(desc_input_size);
    uint8_t input_size_len = input_size_shape.GetShapeSize() * sizeof(int64_t);
    int64_t data[] = {64, 64, 12, 12};
    input_size_tensor.SetData(reinterpret_cast<uint8_t*>(data), input_size_len);
    auto input_size = ge::op::Const("input_size").set_attr_value(input_size_tensor);
    op.set_input_input_size(input_size);
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

//set pads be negative
TEST_F(Conv2DBackpropInputProtoTest, Conv2DBackpropInputSplicDataTest8) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({4, 64, 10, 10},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    auto input_size_shape = ge::Shape({4});
    ge::TensorDesc desc_input_size(input_size_shape, ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    ge::Tensor input_size_tensor(desc_input_size);
    uint8_t input_size_len = input_size_shape.GetShapeSize() * sizeof(int64_t);
    int64_t data[] = {64, 64, 12, 12};
    input_size_tensor.SetData(reinterpret_cast<uint8_t*>(data), input_size_len);
    auto input_size = ge::op::Const("input_size").set_attr_value(input_size_tensor);
    op.set_input_input_size(input_size);
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {-1, -1, -1, -1});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

//don't set pads
TEST_F(Conv2DBackpropInputProtoTest, Conv2DBackpropInputSplicDataTest9) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({4, 64, 10, 10},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    auto input_size_shape = ge::Shape({4});
    ge::TensorDesc desc_input_size(input_size_shape, ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    ge::Tensor input_size_tensor(desc_input_size);
    uint8_t input_size_len = input_size_shape.GetShapeSize() * sizeof(int64_t);
    int64_t data[] = {64, 64, 12, 12};
    input_size_tensor.SetData(reinterpret_cast<uint8_t*>(data), input_size_len);
    auto input_size = ge::op::Const("input_size").set_attr_value(input_size_tensor);
    op.set_input_input_size(input_size);
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

//cut filter size
TEST_F(Conv2DBackpropInputProtoTest, Conv2DBackpropInputSplicDataTest10) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({4, 64, 10, 10},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    auto input_size_shape = ge::Shape({4});
    ge::TensorDesc desc_input_size(input_size_shape, ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    ge::Tensor input_size_tensor(desc_input_size);
    uint8_t input_size_len = input_size_shape.GetShapeSize() * sizeof(int64_t);
    int64_t data[] = {64, 64, 12, 12};
    input_size_tensor.SetData(reinterpret_cast<uint8_t*>(data), input_size_len);
    auto input_size = ge::op::Const("input_size").set_attr_value(input_size_tensor);
    op.set_input_input_size(input_size);
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

//set wc be 0
TEST_F(Conv2DBackpropInputProtoTest, Conv2DBackpropInputSplicDataTest11) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({4, 64, 10, 10},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 0, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 0, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    auto input_size_shape = ge::Shape({4});
    ge::TensorDesc desc_input_size(input_size_shape, ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    ge::Tensor input_size_tensor(desc_input_size);
    uint8_t input_size_len = input_size_shape.GetShapeSize() * sizeof(int64_t);
    int64_t data[] = {64, 64, 12, 12};
    input_size_tensor.SetData(reinterpret_cast<uint8_t*>(data), input_size_len);
    auto input_size = ge::op::Const("input_size").set_attr_value(input_size_tensor);
    op.set_input_input_size(input_size);
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);

    auto status = op.InferShapeAndType();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

//set xc % wc != 0
TEST_F(Conv2DBackpropInputProtoTest, Conv2DBackpropInputSplicDataTest12) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({4, 64, 10, 10},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({64, 64, 3, 3},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {64, 64, 3, 3}, ge::FORMAT_NCHW));
    auto input_size_shape = ge::Shape({4});
    ge::TensorDesc desc_input_size(input_size_shape, ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    ge::Tensor input_size_tensor(desc_input_size);
    uint8_t input_size_len = input_size_shape.GetShapeSize() * sizeof(int64_t);
    int64_t data[] = {64, 32, 12, 12};
    input_size_tensor.SetData(reinterpret_cast<uint8_t*>(data), input_size_len);
    auto input_size = ge::op::Const("input_size").set_attr_value(input_size_tensor);
    op.set_input_input_size(input_size);
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);

    auto status = op.InferShapeAndType();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// set padding be else
TEST_F(Conv2DBackpropInputProtoTest, Conv2DBackpropInputSplicDataTest13) {
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
    op.SetAttr("padding", "ELSE");
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("offset_x", 0);
    
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// no data slice
TEST_F(Conv2DBackpropInputProtoTest, Conv2DBackpropInputSplicDataTest14) {
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
    std::vector<std::vector<int64_t>> y_data_slice ={{6, 20}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// infer input in H success
TEST_F(Conv2DBackpropInputProtoTest, Conv2DBackpropInputSplicDataTest15) {
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
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {6, 20}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// infer input in W success
TEST_F(Conv2DBackpropInputProtoTest, Conv2DBackpropInputSplicDataTest16) {
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
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {6, 20}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// cannot support cut in block C
TEST_F(Conv2DBackpropInputProtoTest, Conv2DBackpropInputSplicDataTest17) {
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
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {}, {6, 20}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, 50331645);
}

// infer input in N/H/W without overlap success
TEST_F(Conv2DBackpropInputProtoTest, Conv2DBackpropInputSplicDataTest18) {
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
    std::vector<std::vector<int64_t>> y_data_slice ={{6, 20}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// no data slice
TEST_F(Conv2DBackpropInputProtoTest, Conv2DBackpropInputSplicDataTest19) {
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
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}
