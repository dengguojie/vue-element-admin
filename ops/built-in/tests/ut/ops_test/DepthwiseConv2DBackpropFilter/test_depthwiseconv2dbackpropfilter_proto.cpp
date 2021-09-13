#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "nn_calculation_ops.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "op_desc.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"

// ---------------DepthwiseConv2DBackpropFilter-------------------
class DepthwiseConv2DBackpropFilterProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DepthwiseConv2DBackpropFilter Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DepthwiseConv2DBackpropFilter Proto Test TearDown" << std::endl;
  }
};

// base ut
TEST_F(DepthwiseConv2DBackpropFilterProtoTest, DepthwiseConv2DBackpropFilterBaseTest) {
    ge::op::DepthwiseConv2DBackpropFilter op;
    op.UpdateInputDesc("input", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 256, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("filter_grad", create_desc_with_ori({1, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 256, 1, 1}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format","NCHW");
    std::string padding = "VALID";
    op.SetAttr("padding", padding);

    std::vector<int64_t> dims_filter_size{1, 256, 1, 1};
    ge::Tensor constTensor;
    ge::TensorDesc tensor_desc_filter_size(ge::Shape(),
      ge::FORMAT_NCHW, ge::DT_INT32);
    int element_size = dims_filter_size.size();
    tensor_desc_filter_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_filter_size);

    int *conv_filter_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_filter_size_tensor_value + i) = dims_filter_size[i];
    }
    constTensor.SetData((uint8_t *) conv_filter_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("filter_size").set_attr_value(constTensor);
    op.set_input_filter_size(const0);
    delete[] conv_filter_size_tensor_value;
    op.UpdateInputDesc("filter_size", tensor_desc_filter_size);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dynamic c
TEST_F(DepthwiseConv2DBackpropFilterProtoTest, DepthwiseConv2DBackpropFilterDynamicCTest) {
    ge::op::DepthwiseConv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({5,-1,60,50},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {5,-1,60,50},
                                               ge::FORMAT_NCHW,
                                               {{5, 5}, {10, 64}, {60, 60}, {50, 60}}));
    op.UpdateInputDesc("input",
                       create_desc_shape_range({5,-1,120,100},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {5,-1,120,100},
                                               ge::FORMAT_NCHW,
                                               {{5, 5}, {10, 64}, {120, 120}, {100, 100}}));
    op.UpdateOutputDesc("filter_grad", create_desc_with_ori({1, 64, 7, 6},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 64, 7, 6}, ge::FORMAT_NCHW));
    auto filter_ori_shape_data = ge::op::Data("filter_size");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    filter_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    filter_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_filter_size(filter_ori_shape_data);
    op.UpdateInputDesc("filter_size", ori_tensorDesc);

    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {2, 3, 2, 2});
    op.SetAttr("padding", "SAME");
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dynamic nhw
TEST_F(DepthwiseConv2DBackpropFilterProtoTest, DepthwiseConv2DBackpropFilterDynamicNHWTest) {
    ge::op::DepthwiseConv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({-1,32,-1,-1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1,32,-1,-1},
                                               ge::FORMAT_NCHW,
                                               {{1, 2}, {32, 32}, {63, 65}, {63, -1}}));
    op.UpdateInputDesc("input",
                       create_desc_shape_range({-1,32,-1,-1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1,32,-1,-1},
                                               ge::FORMAT_NCHW,
                                               {{1, 2}, {32, 32}, {63, 65}, {63, -1}}));
    op.UpdateOutputDesc("filter_grad",
                       create_desc_with_ori({1, 32, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 32, 3, 3}, ge::FORMAT_NCHW));

    std::vector<int64_t> dims_filter_size{1, 32, 3, 3};
    ge::Tensor constTensor;
    ge::TensorDesc tensor_desc_filter_size(ge::Shape(),
      ge::FORMAT_NCHW, ge::DT_INT32);
    int element_size = dims_filter_size.size();
    tensor_desc_filter_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_filter_size);

    int *conv_filter_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_filter_size_tensor_value + i) = dims_filter_size[i];
    }
    constTensor.SetData((uint8_t *) conv_filter_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("filter_size").set_attr_value(constTensor);
    op.set_input_filter_size(const0);
    delete[] conv_filter_size_tensor_value;
    op.UpdateInputDesc("filter_size", tensor_desc_filter_size);

    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {-1, -1, -1, -1});
    op.SetAttr("padding", "SAME");
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// -2
TEST_F(DepthwiseConv2DBackpropFilterProtoTest, DepthwiseConv2DBackpropFilterUnKnownRankTest) {
    ge::op::DepthwiseConv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({-2},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {-2}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("input", create_desc_with_ori({-2},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {-2}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("filter_grad",
                       create_desc_with_ori({1, 32, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 32, 3, 3}, ge::FORMAT_NCHW));

    std::vector<int64_t> dims_filter_size{1, 32, 3, 3};
    ge::Tensor constTensor;
    ge::TensorDesc tensor_desc_filter_size(ge::Shape(),
      ge::FORMAT_NCHW, ge::DT_INT32);
    int element_size = dims_filter_size.size();
    tensor_desc_filter_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_filter_size);

    int *conv_filter_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_filter_size_tensor_value + i) = dims_filter_size[i];
    }
    constTensor.SetData((uint8_t *) conv_filter_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("filter_size").set_attr_value(constTensor);
    op.set_input_filter_size(const0);
    delete[] conv_filter_size_tensor_value;
    op.UpdateInputDesc("filter_size", tensor_desc_filter_size);

    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("padding", "VALID");
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// stride out of size 4
TEST_F(DepthwiseConv2DBackpropFilterProtoTest, DepthwiseConv2DBackpropFilterVerifyStrideTest2) {
    ge::op::DepthwiseConv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 256, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("input_grad", create_desc_with_ori({128, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 1, 1}, ge::FORMAT_NCHW));

    std::vector<int64_t> dims_filter_size{128, 256, 14, 14};
    ge::Tensor constTensor;
    ge::TensorDesc tensor_desc_filter_size(ge::Shape(),
      ge::FORMAT_NCHW, ge::DT_INT32);
    int element_size = dims_filter_size.size();
    tensor_desc_filter_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_filter_size);

    int *conv_filter_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_filter_size_tensor_value + i) = dims_filter_size[i];
    }
    constTensor.SetData((uint8_t *) conv_filter_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("filter_size").set_attr_value(constTensor);
    op.set_input_filter_size(const0);
    delete[] conv_filter_size_tensor_value;
    op.UpdateInputDesc("filter_size", tensor_desc_filter_size);

    op.SetAttr("strides", {1, 1, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// filter_size 
TEST_F(DepthwiseConv2DBackpropFilterProtoTest, DepthwiseConv2DBackpropFilterVerifyFilterSizeDimTest) {
    ge::op::DepthwiseConv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({5,-1,60,50},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {5,-1,60,50},
                                               ge::FORMAT_NCHW,
                                               {{5, 5}, {10, 64}, {60, 60}, {50, 60}}));
    op.UpdateInputDesc("input",
                       create_desc_shape_range({5,-1,120,100},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {5,-1,120,100},
                                               ge::FORMAT_NCHW,
                                               {{5, 5}, {10, 64}, {120, 120}, {100, 100}}));
    op.UpdateOutputDesc("filter_grad", create_desc_with_ori({1, 64, 7, 6},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 64, 7, 6}, ge::FORMAT_NCHW));
    
    std::vector<int64_t> dims_filter_size{1, 256, 1, 1, 1};
    ge::Tensor constTensor;
    ge::TensorDesc tensor_desc_filter_size(ge::Shape(),
      ge::FORMAT_NCHW, ge::DT_INT32);
    int element_size = dims_filter_size.size();
    tensor_desc_filter_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_filter_size);

    int *conv_filter_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_filter_size_tensor_value + i) = dims_filter_size[i];
    }
    constTensor.SetData((uint8_t *) conv_filter_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("filter_size").set_attr_value(constTensor);
    op.set_input_filter_size(const0);
    delete[] conv_filter_size_tensor_value;
    op.UpdateInputDesc("filter_size", tensor_desc_filter_size);

    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {2, 3, 2, 2});
    op.SetAttr("padding", "SAME");
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// strides 
TEST_F(DepthwiseConv2DBackpropFilterProtoTest, DepthwiseConv2DBackpropFilterVerifyStridesDimTest) {
    ge::op::DepthwiseConv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({5,-1,60,50},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {5,-1,60,50},
                                               ge::FORMAT_NCHW,
                                               {{5, 5}, {10, 64}, {60, 60}, {50, 60}}));
    op.UpdateInputDesc("input",
                       create_desc_shape_range({5,-1,120,100},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {5,-1,120,100},
                                               ge::FORMAT_NCHW,
                                               {{5, 5}, {10, 64}, {120, 120}, {100, 100}}));
    op.UpdateOutputDesc("filter_grad", create_desc_with_ori({1, 64, 7, 6},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 64, 7, 6}, ge::FORMAT_NCHW));
    auto filter_ori_shape_data = ge::op::Data("filter_size");
    std::vector<int64_t> ori_dims{5};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    filter_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    filter_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_filter_size(filter_ori_shape_data);
    op.UpdateInputDesc("filter_size", ori_tensorDesc);

    op.SetAttr("strides", {1, 1, 2, 2, 1});
    op.SetAttr("pads", {2, 3, 2, 2});
    op.SetAttr("padding", "SAME");
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// strides 
TEST_F(DepthwiseConv2DBackpropFilterProtoTest, DepthwiseConv2DBackpropFilterVerifyNoStridesTest) {
    ge::op::DepthwiseConv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({5,-1,60,50},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {5,-1,60,50},
                                               ge::FORMAT_NCHW,
                                               {{5, 5}, {10, 64}, {60, 60}, {50, 60}}));
    op.UpdateInputDesc("input",
                       create_desc_shape_range({5,-1,120,100},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {5,-1,120,100},
                                               ge::FORMAT_NCHW,
                                               {{5, 5}, {10, 64}, {120, 120}, {100, 100}}));
    op.UpdateOutputDesc("filter_grad", create_desc_with_ori({1, 64, 7, 6},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 64, 7, 6}, ge::FORMAT_NCHW));
    auto filter_ori_shape_data = ge::op::Data("filter_size");
    std::vector<int64_t> ori_dims{5};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    filter_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    filter_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_filter_size(filter_ori_shape_data);
    op.UpdateInputDesc("filter_size", ori_tensorDesc);

    op.SetAttr("pads", {2, 3, 2, 2});
    op.SetAttr("padding", "SAME");
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// dilations 
TEST_F(DepthwiseConv2DBackpropFilterProtoTest, DepthwiseConv2DBackpropFilterVerifyDilationsDimTest) {
    ge::op::DepthwiseConv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({5,-1,60,50},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {5,-1,60,50},
                                               ge::FORMAT_NCHW,
                                               {{5, 5}, {10, 64}, {60, 60}, {50, 60}}));
    op.UpdateInputDesc("input",
                       create_desc_shape_range({5,-1,120,100},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {5,-1,120,100},
                                               ge::FORMAT_NCHW,
                                               {{5, 5}, {10, 64}, {120, 120}, {100, 100}}));
    op.UpdateOutputDesc("filter_grad", create_desc_with_ori({1, 64, 7, 6},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 64, 7, 6}, ge::FORMAT_NCHW));
    auto filter_ori_shape_data = ge::op::Data("filter_size");
    std::vector<int64_t> ori_dims{5};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    filter_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    filter_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_filter_size(filter_ori_shape_data);
    op.UpdateInputDesc("filter_size", ori_tensorDesc);

    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {2, 3, 2, 2});
    op.SetAttr("padding", "SAME");
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// data_format 
TEST_F(DepthwiseConv2DBackpropFilterProtoTest, DepthwiseConv2DBackpropFilterVerifyDataFormatTest) {
    ge::op::DepthwiseConv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({5,-1,60,50},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {5,-1,60,50},
                                               ge::FORMAT_NCHW,
                                               {{5, 5}, {10, 64}, {60, 60}, {50, 60}}));
    op.UpdateInputDesc("input",
                       create_desc_shape_range({5,-1,120,100},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {5,-1,120,100},
                                               ge::FORMAT_NCHW,
                                               {{5, 5}, {10, 64}, {120, 120}, {100, 100}}));
    op.UpdateOutputDesc("filter_grad", create_desc_with_ori({1, 64, 7, 6},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 64, 7, 6}, ge::FORMAT_NCHW));
    auto filter_ori_shape_data = ge::op::Data("filter_size");
    std::vector<int64_t> ori_dims{5};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    filter_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    filter_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_filter_size(filter_ori_shape_data);
    op.UpdateInputDesc("filter_size", ori_tensorDesc);

    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {2, 3, 2, 2});
    op.SetAttr("padding", "SAME");
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","HWCN");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// padding 
TEST_F(DepthwiseConv2DBackpropFilterProtoTest, DepthwiseConv2DBackpropFilterVerifyPaddingTest) {
    ge::op::DepthwiseConv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({5,-1,60,50},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {5,-1,60,50},
                                               ge::FORMAT_NCHW,
                                               {{5, 5}, {10, 64}, {60, 60}, {50, 60}}));
    op.UpdateInputDesc("input",
                       create_desc_shape_range({5,-1,120,100},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {5,-1,120,100},
                                               ge::FORMAT_NCHW,
                                               {{5, 5}, {10, 64}, {120, 120}, {100, 100}}));
    op.UpdateOutputDesc("filter_grad", create_desc_with_ori({1, 64, 7, 6},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 64, 7, 6}, ge::FORMAT_NCHW));
    auto filter_ori_shape_data = ge::op::Data("filter_size");
    std::vector<int64_t> ori_dims{5};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    filter_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    filter_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_filter_size(filter_ori_shape_data);
    op.UpdateInputDesc("filter_size", ori_tensorDesc);

    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("padding", "LIST");
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// pads 
TEST_F(DepthwiseConv2DBackpropFilterProtoTest, DepthwiseConv2DBackpropFilterVerifyPadsDimTest) {
    ge::op::DepthwiseConv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({5,-1,60,50},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {5,-1,60,50},
                                               ge::FORMAT_NCHW,
                                               {{5, 5}, {10, 64}, {60, 60}, {50, 60}}));
    op.UpdateInputDesc("input",
                       create_desc_shape_range({5,-1,120,100},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {5,-1,120,100},
                                               ge::FORMAT_NCHW,
                                               {{5, 5}, {10, 64}, {120, 120}, {100, 100}}));
    op.UpdateOutputDesc("filter_grad", create_desc_with_ori({1, 64, 7, 6},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 64, 7, 6}, ge::FORMAT_NCHW));
    auto filter_ori_shape_data = ge::op::Data("filter_size");
    std::vector<int64_t> ori_dims{5};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    filter_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    filter_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_filter_size(filter_ori_shape_data);
    op.UpdateInputDesc("filter_size", ori_tensorDesc);

    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {2, 3, 2});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// pads 
TEST_F(DepthwiseConv2DBackpropFilterProtoTest, DepthwiseConv2DBackpropFilterVerifyPadsTest) {
    ge::op::DepthwiseConv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({5,-1,60,50},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {5,-1,60,50},
                                               ge::FORMAT_NCHW,
                                               {{5, 5}, {10, 64}, {60, 60}, {50, 60}}));
    op.UpdateInputDesc("input",
                       create_desc_shape_range({5,-1,120,100},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {5,-1,120,100},
                                               ge::FORMAT_NCHW,
                                               {{5, 5}, {10, 64}, {120, 120}, {100, 100}}));
    op.UpdateOutputDesc("filter_grad", create_desc_with_ori({1, 64, 7, 6},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 64, 7, 6}, ge::FORMAT_NCHW));
    auto filter_ori_shape_data = ge::op::Data("filter_size");
    std::vector<int64_t> ori_dims{5};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    filter_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    filter_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_filter_size(filter_ori_shape_data);
    op.UpdateInputDesc("filter_size", ori_tensorDesc);

    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {2, 3, 2, -1});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}
