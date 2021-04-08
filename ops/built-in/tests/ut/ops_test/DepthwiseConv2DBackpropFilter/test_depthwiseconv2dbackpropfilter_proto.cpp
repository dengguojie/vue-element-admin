#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "nn_calculation_ops.h"

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

// dynamic nhw
TEST_F(DepthwiseConv2DBackpropFilterProtoTest, DepthwiseConv2DBackpropFilterDynamicCTest) {
    ge::op::DepthwiseConv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({-1,32,-1,-1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1,32,-1,-1},
                                               ge::FORMAT_NCHW,
                                               {{1, 2}, {32, 32}, {63, 65}, {63, 65}}));
    op.UpdateInputDesc("input",
                       create_desc_shape_range({-1,32,-1,-1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1,32,-1,-1},
                                               ge::FORMAT_NCHW,
                                               {{1, 2}, {32, 32}, {63, 65}, {63, 65}}));
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
