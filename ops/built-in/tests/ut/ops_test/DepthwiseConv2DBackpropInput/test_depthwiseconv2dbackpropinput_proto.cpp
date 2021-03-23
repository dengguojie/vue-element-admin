#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "nn_calculation_ops.h"


class DepthwiseConv2dBackpropInputProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DepthwiseConv2dBackpropInput Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DepthwiseConv2dBackpropInput Proto Test TearDown" << std::endl;
  }
};

// Base_Pass_Case
TEST_F(DepthwiseConv2dBackpropInputProtoTest, Base_Pass_Case){
    ge::op::DepthwiseConv2DBackpropInput op;
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({-1, 32, -1, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, 32, -1, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 3}, {32, 32}, {63, 65}, {63, 65}}));
    op.UpdateInputDesc("filter",
                       create_desc_with_ori({1, 32, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 32, 3, 3}, ge::FORMAT_NCHW));

    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{1 ,32 ,64 , 64};
    ge::TensorDesc tensor_desc_input_size(ge::Shape(),
      ge::FORMAT_NCHW, ge::DT_INT32);
    int element_size = dims_input_size.size();
    tensor_desc_input_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_input_size);

    int *conv_input_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_input_size_tensor_value + i) = dims_input_size[i];
    }
    constTensor.SetData((uint8_t *) conv_input_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("input_size").set_attr_value(constTensor);
    op.set_input_input_size(const0);

    delete[] conv_input_size_tensor_value;
      
    op.UpdateInputDesc("input_size", tensor_desc_input_size);

    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("padding", "SAME");
    op.SetAttr("pads", {-1, -1, -1, -1});
    op.SetAttr("data_format", "NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("input_grad");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

// Input_Size
TEST_F(DepthwiseConv2dBackpropInputProtoTest, Input_Size){
    ge::op::DepthwiseConv2DBackpropInput op;
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({-1, 32, -1, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, 32, -1, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 3}, {32, 32}, {63, 65}, {63, 65}}));
    op.UpdateInputDesc("filter",
                       create_desc_with_ori({1, 32, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 32, 3, 3}, ge::FORMAT_NCHW));

    auto depthwise_conv2d_backprop_input_ori_shape_data = ge::op::Data("input_size");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    depthwise_conv2d_backprop_input_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    depthwise_conv2d_backprop_input_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_input_size(depthwise_conv2d_backprop_input_ori_shape_data);
    op.UpdateInputDesc("input_size", ori_tensorDesc);
    op.UpdateOutputDesc("input_grad",
                    create_desc_shape_range({-1, 32, -1, -1},
                                            ge::DT_FLOAT16,
                                            ge::FORMAT_NCHW,
                                            {-1, 32, -1, -1},
                                            ge::FORMAT_NCHW,
                                            {{1, 2}, {32, 32}, {63, 65}, {63, 65}}));
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("padding", "SAME");
    op.SetAttr("pads", {-1, -1, -1, -1});
    op.SetAttr("data_format", "NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
