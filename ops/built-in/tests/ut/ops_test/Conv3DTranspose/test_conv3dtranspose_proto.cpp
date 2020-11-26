#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "nn_calculation_ops.h"

// ---------------Conv3DTranspose-------------------
class Conv3DTransposeProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv3DTranspose Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv3DTranspose Proto Test TearDown" << std::endl;
  }
};


// base ut1   FORMAT_NDHWC and FORMAT_DHWCN
TEST_F(Conv3DTransposeProtoTest, Conv3DTransposeTest) {
    ge::op::Conv3DTranspose conv3dtranspose;
    conv3dtranspose.UpdateInputDesc("x", create_desc_with_ori({2, 2, 2, 10, 10},
      ge::DT_FLOAT16, ge::FORMAT_NDHWC,{2, 2, 2, 10, 10},ge::FORMAT_NDHWC));
    conv3dtranspose.UpdateInputDesc("filter", create_desc_with_ori(
      {1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},
      ge::FORMAT_DHWCN));
    conv3dtranspose.UpdateOutputDesc("y", create_desc_with_ori({1, 4, 6, 8, 10},
      ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
    ge::TensorDesc desc_input_size(ge::Shape({1, 4, 6, 8, 10}),
      ge::FORMAT_NDHWC, ge::DT_INT32);
    int element_size = 5;
    desc_input_size.SetSize(element_size * sizeof(int32_t));

    ge::Tensor input_size_tensor;
    input_size_tensor.SetTensorDesc(desc_input_size);
    int *input_size_data = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(input_size_data + i) = 0;
    }
    input_size_tensor.SetData((uint8_t *) input_size_data,
                              element_size * sizeof(int32_t));
    auto const_data = ge::op::Constant("input_size")
                                .set_attr_value(input_size_tensor);
    conv3dtranspose.set_input_input_size(const_data);

    conv3dtranspose.UpdateInputDesc("input_size", desc_input_size);

    delete[] input_size_data;

    conv3dtranspose.SetAttr("strides", {1, 2, 2, 2, 1});
    conv3dtranspose.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    conv3dtranspose.SetAttr("dilations", {1, 1, 1, 1, 1});
    conv3dtranspose.SetAttr("output_padding", {0, 0, 0, 0, 0});
    auto status = conv3dtranspose.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = conv3dtranspose.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// fail get input_size
TEST_F(Conv3DTransposeProtoTest, Conv3DTransposeTest1) {
    ge::op::Conv3DTranspose conv3dtranspose;
    conv3dtranspose.UpdateInputDesc("x", create_desc_with_ori({2, 2, 2, 10, 10},
      ge::DT_FLOAT16, ge::FORMAT_NDHWC, {2, 2, 2, 10, 10}, ge::FORMAT_NDHWC));
    conv3dtranspose.UpdateInputDesc("filter", create_desc_with_ori(
      {1, 2, 3, 4, 10}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,{1, 2, 3, 4, 10},
      ge::FORMAT_DHWCN));
    conv3dtranspose.UpdateOutputDesc("y", create_desc_with_ori({1, 4, 6, 8, 10},
      ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},ge::FORMAT_NDHWC));
    conv3dtranspose.UpdateOutputDesc("input_size", create_desc_with_ori(
      {1, 4, 6, 8, 10}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,{1, 4, 6, 8, 10},
      ge::FORMAT_NDHWC));

    conv3dtranspose.SetAttr("strides", {1, 2, 2, 2, 1});
    conv3dtranspose.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    conv3dtranspose.SetAttr("dilations", {1, 1, 1, 1, 1});
    conv3dtranspose.SetAttr("output_padding", {0, 0, 0, 0, 0});
    auto status = conv3dtranspose.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = conv3dtranspose.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
