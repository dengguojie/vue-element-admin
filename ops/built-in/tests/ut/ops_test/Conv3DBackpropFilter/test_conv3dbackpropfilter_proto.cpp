#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "nn_calculation_ops.h"

class Conv3DBackpropFilterProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv3DBackpropFilter Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv3DBackpropFilter Proto Test TearDown" << std::endl;
  }
};
// Base_Pass_Case
TEST_F(Conv3DBackpropFilterProtoTest, Base_Pass_Case){
    ge::op::Conv3DBackpropFilter op;

    op.UpdateInputDesc("x", create_desc_with_ori(
      {64, 2, 2, 2, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {64, 2, 2, 2, 32}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 2, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 3, 3, 32}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    // Set Const node
    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{64, 2, 2, 2, 32};
    ge::TensorDesc tensor_desc_filter_size(ge::Shape(),
      ge::FORMAT_NDHWC, ge::DT_INT32);
    int element_size = 5;

    tensor_desc_filter_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_filter_size);
    int *conv_filter_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_filter_size_tensor_value + i) = 0;
    }
    constTensor.SetData((uint8_t *) conv_filter_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("filter_size").set_attr_value(constTensor);
    op.set_input_filter_size(const0);

    op.UpdateInputDesc("filter_size", tensor_desc_filter_size);

    delete[] conv_filter_size_tensor_value;

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// No_Filter_Size_Failed
TEST_F(Conv3DBackpropFilterProtoTest, No_Filter_Size_Failed){
    ge::op::Conv3DBackpropFilter op;

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
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// x dtypes should equal to out_backprop's dtype
TEST_F(Conv3DBackpropFilterProtoTest, Conv3DBackpropFilterBaseTest1){
    ge::op::Conv3DBackpropFilter op;

    op.UpdateInputDesc("x", create_desc_with_ori(
      {64, 2, 2, 2, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {64, 2, 2, 2, 32}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 2, 3, 3, 32}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {16, 2, 3, 3, 32}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    // Set Const node
    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{64, 2, 2, 2, 32};
    ge::TensorDesc tensor_desc_filter_size(ge::Shape(),
      ge::FORMAT_NDHWC, ge::DT_INT32);
    int element_size = 5;

    tensor_desc_filter_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_filter_size);
    int *conv_filter_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_filter_size_tensor_value + i) = 0;
    }
    constTensor.SetData((uint8_t *) conv_filter_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("filter_size").set_attr_value(constTensor);
    op.set_input_filter_size(const0);

    op.UpdateInputDesc("filter_size", tensor_desc_filter_size);

    delete[] conv_filter_size_tensor_value;

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// x shape should be 5d
TEST_F(Conv3DBackpropFilterProtoTest, Conv3DBackpropFilterBaseTest2){
    ge::op::Conv3DBackpropFilter op;

    op.UpdateInputDesc("x", create_desc_with_ori(
      {64, 2, 2, 2}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {64, 2, 2, 2}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 2, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 3, 3, 32}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    // Set Const node
    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{64, 2, 2, 2, 32};
    ge::TensorDesc tensor_desc_filter_size(ge::Shape(),
      ge::FORMAT_NDHWC, ge::DT_INT32);
    int element_size = 5;

    tensor_desc_filter_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_filter_size);
    int *conv_filter_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_filter_size_tensor_value + i) = 0;
    }
    constTensor.SetData((uint8_t *) conv_filter_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("filter_size").set_attr_value(constTensor);
    op.set_input_filter_size(const0);

    op.UpdateInputDesc("filter_size", tensor_desc_filter_size);

    delete[] conv_filter_size_tensor_value;

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// out_backprop shape should be 5d
TEST_F(Conv3DBackpropFilterProtoTest, Conv3DBackpropFilterBaseTest3){
    ge::op::Conv3DBackpropFilter op;

    op.UpdateInputDesc("x", create_desc_with_ori(
      {64, 2, 2, 2, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {64, 2, 2, 2, 32}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 2, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 3, 3}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    // Set Const node
    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{64, 2, 2, 2, 32};
    ge::TensorDesc tensor_desc_filter_size(ge::Shape(),
      ge::FORMAT_NDHWC, ge::DT_INT32);
    int element_size = 5;

    tensor_desc_filter_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_filter_size);
    int *conv_filter_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_filter_size_tensor_value + i) = 0;
    }
    constTensor.SetData((uint8_t *) conv_filter_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("filter_size").set_attr_value(constTensor);
    op.set_input_filter_size(const0);

    op.UpdateInputDesc("filter_size", tensor_desc_filter_size);

    delete[] conv_filter_size_tensor_value;

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// strides shape should be 5d
TEST_F(Conv3DBackpropFilterProtoTest, Conv3DBackpropFilterBaseTest4){
    ge::op::Conv3DBackpropFilter op;

    op.UpdateInputDesc("x", create_desc_with_ori(
      {64, 2, 2, 2, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {64, 2, 2, 2, 32}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 2, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 3, 3, 32}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    // Set Const node
    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{64, 2, 2, 2, 32};
    ge::TensorDesc tensor_desc_filter_size(ge::Shape(),
      ge::FORMAT_NDHWC, ge::DT_INT32);
    int element_size = 5;

    tensor_desc_filter_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_filter_size);
    int *conv_filter_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_filter_size_tensor_value + i) = 0;
    }
    constTensor.SetData((uint8_t *) conv_filter_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("filter_size").set_attr_value(constTensor);
    op.set_input_filter_size(const0);

    op.UpdateInputDesc("filter_size", tensor_desc_filter_size);

    delete[] conv_filter_size_tensor_value;

    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// get strides list failed
TEST_F(Conv3DBackpropFilterProtoTest, Conv3DBackpropFilterBaseTest5){
    ge::op::Conv3DBackpropFilter op;

    op.UpdateInputDesc("x", create_desc_with_ori(
      {64, 2, 2, 2, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {64, 2, 2, 2, 32}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 2, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 3, 3, 32}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    // Set Const node
    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{64, 2, 2, 2, 32};
    ge::TensorDesc tensor_desc_filter_size(ge::Shape(),
      ge::FORMAT_NDHWC, ge::DT_INT32);
    int element_size = 5;

    tensor_desc_filter_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_filter_size);
    int *conv_filter_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_filter_size_tensor_value + i) = 0;
    }
    constTensor.SetData((uint8_t *) conv_filter_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("filter_size").set_attr_value(constTensor);
    op.set_input_filter_size(const0);

    op.UpdateInputDesc("filter_size", tensor_desc_filter_size);

    delete[] conv_filter_size_tensor_value;

    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// padding is SAME in dynamic_shape
TEST_F(Conv3DBackpropFilterProtoTest, Conv3DBackpropFilterBaseTest7){
    ge::op::Conv3DBackpropFilter op;

    op.UpdateInputDesc("x", create_desc_with_ori(
      {-2}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {-2}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 2, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 3, 3, 32}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    // Set Const node
    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{64, 2, 2, 2, 32};
    ge::TensorDesc tensor_desc_filter_size(ge::Shape(),
      ge::FORMAT_NDHWC, ge::DT_INT32);
    int element_size = 5;

    tensor_desc_filter_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_filter_size);
    int *conv_filter_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_filter_size_tensor_value + i) = 0;
    }
    constTensor.SetData((uint8_t *) conv_filter_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("filter_size").set_attr_value(constTensor);
    op.set_input_filter_size(const0);

    op.UpdateInputDesc("filter_size", tensor_desc_filter_size);

    delete[] conv_filter_size_tensor_value;

    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("padding", "SAME");

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// padding is VALID in dynamic_shape
TEST_F(Conv3DBackpropFilterProtoTest, Conv3DBackpropFilterBaseTest8){
    ge::op::Conv3DBackpropFilter op;

    op.UpdateInputDesc("x", create_desc_with_ori(
      {-2}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {-2}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori(
      {16, 2, 3, 3, 32}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {16, 2, 3, 3, 32}, ge::FORMAT_NCDHW));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {2, 16, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {2, 16, 2, 16, 16}, ge::FORMAT_NCDHW));
    // Set Const node
    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{64, 2, 2, 2, 32};
    ge::TensorDesc tensor_desc_filter_size(ge::Shape(),
      ge::FORMAT_NDHWC, ge::DT_INT32);
    int element_size = 5;

    tensor_desc_filter_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_filter_size);
    int *conv_filter_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_filter_size_tensor_value + i) = 0;
    }
    constTensor.SetData((uint8_t *) conv_filter_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("filter_size").set_attr_value(constTensor);
    op.set_input_filter_size(const0);

    op.UpdateInputDesc("filter_size", tensor_desc_filter_size);

    delete[] conv_filter_size_tensor_value;

    op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("padding", "VALID");

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}