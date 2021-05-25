#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_calculation_ops.h"
#include "array_ops.h"
#include "utils/attr_utils.h"
#include "utils/op_desc_utils.h"


class Conv2DBackpropInputProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv2DBackpropInput Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv2DBackpropInput Proto Test TearDown" << std::endl;
  }
};

// fix VALID Const
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputFix) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop",
                       create_desc_with_ori({1, 32, 24, 24}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 32, 24, 24}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 24, 24}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                  {1, 16, 24, 24}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("padding", "VALID");
    op.SetAttr("data_format", "NCHW");

    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{1, 16, 24, 24};
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

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dynamic hw VALID Const
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputOptiWithPads) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({1, 32, -1, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {1, 32, -1, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 1}, {32, 32}, {6, 26}, {6, 26}}));
    op.UpdateOutputDesc("y", create_desc_shape_range({1, 16, -1, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_NCHW,
                                                        {1, 16, -1, -1},
                                                        ge::FORMAT_NCHW,
                                                        {{1, 1}, {16, 16}, {6, 26}, {6, 26}}));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("padding", "VALID");
    op.SetAttr("data_format", "NCHW");

    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{1, 16, 24, 24};
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

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dynamic nwc SAME var range -1
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputDynamicNWC) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({-1, -1, 24, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, -1, 24, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 5}, {16, 32}, {14, 24}, {6, -1}}));
    op.UpdateOutputDesc("y", create_desc_shape_range({-1, 16, 24, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_NCHW,
                                                        {-1, 16, 24, -1},
                                                        ge::FORMAT_NCHW,
                                                        {{1, 5}, {16, 16}, {24, 24}, {1, -1}}));

    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("padding", "SAME");
    op.SetAttr("data_format", "NCHW");
    op.SetAttr("groups", 1);

    auto fmap_ori_shape_data = ge::op::Data("input_size");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    fmap_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    fmap_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_input_size(fmap_ori_shape_data);
    op.UpdateInputDesc("input_size", ori_tensorDesc);
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    auto input_sizes_desc = op_desc->MutableInputDesc("input_size");
    ge::AttrUtils::SetListInt(*input_sizes_desc, "_pre_op_in_range", {1, 10, 16, 32, 24, 24, 6, -1});

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// no output shape and no value range
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputNoOutputShape) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({-1, -1, 24, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, -1, 24, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 5}, {16, 32}, {14, 24}, {6, -1}}));

    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("padding", "SAME");
    op.SetAttr("data_format", "NCHW");
    op.SetAttr("groups", 1);

    auto fmap_ori_shape_data = ge::op::Data("input_size");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    fmap_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    fmap_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_input_size(fmap_ori_shape_data);
    op.UpdateInputDesc("input_size", ori_tensorDesc);
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    auto input_sizes_desc = op_desc->MutableInputDesc("input_size");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// dynamic opti ut outbackprop shape [-2]
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputSpecialShape) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({-2},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-2},
                                               ge::FORMAT_NCHW,
                                               {{}}));
    op.UpdateOutputDesc("y", create_desc_shape_range({-1, 16, -1, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_NCHW,
                                                        {-1, 16, -1, -1},
                                                        ge::FORMAT_NCHW,
                                                        {{1, 5}, {16, 16}, {1, -1}, {1, -1}}));

    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("padding", "VALID");
    op.SetAttr("data_format", "NCHW");

    auto fmap_ori_shape_data = ge::op::Data("input_size");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    fmap_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    fmap_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_input_size(fmap_ori_shape_data);
    op.UpdateInputDesc("input_size", ori_tensorDesc);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dynamic opti ut outbackprop shape [-1, c, -1, -1] with no range
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputSpecialRange) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({-1, 32, -1, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, 32, -1, -1},
                                               ge::FORMAT_NCHW,
                                               {}));
    op.UpdateOutputDesc("y", create_desc_shape_range({-1, 16, -1, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_NCHW,
                                                        {-1, 16, -1, -1},
                                                        ge::FORMAT_NCHW,
                                                        {{1, 5}, {16, 16}, {1, -1}, {1, -1}}));

    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("padding", "VALID");
    op.SetAttr("data_format", "NCHW");

    auto fmap_ori_shape_data = ge::op::Data("input_size");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    fmap_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    fmap_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_input_size(fmap_ori_shape_data);
    op.UpdateInputDesc("input_size", ori_tensorDesc);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dynamic opti ut outbackprop shape [-1, c, -1, -1] with range < 4
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputSpecialRange_1) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({-1, 32, -1, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, 32, -1, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 5}}));
    op.UpdateOutputDesc("y", create_desc_shape_range({-1, 16, -1, -1},
                                                        ge::DT_FLOAT16,
                                                        ge::FORMAT_NCHW,
                                                        {-1, 16, -1, -1},
                                                        ge::FORMAT_NCHW,
                                                        {{1, 5}, {16, 16}, {1, -1}, {1, -1}}));

    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("padding", "VALID");
    op.SetAttr("data_format", "NCHW");

    auto fmap_ori_shape_data = ge::op::Data("input_size");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    fmap_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    fmap_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_input_size(fmap_ori_shape_data);
    op.UpdateInputDesc("input_size", ori_tensorDesc);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// dynamic general ut with dilations<0
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputGeneWithDilation) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("input_size",
                       create_desc({4}, ge::DT_INT32));
    op.UpdateInputDesc("filter", create_desc({32, 16, 3, 3}, ge::DT_FLOAT16));
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({1, 32, -1, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {1, 32, -1, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 1}, {32, 32}, {1, 18}, {1, 18}}));
    op.SetAttr("dilations", {-1, -1, -1, -1});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {1, 1, 1, 1});

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// check fm type diff filter type
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputVerifyFilterTest1) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc({32, 16, 1, 1}, ge::DT_INT32));
    op.UpdateInputDesc("out_backprop", create_desc({1, 32, -1, -1}, ge::DT_FLOAT16));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// check filter out of size 4
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputVerifyFilterTest2) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc({32, 16, 1, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("out_backprop", create_desc({1, 32, -1, -1}, ge::DT_FLOAT16));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// check fm type diff out_backprop type
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputVerifyOutBackpropTest1) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc({32, 16, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("out_backprop", create_desc({1, 32, -1, -1}, ge::DT_INT32));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// check out_backprop size
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputVerifyOutBackpropTest2) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc({32, 16, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("out_backprop", create_desc({1, 32, -1, -1, 1}, ge::DT_INT8));

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// pad size 3
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputVerifyPadsTest) {
    ge::op::Conv2DBackpropInput op;
    op.UpdateInputDesc("filter", create_desc({32, 16, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("out_backprop", create_desc({1, 32, -1, -1}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// fuzzy compile
TEST_F(Conv2DBackpropInputProtoTest, conv2dbackpropinputFuzzyCompile) {
    ge::op::Conv2DBackpropInput op;
    op.SetAttr("_fuzz_build", true);
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop",
                       create_desc_with_ori({1, 32, 24, 24}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {1, 32, 24, 24}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({1, 16, 24, 24}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                  {1, 16, 24, 24}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("padding", "VALID");
    op.SetAttr("data_format", "NCHW");

    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{1, 16, 24, 24};
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

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tesor_desc_x = op_desc->MutableInputDesc("out_backprop");
    std::vector<std::pair<int64_t, int64_t>> input_range;
    tesor_desc_x->GetShapeRange(input_range);
    std::vector<std::pair<int64_t, int64_t>> expect_input_range = {{1, 2}, {32, 32}, {16, 32}, {16, 32}};
    EXPECT_EQ((input_range == expect_input_range), true);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}