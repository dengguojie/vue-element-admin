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


// ---------------Conv2DTranspose-------------------
class Conv2DTransposeProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv2DTranspose Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv2DTranspose Proto Test TearDown" << std::endl;
  }
};

// base ut
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyBaseTest) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{128, 256, 14, 14};
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

// strict static with input_size all zero
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyInputSizeAllZero) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{0, 0, 0, 0};
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

// dynamic nwc ut
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeDynamicNWC) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("x",
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
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

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
    std::vector<std::pair<int64_t, int64_t>> value_range = {{1, 10}, {16, 32}, {24, 24}, {6, -1}};
    input_sizes_desc->SetValueRange(value_range);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dynamic opti ut outbackprop shape [-1, c, -1, -1] with no range
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeSpecialRange) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("x",
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
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);
    op.SetAttr("groups", 1);
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
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    std::vector<std::pair<int64_t, int64_t>> output_range;
    tensor_desc_y->GetShapeRange(output_range);
    std::vector<std::pair<int64_t, int64_t>> expect_y_range = {{1, -1}, {16, 16}, {1, -1}, {1, -1}};
    EXPECT_EQ((output_range == expect_y_range), true);
}

// dynamic opti ut outbackprop shape [-1, c, -1, -1] with range < 4
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeSpecialRange_1) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("x",
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
    op.SetAttr("groups", 1);
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);
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

// dynamic opti ut outbackprop shape [-2]
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeDynamicRank) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("filter", create_desc_with_ori({32, 16, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                            {32, 16, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("x",
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
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

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

//check filter out of size 4
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyfilterTest2) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// check x size
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyInputTest1) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// no stride
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyStrideTest1) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    //   op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// stride out of size 4
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyStrideTest2) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// dilations out of size 4
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyDilationsTest2) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// no pad
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyPadsTest1) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    //   op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// pad size 3
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyPadsTest2) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// pad out of -1
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyPadsTest3) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, -1});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// no output_padding
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyOutputpaddingTest1) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");
    //   op.SetAttr("output_padding", {0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// output_padding out of size 4
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyOutputpaddingTest2) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// input size not 4
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeVerifyInputSizesTest2) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("filter", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128, 256, 14, 14}, ge::DT_FLOAT16));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}


// data slice
// cut N
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeSplicDataTest0) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 64, 12, 12}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 64, 12, 12},ge::FORMAT_NCHW));
    op.SetAttr("input_size", {4, 64, 12, 12});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    std::vector<std::vector<int64_t>> y_data_slice ={{0, 2}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();

    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
    std::vector<std::vector<int64_t>> expect_x_data_slice = {{0, 2}, {}, {}, {}, {}};
    EXPECT_EQ(expect_x_data_slice, x_data_slice);
}

// cut Cin
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeSplicDataTest1) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 64, 12, 12}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 64, 12, 12},ge::FORMAT_NCHW));
    op.SetAttr("input_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {0, 2}, {}, {}, {}};
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

// cut H
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeSplicDataTest2) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 10, 10}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 64, 10, 10}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 64, 12, 12}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 64, 12, 12},ge::FORMAT_NCHW));
    op.SetAttr("input_size", {4, 64, 12, 12});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {0, 5}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();

    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
    std::vector<std::int64_t> new_pads;
    op.GetAttr("pads", new_pads);
    std::vector<std::int64_t> new_pads_expect = {0, 2, 0, 0};
    std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {}, {0, 5}, {}, {}};
    EXPECT_EQ(expect_x_data_slice, x_data_slice);
    EXPECT_EQ(new_pads_expect, new_pads);
}

// cut w
TEST_F(Conv2DTransposeProtoTest, conv2dTransposeSplicDataTest3) {
    ge::op::Conv2DTranspose op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 5, 5}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 64, 5, 5}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("filter", create_desc_with_ori({64, 64, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{64, 64, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 64, 13, 13}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 64, 13, 13},ge::FORMAT_NCHW));
    op.SetAttr("input_size", {4, 64, 13, 13});
    op.SetAttr("strides", {1, 1, 3, 3});
    op.SetAttr("pads", {1, 1, 1, 1});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    op.SetAttr("offset_x", 0);

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {3, 10}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();

    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
    std::vector<std::int64_t> new_pads;
    op.GetAttr("pads", new_pads);
    std::vector<std::int64_t> new_pads_expect = {1, 1, 1, 0};
    std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {}, {}, {1, 3}, {}};
    EXPECT_EQ(expect_x_data_slice, x_data_slice);
    EXPECT_EQ(new_pads_expect, new_pads);
}