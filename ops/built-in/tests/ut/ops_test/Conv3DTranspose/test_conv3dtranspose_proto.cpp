#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "nn_calculation_ops.h"
#include "utils/attr_utils.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "op_desc.h"
#include "utils/op_desc_utils.h"
#include "graph/debug/ge_attr_define.h"

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

TEST_F(Conv3DTransposeProtoTest, conv3d_transpose_dynamic_cut_info_n){
    ge::op::Conv3DTranspose op;
    op.UpdateInputDesc("input_size", create_desc_with_ori(
      {-2}, ge::DT_INT32, ge::FORMAT_ND, {-2}, ge::FORMAT_ND));
    op.UpdateInputDesc("x", create_desc_with_ori(
      {-1, 3, -1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {-1, 3, -1, -1, -1}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {8, 8, 2, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {8, 8, 2, 3, 3}, ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {-1, 3, -1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {-1, 3, -1, -1, -1}, ge::FORMAT_NCDHW));

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {1, 1, 1, 1, 1, 1});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    std::vector<std::vector<int64_t>> y_data_slice = {{0, 1}, {}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
    std::vector<std::vector<int64_t>> expect_x_data_slice = {{-1, -1}, {}, {}, {}, {}, {}};
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(expect_x_data_slice, x_data_slice);
}

TEST_F(Conv3DTransposeProtoTest, conv3d_transpose_dynamic_cut_info_d){
    ge::op::Conv3DTranspose op;
    op.UpdateInputDesc("input_size", create_desc_with_ori(
      {1, 3, 4, 4, 5}, ge::DT_INT32, ge::FORMAT_ND, {1, 3, 4, 4, 5}, ge::FORMAT_ND));
    op.UpdateInputDesc("x", create_desc_with_ori(
      {-1, 3, -1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {-1, 3, -1, -1, -1}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {8, 8, 2, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {8, 8, 2, 3, 3}, ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {-1, 3, -1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {-1, 3, -1, -1, -1}, ge::FORMAT_NCDHW));

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {1, 1, 1, 1, 1, 1});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    std::vector<std::vector<int64_t>> y_data_slice = {{}, {0, 1}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
    std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {-1, -1}, {}, {}, {}, {}};
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(expect_x_data_slice, x_data_slice);
}

TEST_F(Conv3DTransposeProtoTest, conv3d_transpose_dynamic_cut_info_h){
    ge::op::Conv3DTranspose op;
    op.UpdateInputDesc("input_size", create_desc_with_ori(
      {1, 3, 4, 4, 5}, ge::DT_INT32, ge::FORMAT_ND, {1, 3, 4, 4, 5}, ge::FORMAT_ND));
    op.UpdateInputDesc("x", create_desc_with_ori(
      {-1, 3, -1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {-1, 3, -1, -1, -1}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {8, 8, 2, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {8, 8, 2, 3, 3}, ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {-1, 3, -1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {-1, 3, -1, -1, -1}, ge::FORMAT_NCDHW));

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {1, 1, 1, 1, 1, 1});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    std::vector<std::vector<int64_t>> y_data_slice = {{}, {}, {}, {0, 1}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
    std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {}, {}, {-1, -1}, {}, {}};
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(expect_x_data_slice, x_data_slice);
}

TEST_F(Conv3DTransposeProtoTest, conv3d_transpose_dynamic_cut_info_w){
    ge::op::Conv3DTranspose op;
    op.UpdateInputDesc("input_size", create_desc_with_ori(
      {1, 3, 4, 4, 5}, ge::DT_INT32, ge::FORMAT_ND, {1, 3, 4, 4, 5}, ge::FORMAT_ND));
    op.UpdateInputDesc("x", create_desc_with_ori(
      {-1, 3, -1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {-1, 3, -1, -1, -1}, ge::FORMAT_NCDHW));
    op.UpdateInputDesc("filter", create_desc_with_ori(
      {8, 8, 2, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {8, 8, 2, 3, 3}, ge::FORMAT_DHWCN));
    op.UpdateOutputDesc("y", create_desc_with_ori(
      {-1, 3, -1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
      {-1, 3, -1, -1, -1}, ge::FORMAT_NCDHW));

    op.SetAttr("strides", {1, 1, 1, 1, 1});
    op.SetAttr("pads", {1, 1, 1, 1, 1, 1});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("output_padding", {0, 0, 0, 0, 0});
    std::vector<std::vector<int64_t>> y_data_slice = {{}, {}, {}, {}, {0, 1}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
    std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {}, {}, {}, {-1, -1}, {}};
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(expect_x_data_slice, x_data_slice);
}