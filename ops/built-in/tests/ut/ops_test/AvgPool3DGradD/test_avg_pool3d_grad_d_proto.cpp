#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"
#include "graph/utils/type_utils.h"
#include "op_desc.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/common_error_codes.h"


class AvgPool3DGradDProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AvgPool3DGradD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AvgPool3DGradD Proto Test TearDown" << std::endl;
  }
};

// Base1 pass test case
TEST_F(AvgPool3DGradDProtoTest, avgPool3DGradD_base1)
{
    ge::op::AvgPool3DGradD op;

    op.UpdateInputDesc("grads", create_desc_with_ori(
      {9, 6, 4, 14, 48}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {9, 6, 4, 14, 48}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("filter", create_desc_with_ori(
      {1, 2, 2, 1, 48}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {1, 2, 2, 1, 48}, ge::FORMAT_DHWCN));

    op.UpdateInputDesc("multiplier", create_desc_with_ori(
      {9, 6, 4, 14, 48}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {9, 6, 4, 14, 48}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("output", create_desc_with_ori(
      {9, 6, 28, 28, 48}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {9, 6, 28, 28, 48}, ge::FORMAT_NDHWC));

    op.SetAttr("orig_input_shape", {9, 6, 28, 28, 48});
    op.SetAttr("ksize", {1, 1, 2, 2, 1});
    op.SetAttr("strides", {1, 1, 9, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 1, 0, 0});
    op.SetAttr("ceil_mode", false);
    op.SetAttr("count_include_pad", false);
    op.SetAttr("divisor_override", 0);
    op.SetAttr("data_format", "NDHWC");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// Base2 pass test case
TEST_F(AvgPool3DGradDProtoTest, avgPool3DGradD_base2)
{
    ge::op::AvgPool3DGradD op;

    op.UpdateInputDesc("grads", create_desc_with_ori(
      {9, 6, 4, 14, 48}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {9, 6, 4, 14, 48}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("filter", create_desc_with_ori(
      {1, 2, 2, 1, 48}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {1, 2, 2, 1, 48}, ge::FORMAT_DHWCN));

    op.UpdateInputDesc("multiplier", create_desc_with_ori(
      {9, 6, 4, 14, 48}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {9, 6, 4, 14, 48}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("output", create_desc_with_ori(
      {9, 6, 28, 28, 48}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {9, 6, 28, 28, 48}, ge::FORMAT_NDHWC));

    op.SetAttr("orig_input_shape", {9, 6, 28, 28, 48});
    op.SetAttr("ksize", {1, 1, 2, 2, 1});
    op.SetAttr("strides", {1, 1, 9, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 1, 0, 0});
    op.SetAttr("ceil_mode", false);
    op.SetAttr("count_include_pad", false);
    op.SetAttr("divisor_override", 0);
    op.SetAttr("padding", "SAME");
    op.SetAttr("data_format", "NDHWC");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// infer data slice --- empty query
TEST_F(AvgPool3DGradDProtoTest, avgPool3DGradD_empty_slice_Failed)
{
    ge::op::AvgPool3DGradD op;

    op.UpdateInputDesc("grads", create_desc_with_ori(
      {9, 6, 4, 14, 48}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {9, 6, 4, 14, 48}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("filter", create_desc_with_ori(
      {1, 2, 2, 1, 48}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {1, 2, 2, 1, 48}, ge::FORMAT_DHWCN));

    op.UpdateInputDesc("multiplier", create_desc_with_ori(
      {9, 6, 4, 14, 48}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {9, 6, 4, 14, 48}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("output", create_desc_with_ori(
      {9, 6, 28, 28, 48}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {9, 6, 28, 28, 48}, ge::FORMAT_NDHWC));

    op.SetAttr("orig_input_shape", {9, 6, 28, 28, 48});
    op.SetAttr("ksize", {1, 1, 2, 2, 1});
    op.SetAttr("strides", {1, 1, 9, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 1, 0, 0});
    op.SetAttr("ceil_mode", false);
    op.SetAttr("count_include_pad", false);
    op.SetAttr("divisor_override", 0);
    op.SetAttr("data_format", "NDHWC");

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("output");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// infer data slice --- empty query
TEST_F(AvgPool3DGradDProtoTest, avgPool3DGradD_data_slice_query_more_than_one)
{
    ge::op::AvgPool3DGradD op;

    op.UpdateInputDesc("grads", create_desc_with_ori(
      {9, 6, 4, 14, 48}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {9, 6, 4, 14, 48}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("filter", create_desc_with_ori(
      {1, 2, 2, 1, 48}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {1, 2, 2, 1, 48}, ge::FORMAT_DHWCN));

    op.UpdateInputDesc("multiplier", create_desc_with_ori(
      {9, 6, 4, 14, 48}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {9, 6, 4, 14, 48}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("output", create_desc_with_ori(
      {9, 6, 28, 28, 48}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {9, 6, 28, 28, 48}, ge::FORMAT_NDHWC));

    op.SetAttr("orig_input_shape", {9, 6, 28, 28, 48});
    op.SetAttr("ksize", {1, 1, 2, 2, 1});
    op.SetAttr("strides", {1, 1, 9, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 1, 0, 0});
    op.SetAttr("ceil_mode", false);
    op.SetAttr("count_include_pad", false);
    op.SetAttr("divisor_override", 0);
    op.SetAttr("data_format", "NDHWC");

    std::vector<std::vector<int64_t>> y_data_slice ={{0,1}, {0,1}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("output");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// infer data slice --- cut N
TEST_F(AvgPool3DGradDProtoTest, avgPool3DGradD_data_slice_cut_n_infer)
{
    ge::op::AvgPool3DGradD op;

    op.UpdateInputDesc("grads", create_desc_with_ori(
      {9, 6, 4, 14, 48}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {9, 6, 4, 14, 48}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("filter", create_desc_with_ori(
      {1, 2, 2, 1, 48}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {1, 2, 2, 1, 48}, ge::FORMAT_DHWCN));

    op.UpdateInputDesc("multiplier", create_desc_with_ori(
      {9, 6, 4, 14, 48}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {9, 6, 4, 14, 48}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("output", create_desc_with_ori(
      {9, 6, 28, 28, 48}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {9, 6, 28, 28, 48}, ge::FORMAT_NDHWC));

    op.SetAttr("orig_input_shape", {9, 6, 28, 28, 48});
    op.SetAttr("ksize", {1, 1, 2, 2, 1});
    op.SetAttr("strides", {1, 1, 9, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 1, 0, 0});
    op.SetAttr("ceil_mode", false);
    op.SetAttr("count_include_pad", false);
    op.SetAttr("divisor_override", 0);
    op.SetAttr("data_format", "NDHWC");

    std::vector<std::vector<int64_t>> y_data_slice ={{0,1}, {}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("output");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_dedy = op_desc->MutableInputDesc("grads");
    std::vector<std::vector<int64_t>> dedy_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_dedy, ge::ATTR_NAME_DATA_SLICE, dedy_data_slice);
    std::vector<std::vector<int64_t>> expect_dedy_data_slice = {{0,1}, {}, {}, {}, {}, {}};
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(expect_dedy_data_slice, dedy_data_slice);
}

// infer data slice --- cut D
TEST_F(AvgPool3DGradDProtoTest, avgPool3DGradD_data_slice_cut_d_infer)
{
    ge::op::AvgPool3DGradD op;

    op.UpdateInputDesc("grads", create_desc_with_ori(
      {9, 6, 4, 14, 48}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {9, 6, 4, 14, 48}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("filter", create_desc_with_ori(
      {1, 2, 2, 1, 48}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {1, 2, 2, 1, 48}, ge::FORMAT_DHWCN));

    op.UpdateInputDesc("multiplier", create_desc_with_ori(
      {9, 6, 4, 14, 48}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {9, 6, 4, 14, 48}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("output", create_desc_with_ori(
      {9, 6, 28, 28, 48}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {9, 6, 28, 28, 48}, ge::FORMAT_NDHWC));

    op.SetAttr("orig_input_shape", {9, 6, 28, 28, 48});
    op.SetAttr("ksize", {1, 1, 2, 2, 1});
    op.SetAttr("strides", {1, 1, 9, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 1, 0, 0});
    op.SetAttr("ceil_mode", false);
    op.SetAttr("count_include_pad", false);
    op.SetAttr("divisor_override", 0);
    op.SetAttr("data_format", "NDHWC");

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {0,1}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("output");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_dedy = op_desc->MutableInputDesc("grads");
    std::vector<std::vector<int64_t>> dedy_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_dedy, ge::ATTR_NAME_DATA_SLICE, dedy_data_slice);
    std::vector<std::vector<int64_t>> expect_dedy_data_slice = {{}, {0,1}, {}, {}, {}, {}};
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(expect_dedy_data_slice, dedy_data_slice);
}

// infer data slice --- cut H
TEST_F(AvgPool3DGradDProtoTest, avgPool3DGradD_data_slice_cut_h_infer)
{
    ge::op::AvgPool3DGradD op;

    op.UpdateInputDesc("grads", create_desc_with_ori(
      {9, 6, 4, 14, 48}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {9, 6, 4, 14, 48}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("filter", create_desc_with_ori(
      {1, 2, 2, 1, 48}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {1, 2, 2, 1, 48}, ge::FORMAT_DHWCN));

    op.UpdateInputDesc("multiplier", create_desc_with_ori(
      {9, 6, 4, 14, 48}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {9, 6, 4, 14, 48}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("output", create_desc_with_ori(
      {9, 6, 28, 28, 48}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {9, 6, 28, 28, 48}, ge::FORMAT_NDHWC));

    op.SetAttr("orig_input_shape", {9, 6, 28, 28, 48});
    op.SetAttr("ksize", {1, 1, 2, 2, 1});
    op.SetAttr("strides", {1, 1, 9, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 1, 0, 0});
    op.SetAttr("ceil_mode", false);
    op.SetAttr("count_include_pad", false);
    op.SetAttr("divisor_override", 0);
    op.SetAttr("data_format", "NDHWC");

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {0,1}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("output");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_dedy = op_desc->MutableInputDesc("grads");
    std::vector<std::vector<int64_t>> dedy_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_dedy, ge::ATTR_NAME_DATA_SLICE, dedy_data_slice);
    std::vector<std::vector<int64_t>> expect_dedy_data_slice = {{}, {}, {}, {0,0}, {}, {}};
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(expect_dedy_data_slice, dedy_data_slice);
}

// infer data slice --- cut W
TEST_F(AvgPool3DGradDProtoTest, avgPool3DGradD_data_slice_cut_w_infer)
{
    ge::op::AvgPool3DGradD op;

    op.UpdateInputDesc("grads", create_desc_with_ori(
      {9, 6, 4, 14, 48}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {9, 6, 4, 14, 48}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("filter", create_desc_with_ori(
      {1, 2, 2, 1, 48}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {1, 2, 2, 1, 48}, ge::FORMAT_DHWCN));

    op.UpdateInputDesc("multiplier", create_desc_with_ori(
      {9, 6, 4, 14, 48}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {9, 6, 4, 14, 48}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("output", create_desc_with_ori(
      {9, 6, 28, 28, 48}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {9, 6, 28, 28, 48}, ge::FORMAT_NDHWC));

    op.SetAttr("orig_input_shape", {9, 6, 28, 28, 48});
    op.SetAttr("ksize", {1, 1, 2, 2, 1});
    op.SetAttr("strides", {1, 1, 9, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 1, 0, 0});
    op.SetAttr("ceil_mode", false);
    op.SetAttr("count_include_pad", false);
    op.SetAttr("divisor_override", 0);
    op.SetAttr("data_format", "NDHWC");

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {}, {0,1}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("output");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_dedy = op_desc->MutableInputDesc("grads");
    std::vector<std::vector<int64_t>> dedy_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_dedy, ge::ATTR_NAME_DATA_SLICE, dedy_data_slice);
    std::vector<std::vector<int64_t>> expect_dedy_data_slice = {{}, {}, {}, {}, {0,0}, {}};
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(expect_dedy_data_slice, dedy_data_slice);
}

// infer data slice --- global
TEST_F(AvgPool3DGradDProtoTest, avgPool3DGradD_data_slice_cut_global)
{
    ge::op::AvgPool3DGradD op;
    op.UpdateInputDesc("grads", create_desc_with_ori(
      {9, 1, 1, 1, 48}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {9, 1, 1, 1, 48}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("filter", create_desc_with_ori(
      {1, 2, 2, 1, 48}, ge::DT_FLOAT16, ge::FORMAT_DHWCN,
      {1, 2, 2, 1, 48}, ge::FORMAT_DHWCN));

    op.UpdateInputDesc("multiplier", create_desc_with_ori(
      {9, 1, 1, 1, 48}, ge::DT_FLOAT, ge::FORMAT_NDHWC,
      {9, 1, 1, 1, 48}, ge::FORMAT_NDHWC));

    op.UpdateInputDesc("output", create_desc_with_ori(
      {9, 1, 2, 2, 48}, ge::DT_FLOAT16, ge::FORMAT_NDHWC,
      {9, 1, 2, 2, 48}, ge::FORMAT_NDHWC));

    op.SetAttr("orig_input_shape", {9, 1, 2, 2, 48});
    op.SetAttr("ksize", {1, 1, 2, 2, 1});
    op.SetAttr("strides", {1, 1, 9, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 2, 0, 0});
    op.SetAttr("ceil_mode", false);
    op.SetAttr("count_include_pad", false);
    op.SetAttr("divisor_override", 0);
    op.SetAttr("data_format", "NDHWC");

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {}, {0,1}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("output");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::NOT_SUPPORT_SLICE);
}
TEST_F(AvgPool3DGradDProtoTest, VerifyAvgPool3DGradD_001) {
  ge::op::AvgPool3DGradD op;
  std::vector<int64_t> orig_input_shape = {9, 1, 2, 2};
  op.SetAttr("orig_input_shape", orig_input_shape);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DGradDProtoTest, VerifyAvgPool3DGradD_002) {
  ge::op::AvgPool3DGradD op;
  std::vector<int64_t> orig_input_shape = {9, 1, 2, 2, 48};
  std::vector<int64_t> ksize = {1, 1, 2, 2};
  op.SetAttr("orig_input_shape", orig_input_shape);
  op.SetAttr("ksize", ksize);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DGradDProtoTest, VerifyAvgPool3DGradD_003) {
  ge::op::AvgPool3DGradD op;
  std::vector<int64_t> orig_input_shape = {9, 1, 2, 2, 48};
  std::vector<int64_t> ksize = {1, 1, 2, 2, 1};
  std::vector<int64_t> strides = {1, 1, 9, 2};
  op.SetAttr("orig_input_shape", orig_input_shape);
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DGradDProtoTest, VerifyAvgPool3DGradD_004) {
  ge::op::AvgPool3DGradD op;
  std::vector<int64_t> orig_input_shape = {9, 1, 2, 2, 48};
  std::vector<int64_t> ksize = {1, 1, 2, 2, 1};
  std::vector<int64_t> strides = {1, 1, 9, 2, 1};
  op.SetAttr("orig_input_shape", orig_input_shape);
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("pads", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DGradDProtoTest, VerifyAvgPool3DGradD_005) {
  ge::op::AvgPool3DGradD op;
  std::vector<int64_t> orig_input_shape = {9, 1, 2, 2, 48};
  std::vector<int64_t> ksize = {1, 1, 2, 2, 1};
  std::vector<int64_t> strides = {1, 1, 9, 2, 1};
  std::vector<int64_t> pads = {0, 0, 0, 2, 0};
  op.SetAttr("orig_input_shape", orig_input_shape);
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("pads", pads);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DGradDProtoTest, VerifyAvgPool3DGradD_006) {
  ge::op::AvgPool3DGradD op;
  std::vector<int64_t> orig_input_shape = {9, 1, 2, 2, 48};
  std::vector<int64_t> ksize = {1, 1, 2, 2, 1};
  std::vector<int64_t> strides = {1, 1, 9, 2, 1};
  std::vector<int64_t> pads = {0, 0, 0, 2, 0, 0};
  op.SetAttr("orig_input_shape", orig_input_shape);
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("pads", pads);
  op.SetAttr("data_format", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DGradDProtoTest, VerifyAvgPool3DGradD_007) {
  ge::op::AvgPool3DGradD op;
  std::vector<int64_t> orig_input_shape = {9, 1, 2, 2, 48};
  std::vector<int64_t> ksize = {1, 1, 2, 2, 1};
  std::vector<int64_t> strides = {1, 1, 9, 2, 1};
  std::vector<int64_t> pads = {0, 0, 0, 2, 0, 0};
  op.SetAttr("orig_input_shape", orig_input_shape);
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("pads", pads);
  op.SetAttr("data_format", "ND");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DGradDProtoTest, InfershapeAvgPool3DGradD_001) {
  ge::op::AvgPool3DGradD op;
  std::vector<int64_t> orig_input_shape = {9, 1, 2, 2};
  op.SetAttr("orig_input_shape", orig_input_shape);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DGradDProtoTest, InfershapeAvgPool3DGradD_002) {
  ge::op::AvgPool3DGradD op;
  std::vector<int64_t> orig_input_shape = {9, 1, 2, 2, 48};
  std::vector<int64_t> ksize = {};
  op.SetAttr("orig_input_shape", orig_input_shape);
  op.SetAttr("ksize", ksize);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DGradDProtoTest, InfershapeAvgPool3DGradD_003) {
  ge::op::AvgPool3DGradD op;
  std::vector<int64_t> orig_input_shape = {9, 1, 2, 2, 48};
  std::vector<int64_t> ksize = {1, 1, 2, 2, 1};
  std::vector<int64_t> strides = {};
  op.SetAttr("orig_input_shape", orig_input_shape);
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DGradDProtoTest, InfershapeAvgPool3DGradD_004) {
  ge::op::AvgPool3DGradD op;
  std::vector<int64_t> orig_input_shape = {9, 1, 2, 2, 48};
  std::vector<int64_t> ksize = {1, 1, 2, 2, 1};
  std::vector<int64_t> strides = {1, 1, 9, 2, 1};
  op.SetAttr("orig_input_shape", orig_input_shape);
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("pads", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DGradDProtoTest, InfershapeAvgPool3DGradD_005) {
  ge::op::AvgPool3DGradD op;
  std::vector<int64_t> orig_input_shape = {9, 1, 2, 2, 48};
  std::vector<int64_t> ksize = {1, 1, 2, 2, 1};
  std::vector<int64_t> strides = {1, 1, 9, 2, 1};
  op.SetAttr("orig_input_shape", orig_input_shape);
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("pads", true);
  op.SetAttr("data_format", strides);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DGradDProtoTest, InfershapeAvgPool3DGradD_006) {
  ge::op::AvgPool3DGradD op;
  std::vector<int64_t> orig_input_shape = {9, 1, 2, 2, 48};
  std::vector<int64_t> ksize = {1, 1, 2, 2, 1};
  std::vector<int64_t> strides = {1, 1, 9, 2, 1};
  op.SetAttr("orig_input_shape", orig_input_shape);
  op.SetAttr("ksize", ksize);
  op.SetAttr("strides", strides);
  op.SetAttr("pads", true);
  op.SetAttr("data_format", "ND");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DGradDProtoTest, InfershapeAvgPool3DGradD_007) {
  ge::op::AvgPool3DGradD op;
  op.UpdateInputDesc("grads", create_desc_with_ori({9, 6, 4, 14, 48}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                   {9, 6, 4, 14, 48}, ge::FORMAT_NCDHW));
  op.UpdateInputDesc("filter", create_desc_with_ori({1, 2, 2, 1, 48}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                    {1, 2, 2, 1, 48}, ge::FORMAT_NCDHW));
  op.UpdateInputDesc("multiplier", create_desc_with_ori({9, 6, 4, 14, 48}, ge::DT_FLOAT, ge::FORMAT_NCDHW,
                                                        {9, 6, 4, 14, 48}, ge::FORMAT_NCDHW));
  op.UpdateInputDesc("output", create_desc_with_ori({9, 6, 28, 28, 48}, ge::DT_FLOAT16, ge::FORMAT_NCDHW,
                                                    {9, 6, 28, 28, 48}, ge::FORMAT_NCDHW));
  op.SetAttr("orig_input_shape", {9, 6, 28, 28, 48});
  op.SetAttr("ksize", {1, 1, 2, 2, 1});
  op.SetAttr("strides", {1, 1, 9, 2, 1});
  op.SetAttr("pads", {0, 0, 0, 1, 0, 0});
  op.SetAttr("ceil_mode", false);
  op.SetAttr("count_include_pad", false);
  op.SetAttr("divisor_override", 0);
  op.SetAttr("data_format", "NCDHW");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(AvgPool3DGradDProtoTest, DatasliceAvgPool3DGradD_001) {
  ge::op::AvgPool3DGradD op;
  op.SetAttr("data_format", "NDHWC");
  op.SetAttr("strides", true);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DGradDProtoTest, DatasliceAvgPool3DGradD_002) {
  ge::op::AvgPool3DGradD op;
  std::vector<int32_t> stride_list = {1, 1, 9, 2};
  op.SetAttr("data_format", "NDHWC");
  op.SetAttr("strides", stride_list);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DGradDProtoTest, DatasliceAvgPool3DGradD_003) {
  ge::op::AvgPool3DGradD op;
  std::vector<int32_t> stride_list = {1, -1, -9, 2, 8};
  op.SetAttr("data_format", "NDHWC");
  op.SetAttr("strides", stride_list);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}