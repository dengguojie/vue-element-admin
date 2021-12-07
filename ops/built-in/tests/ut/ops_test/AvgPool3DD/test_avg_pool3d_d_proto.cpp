#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"
#include "utils/op_desc_utils.h"
#include "graph/debug/ge_attr_define.h"

// ----------------AvgPool3DD-------------------
class AvgPool3DDProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AvgPool3DD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AvgPool3DD Proto Test TearDown" << std::endl;
  }
};

TEST_F(AvgPool3DDProtoTest, apply_avg_pool3d_d_verify_test) {
  ge::op::AvgPool3DD op;
  op.UpdateInputDesc("x", create_desc({1, 4, 7, 7, 1024}, ge::DT_FLOAT16));
  op.SetAttr("ksize", {1,2,7,7,1,1});
  op.SetAttr("strides",{1,1,1,1,1,1});
  op.SetAttr("pads",{0,0,0,0,0,0});
  op.SetAttr("ceil_mode",false);
  op.SetAttr("count_include_pad",true);
  op.SetAttr("divisor_override",0);
  op.SetAttr("data_format","NDHWC");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DDProtoTest, InferDataSliceAvgPool3DD_001) {
  ge::op::AvgPool3DD op;
  op.UpdateInputDesc("x", create_desc({1, 4, 7, 7, 1024}, ge::DT_FLOAT16));
  op.SetAttr("data_format", true);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DDProtoTest, InferDataSliceAvgPool3DD_002) {
  ge::op::AvgPool3DD op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 4, 7, 7, 1024}, ge::DT_FLOAT16, ge::FORMAT_DHWCN, {1, 4, 7, 7, 1024},
                                               ge::FORMAT_DHWCN));

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DDProtoTest, InferDataSliceAvgPool3DD_003) {
  ge::op::AvgPool3DD op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 4, 7, 7, 1024}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 4, 7, 7, 1024}, ge::FORMAT_ND));
  op.SetAttr("data_format", "NCDHW");
  op.SetAttr("strides", false);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DDProtoTest, InferDataSliceAvgPool3DD_004) {
  ge::op::AvgPool3DD op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 4, 7, 7, 1024}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 4, 7, 7, 1024}, ge::FORMAT_ND));
  op.SetAttr("data_format", "NCDHW");
  op.SetAttr("strides", {1, 1, 1, 1, 1, 1});
  op.SetAttr("ksize", true);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DDProtoTest, InferDataSliceAvgPool3DD_005) {
  ge::op::AvgPool3DD op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 4, 7, 7, 1024}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 4, 7, 7, 1024}, ge::FORMAT_ND));
  op.SetAttr("data_format", "NCDHW");
  op.SetAttr("ksize", {1});
  op.SetAttr("strides", {1});
  op.SetAttr("pads", true);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DDProtoTest, InferDataSliceAvgPool3DD_006) {
  ge::op::AvgPool3DD op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 4, 7, 7, 1024}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 4, 7, 7, 1024}, ge::FORMAT_ND));
  op.SetAttr("data_format", "NCDHW");
  op.SetAttr("ksize", {1, 2, 7});
  op.SetAttr("strides", {1, 1, 1});
  op.SetAttr("pads", true);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DDProtoTest, InferDataSliceAvgPool3DD_007) {
  ge::op::AvgPool3DD op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 4, 7, 7, 1024}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 4, 7, 7, 1024}, ge::FORMAT_ND));
  op.SetAttr("data_format", "NCDHW");
  op.SetAttr("ksize", {1, 2, 7});
  op.SetAttr("strides", {1, 1, 1});
  op.SetAttr("pads", {0, 0, 0});

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DDProtoTest, InferDataSliceAvgPool3DD_008) {
  ge::op::AvgPool3DD op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 4, 7, 7, 1024}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 4, 7, 7, 1024}, ge::FORMAT_ND));
  op.SetAttr("data_format", "NCDHW");
  op.SetAttr("ksize", {1, 2, 7});
  op.SetAttr("strides", {1, 1, 1});
  op.SetAttr("pads", {0, 0, 0, 0, 0, -1});

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DDProtoTest, InferDataSliceAvgPool3DD_009) {
  ge::op::AvgPool3DD op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 4, 7, 7, 1024}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 4, 7, 7, 1024}, ge::FORMAT_ND));
  op.SetAttr("data_format", "NCDHW");
  op.SetAttr("ksize", {1, 2, 7});
  op.SetAttr("strides", {1, 1, 1});
  op.SetAttr("padding", "VALID");
  op.SetAttr("pads", {0, 0, 0, 0, 0, -1});

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DDProtoTest, InferDataSliceAvgPool3DD_010) {
  ge::op::AvgPool3DD op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 4, 7, 7, 1024}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 4, 7, 7, 1024}, ge::FORMAT_ND));
  op.SetAttr("data_format", "NCDHW");
  op.SetAttr("ksize", {1, 2, 7});
  op.SetAttr("strides", {1, 1, 1});
  op.SetAttr("padding", "SAME");
  op.SetAttr("pads", {});

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DDProtoTest, InferDataSliceAvgPool3DD_011) {
  ge::op::AvgPool3DD op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 4, 7, 7, 1024}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 4, 7, 7, 1024}, ge::FORMAT_ND));
  op.SetAttr("data_format", "NCDHW");
  op.SetAttr("ksize", {1, 2, 7});
  op.SetAttr("strides", {1, 1, 1});
  op.SetAttr("padding", "zero");

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DDProtoTest, InferDataSliceAvgPool3DD_012) {
  ge::op::AvgPool3DD op;
  // set pooling attr
  std::vector<int32_t> pad_list = {1, 0, 0, 1};
  op.SetAttr("pads", pad_list);
  op.SetAttr("ksize", {1, 2, 7});
  std::vector<int64_t> stride = {1, 1, 1};
  op.SetAttr("strides", stride);
  op.SetAttr("global_pooling", false);
  op.SetAttr("padding", "SAME");

  auto tensor_desc = create_desc_with_ori({1,1,224,224,16}, ge::DT_FLOAT16, ge::FORMAT_ND, {1,3,224,224}, ge::FORMAT_ND);
  op.UpdateInputDesc("x", tensor_desc);

  std::vector<std::vector<int64_t>> output_data_slice ={{}, {10, 20}, {}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(output_tensor_desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(AvgPool3DDProtoTest, InferDataSliceAvgPool3DD_013) {
  ge::op::AvgPool3DD op;
  // set pooling attr
  std::vector<int32_t> pad_list = {1, 0, 0, 1};
  op.SetAttr("pads", pad_list);
  op.SetAttr("ksize", {1, 2, 7});
  std::vector<int64_t> stride = {1, 1, 1};
  op.SetAttr("strides", stride);
  op.SetAttr("global_pooling", false);
  op.SetAttr("padding", "SAME");

  auto tensor_desc = create_desc_with_ori({1,1,224,224,16}, ge::DT_FLOAT16, ge::FORMAT_ND, {1,3,224,224}, ge::FORMAT_ND);
  op.UpdateInputDesc("x", tensor_desc);

  std::vector<std::vector<int64_t>> output_data_slice ={{}, {}, {}, {10, 20}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(output_tensor_desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(AvgPool3DDProtoTest, InfershapeAvgPool3DD_001) {
ge::op::AvgPool3DD op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 4, 7, 7, 1024}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 4, 7, 7, 1024}, ge::FORMAT_ND));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DDProtoTest, InfershapeAvgPool3DD_002) {
ge::op::AvgPool3DD op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 4, 7, 7, 1024}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 4, 7, 7, 1024}, ge::FORMAT_NDHWC));
  op.SetAttr("strides", {1});
  op.SetAttr("ksize", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DDProtoTest, InfershapeAvgPool3DD_003) {
ge::op::AvgPool3DD op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 4, 7, 7, 1024}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 4, 7, 7, 1024}, ge::FORMAT_NDHWC));
  op.SetAttr("strides", {0});
  op.SetAttr("ksize", {0});

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DDProtoTest, InfershapeAvgPool3DD_004) {
ge::op::AvgPool3DD op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 4, 7, 7, 1024}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 4, 7, 7, 1024}, ge::FORMAT_NDHWC));
  op.SetAttr("strides", {1});
  op.SetAttr("ksize", {1});
  op.SetAttr("padding", "VALID");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_y_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {1, 4, 7, 7, 1024};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT16);
}

TEST_F(AvgPool3DDProtoTest, InfershapeAvgPool3DD_005) {
ge::op::AvgPool3DD op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 4, 7, 7, 1024}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 4, 7, 7, 1024}, ge::FORMAT_NDHWC));
  op.SetAttr("strides", {1});
  op.SetAttr("ksize", {1});
  op.SetAttr("pads", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DDProtoTest, InfershapeAvgPool3DD_006) {
ge::op::AvgPool3DD op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 4, 7, 7, 1024}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 4, 7, 7, 1024}, ge::FORMAT_NDHWC));
  op.SetAttr("strides", {1});
  op.SetAttr("ksize", {1});
  op.SetAttr("pads",  {0, 0, 0});

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DDProtoTest, InfershapeAvgPool3DD_007) {
ge::op::AvgPool3DD op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 4, 7, 7, 1024}, ge::DT_FLOAT16, ge::FORMAT_NCDHW, {1, 4, 7, 7, 1024}, ge::FORMAT_NCDHW));
  op.SetAttr("strides", {1});
  op.SetAttr("ksize", {1});
  op.SetAttr("padding", "SAME");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_y_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {1, 4, 7, 7, 1024};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT16);
}

TEST_F(AvgPool3DDProtoTest, InfershapeAvgPool3DD_008) {
ge::op::AvgPool3DD op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 4, 7, 7, 1024}, ge::DT_FLOAT16, ge::FORMAT_DHWCN, {1, 4, 7, 7, 1024}, ge::FORMAT_DHWCN));
  op.SetAttr("strides", {1});
  op.SetAttr("ksize", {1});
  op.SetAttr("padding", "SAME");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_y_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {1, 4, 7, 7, 1024};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT16);
}

TEST_F(AvgPool3DDProtoTest, InfershapeAvgPool3DD_009) {
ge::op::AvgPool3DD op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 4, 7, 7, 1024}, ge::DT_FLOAT16, ge::FORMAT_DHWCN, {1, 4, 7, 7, 1024}, ge::FORMAT_DHWCN));
  op.SetAttr("strides", {1});
  op.SetAttr("ksize", {1});
  op.SetAttr("padding", "SAME");
  op.SetAttr("ceil_mode", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_y_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {1, 4, 7, 7, 1024};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT16);
}