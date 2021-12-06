#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"
#include "utils/op_desc_utils.h"

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