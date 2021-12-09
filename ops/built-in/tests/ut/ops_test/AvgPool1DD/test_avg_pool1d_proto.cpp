#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"

// ----------------AvgPool1DD-------------------
class AvgPool1DDProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AvgPool1DD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AvgPool1DD Proto Test TearDown" << std::endl;
  }
};

// TODO fix me run failed
TEST_F(AvgPool1DDProtoTest, apply_avg_pool1dd_infershape_verify_test) {
  ge::op::AvgPool1DD op;
  op.UpdateInputDesc("x", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT16));
  op.SetAttr("ksize", 4);
  op.SetAttr("strides", 2);
  op.SetAttr("pads", {0, 0});
  op.SetAttr("ceil_mode", false);
  op.SetAttr("count_include_pad", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_y_output_shape = {16, 1, 1, 7999, 16};
  EXPECT_EQ(y_desc.GetShape().GetDims(), expected_y_output_shape);
}

TEST_F(AvgPool1DDProtoTest, InfershapeAvgPool1DD_001) {
  ge::op::AvgPool1DD op;
  op.UpdateInputDesc("x", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT16));
  op.SetAttr("ksize", false);
  op.SetAttr("strides", false);
  op.SetAttr("pads", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1DDProtoTest, InfershapeAvgPool1DD_002) {
  ge::op::AvgPool1DD op;
  op.UpdateInputDesc("x", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT16));
  op.SetAttr("strides", 0);
  op.SetAttr("pads", {0});

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1DDProtoTest, InfershapeAvgPool1DD_003) {
  ge::op::AvgPool1DD op;
  op.UpdateInputDesc("x", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT16));
  op.SetAttr("strides", 1);
  op.SetAttr("pads", {0, 0});
  op.SetAttr("ceil_mode", {0});

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(AvgPool1DDProtoTest, InfershapeAvgPool1DD_004) {
  ge::op::AvgPool1DD op;
  op.UpdateInputDesc("x", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT16));
  op.SetAttr("strides", 1);
  op.SetAttr("pads", {0, 0});
  op.SetAttr("ceil_mode", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(AvgPool1DDProtoTest, VerifyAvgPool1DD_001) {
  ge::op::AvgPool1DD op;
  op.UpdateInputDesc("x", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT16));
  op.UpdateInputDesc("assist_matrix", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1DDProtoTest, VerifyAvgPool1DD_002) {
  ge::op::AvgPool1DD op;
  op.UpdateInputDesc("x", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT16));
  op.UpdateInputDesc("assist_matrix", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT16));
  op.SetAttr("pads", {});

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1DDProtoTest, VerifyAvgPool1DD_003) {
  ge::op::AvgPool1DD op;
  op.UpdateInputDesc("x", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT16));
  op.UpdateInputDesc("assist_matrix", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT16));
  op.SetAttr("pads", {1});

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1DDProtoTest, VerifyAvgPool1DD_004) {
  ge::op::AvgPool1DD op;
  op.UpdateInputDesc("x", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT16));
  op.UpdateInputDesc("assist_matrix", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT16));
  op.SetAttr("pads", {1, 1});

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1DDProtoTest, VerifyAvgPool1DD_005) {
  ge::op::AvgPool1DD op;
  op.UpdateInputDesc("x", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT16));
  op.UpdateInputDesc("assist_matrix", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT16));
  op.SetAttr("pads", {1, 1});
  op.SetAttr("ksize", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1DDProtoTest, VerifyAvgPool1DD_006) {
  ge::op::AvgPool1DD op;
  op.UpdateInputDesc("x", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT16));
  op.UpdateInputDesc("assist_matrix", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT16));
  op.SetAttr("pads", {1, 1});
  op.SetAttr("ksize", 1);
  op.SetAttr("strides", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1DDProtoTest, VerifyAvgPool1DD_007) {
  ge::op::AvgPool1DD op;
  op.UpdateInputDesc("x", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT16));
  op.UpdateInputDesc("assist_matrix", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT16));
  op.SetAttr("pads", {1, 1});
  op.SetAttr("ksize", 1);
  op.SetAttr("strides", 0);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1DDProtoTest, VerifyAvgPool1DD_008) {
  ge::op::AvgPool1DD op;
  op.UpdateInputDesc("x", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT16));
  op.UpdateInputDesc("assist_matrix", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT16));
  std::vector<int64_t> pads = {1, 2};
  op.SetAttr("pads", pads);
  op.SetAttr("ksize", 1);
  op.SetAttr("strides", 1);
  op.SetAttr("ceil_mode", pads);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1DDProtoTest, VerifyAvgPool1DD_009) {
  ge::op::AvgPool1DD op;
  op.UpdateInputDesc("x", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT16));
  op.UpdateInputDesc("assist_matrix", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT16));
  std::vector<int64_t> pads = {1, 2};
  op.SetAttr("pads", pads);
  op.SetAttr("ksize", 1);
  op.SetAttr("strides", 1);
  op.SetAttr("ceil_mode", true);
  op.SetAttr("count_include_pad", pads);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPool1DDProtoTest, VerifyAvgPool1DD_010) {
  ge::op::AvgPool1DD op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 2, 2, 2, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 2, 2, 2, 16}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("assist_matrix", create_desc({16, 1, 1, 16000, 16}, ge::DT_FLOAT16));
  std::vector<int64_t> pads = {1, 2};
  op.SetAttr("pads", pads);
  op.SetAttr("ksize", 1);
  op.SetAttr("strides", 1);
  op.SetAttr("ceil_mode", false);
  op.SetAttr("count_include_pad", true);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}