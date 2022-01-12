#include <gtest/gtest.h>

#include <iostream>

#include "nn_pooling_ops.h"
#include "op_proto_test_util.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

class avg_poolv2 : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "avg_poolv2 SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "avg_poolv2 TearDown" << std::endl; }
};

TEST_F(avg_poolv2, avg_poolv2_infershape_test_fp16_1) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({3, 16, 16, 64}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {3, 16, 16, 64}, ge::FORMAT_NHWC));

  std::string padding_mode = "VALID";
  std::string data_format = "NHWC";
  op.SetAttr("ksize", {1, 2, 2, 1});
  op.SetAttr("strides", {1, 1, 1, 1});
  op.SetAttr("padding_mode", padding_mode);
  op.SetAttr("data_format", data_format);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {3, 15, 15, 64};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(avg_poolv2, avg_poolv2_infershape_test_int8_1) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({3, 16, 16, 64}, ge::DT_INT8, ge::FORMAT_NHWC, {3, 16, 16, 64}, ge::FORMAT_NHWC));

  std::string padding_mode = "VALID";
  std::string data_format = "NHWC";
  op.SetAttr("ksize", {1, 2, 2, 1});
  op.SetAttr("strides", {1, 1, 1, 1});
  op.SetAttr("padding_mode", padding_mode);
  op.SetAttr("data_format", data_format);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {3, 15, 15, 64};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(avg_poolv2, avg_poolv2_infershape_test_fp16_3) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({3, 16, 16, 64}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {3, 16, 16, 64}, ge::FORMAT_NHWC));

  std::string padding_mode = "VALID";
  std::string data_format = "NHWC";
  op.SetAttr("ksize", {1, 2, 2, 1});
  op.SetAttr("strides", {1, 1, 1, 1});
  op.SetAttr("padding_mode", padding_mode);
  op.SetAttr("data_format", data_format);
  op.SetAttr("ceil_mode", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {3, 15, 15, 64};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(avg_poolv2, avg_poolv2_infershape_test_fp16_2) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 128, 32, 32}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 128, 32, 32}, ge::FORMAT_NCHW));

  std::string padding_mode = "SAME";
  std::string data_format = "NCHW";
  op.SetAttr("ksize", {1, 1, 2, 2});
  op.SetAttr("strides", {1, 1, 1, 1});
  op.SetAttr("padding_mode", padding_mode);
  op.SetAttr("data_format", data_format);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1, 128, 32, 32};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

// dynamic hw SAME NHWC
TEST_F(avg_poolv2, avg_poolv2_infershape_test_fp16_dynamic_1) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({1, -1, -1, 128}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, -1, -1, 128},
                                                  ge::FORMAT_NHWC, {{1, 1}, {20, 100}, {20, 100}, {128, 128}}));
  std::string padding_mode = "SAME";
  std::string data_format = "NHWC";
  op.SetAttr("ksize", {1, 2, 2, 1});
  op.SetAttr("strides", {1, 1, 1, 1});
  op.SetAttr("padding_mode", padding_mode);
  op.SetAttr("data_format", data_format);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1, -1, -1, 128};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

// dynamic hw SAME NHWC ceil_mode true
TEST_F(avg_poolv2, avg_poolv2_infershape_test_fp16_dynamic_1_with_ceil_mode) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({1, -1, -1, 128}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, -1, -1, 128},
                                                  ge::FORMAT_NHWC, {{1, 1}, {20, 100}, {20, 100}, {128, 128}}));
  std::string padding_mode = "SAME";
  std::string data_format = "NHWC";
  op.SetAttr("ksize", {1, 2, 2, 1});
  op.SetAttr("strides", {1, 1, 1, 1});
  op.SetAttr("padding_mode", padding_mode);
  op.SetAttr("data_format", data_format);
  op.SetAttr("ceil_mode", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1, -1, -1, 128};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

// dynamic nhw VALID NCHW range -1
TEST_F(avg_poolv2, avg_poolv2_infershape_test_fp16_dynamic_2) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({-1, 128, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {-1, 128, -1, -1},
                                                  ge::FORMAT_NCHW, {{1, -1}, {128, 128}, {20, -1}, {20, 100}}));
  std::string padding_mode = "VALID";
  std::string data_format = "NCHW";
  op.SetAttr("ksize", {1, 1, 2, 2});
  op.SetAttr("strides", {1, 1, 1, 1});
  op.SetAttr("padding_mode", padding_mode);
  op.SetAttr("data_format", data_format);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, 128, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

// dynamic nhw VALID NCHW range -1 ceil_mode=True
TEST_F(avg_poolv2, avg_poolv2_infershape_test_fp16_dynamic_2_with_ceil_mode) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({-1, 128, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {-1, 128, -1, -1},
                                                  ge::FORMAT_NCHW, {{1, -1}, {128, 128}, {20, -1}, {20, 100}}));
  std::string padding_mode = "VALID";
  std::string data_format = "NCHW";
  op.SetAttr("ksize", {1, 1, 2, 2});
  op.SetAttr("strides", {1, 1, 1, 1});
  op.SetAttr("padding_mode", padding_mode);
  op.SetAttr("data_format", data_format);
  op.SetAttr("ceil_mode", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, 128, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

// dynamic -2
TEST_F(avg_poolv2, avg_poolv2_infershape_test_fp16_dynamic_3) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc_with_ori({-2}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {-2}, ge::FORMAT_NCHW));
  std::string padding_mode = "VALID";
  std::string data_format = "NCHW";
  op.SetAttr("ksize", {1, 1, 2, 2});
  op.SetAttr("strides", {1, 1, 1, 1});
  op.SetAttr("padding_mode", padding_mode);
  op.SetAttr("data_format", data_format);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dynamic c SAME NHWC
TEST_F(avg_poolv2, avg_poolv2_infershape_test_fp16_dynamic_4) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({1, 24, 24, -1}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 24, 24, -1},
                                                  ge::FORMAT_NHWC, {{1, 1}, {20, 100}, {20, 100}, {128, 128}}));
  std::string padding_mode = "SAME";
  std::string data_format = "NHWC";
  op.SetAttr("ksize", {1, 2, 2, 1});
  op.SetAttr("strides", {1, 1, 1, 1});
  op.SetAttr("padding_mode", padding_mode);
  op.SetAttr("data_format", data_format);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dynamic c SAME NHWC and ceil_mode = true
TEST_F(avg_poolv2, avg_poolv2_infershape_test_fp16_dynamic_4_with_ceil_mode) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({1, 24, 24, -1}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 24, 24, -1},
                                                  ge::FORMAT_NHWC, {{1, 1}, {20, 100}, {20, 100}, {128, 128}}));
  std::string padding_mode = "SAME";
  std::string data_format = "NHWC";
  op.SetAttr("ksize", {1, 2, 2, 1});
  op.SetAttr("strides", {1, 1, 1, 1});
  op.SetAttr("padding_mode", padding_mode);
  op.SetAttr("data_format", data_format);
  op.SetAttr("ceil_mode", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dynamic h VALID NCHW range -1 ceil_mode=false
TEST_F(avg_poolv2, avg_poolv2_infershape_test_fp16_dynamic_h_with_not_ceil_mode) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({1, 128, -1, 20}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 128, -1, 20},
                                                  ge::FORMAT_NCHW, {{1, 1}, {128, 128}, {20, 100}, {20, 20}}));
  std::string padding_mode = "CALCULATED";
  std::string data_format = "NCHW";
  op.SetAttr("ksize", {1, 1, 2, 2});
  op.SetAttr("strides", {1, 1, 1, 1});
  op.SetAttr("padding_mode", padding_mode);
  op.SetAttr("data_format", data_format);
  op.SetAttr("ceil_mode", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1, 128, -1, 19};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

// dynamic w VALID NCHW range -1 ceil_mode=true
TEST_F(avg_poolv2, avg_poolv2_infershape_test_fp16_dynamic_w_with_ceil_mode) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc_shape_range({1, 128, 20, -1}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 128, 20, -1},
                                                  ge::FORMAT_NCHW, {{1, 1}, {128, 128}, {20, 20}, {20, 100}}));
  std::string padding_mode = "CALCULATED";
  std::string data_format = "NCHW";
  op.SetAttr("ksize", {1, 1, 2, 2});
  op.SetAttr("strides", {1, 1, 1, 1});
  op.SetAttr("padding_mode", padding_mode);
  op.SetAttr("data_format", data_format);
  op.SetAttr("ceil_mode", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1, 128, 19, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(avg_poolv2, avg_poolv2_verify_test_dataformat) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 128, 32, 32}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 128, 32, 32}, ge::FORMAT_NCHW));

  std::string padding_mode = "SAME";
  std::string data_format = "HWCN";
  op.SetAttr("ksize", {1, 1, 2, 2});
  op.SetAttr("strides", {1, 1, 1, 1});
  op.SetAttr("padding_mode", padding_mode);
  op.SetAttr("data_format", data_format);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, avg_poolv2_verify_test_strides) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 128, 32, 32}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 128, 32, 32}, ge::FORMAT_NCHW));

  std::string padding_mode = "SAME";
  std::string data_format = "NCHW";
  op.SetAttr("ksize", {1, 1, 2, 2});
  op.SetAttr("strides", {1, 1, 1, 1, 1});
  op.SetAttr("padding_mode", padding_mode);
  op.SetAttr("data_format", data_format);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, avg_poolv2_verify_test_ksize1) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 128, 32, 32}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 128, 32, 32}, ge::FORMAT_NCHW));

  std::string padding_mode = "SAME";
  std::string data_format = "NCHW";
  op.SetAttr("ksize", {-1, -1, 2, 2});
  op.SetAttr("strides", {1, 1, 1, 1});
  op.SetAttr("padding_mode", padding_mode);
  op.SetAttr("data_format", data_format);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, avg_poolv2_verify_test_ksize2) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({3, 16, 16, 64}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {3, 16, 16, 64}, ge::FORMAT_NHWC));

  std::string padding_mode = "VALID";
  std::string data_format = "NHWC";
  op.SetAttr("ksize", {-1, 2, 2, 1});
  op.SetAttr("strides", {1, 1, 1, 1});
  op.SetAttr("padding_mode", padding_mode);
  op.SetAttr("data_format", data_format);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, avg_poolv2_verify_test_strides1) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 128, 32, 32}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 128, 32, 32}, ge::FORMAT_NCHW));

  std::string padding_mode = "SAME";
  std::string data_format = "NCHW";
  op.SetAttr("ksize", {1, 1, 2, 2});
  op.SetAttr("strides", {-1, -1, 1, 1});
  op.SetAttr("padding_mode", padding_mode);
  op.SetAttr("data_format", data_format);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, avg_poolv2_verify_test_strides2) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({3, 16, 16, 64}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {3, 16, 16, 64}, ge::FORMAT_NHWC));

  std::string padding_mode = "VALID";
  std::string data_format = "NHWC";
  op.SetAttr("ksize", {1, 2, 2, 1});
  op.SetAttr("strides", {1, 1, 1, -1});
  op.SetAttr("padding_mode", padding_mode);
  op.SetAttr("data_format", data_format);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, VerifyAvgPoolV2_001) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc({3, 16, 16, 64}, ge::DT_FLOAT16));
  op.SetAttr("ksize", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, VerifyAvgPoolV2_002) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc({3, 16, 16, 64}, ge::DT_FLOAT16));
  std::vector<int32_t> ksizeList = {1, 2};
  op.SetAttr("ksize", ksizeList);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, VerifyAvgPoolV2_003) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc({3, 16, 16, 64}, ge::DT_FLOAT16));
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, VerifyAvgPoolV2_004) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc({3, 16, 16, 64}, ge::DT_FLOAT16));
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, VerifyAvgPoolV2_005) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc({3, 16, 16, 64}, ge::DT_FLOAT16));
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, 1, 1, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", stridesList);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, VerifyAvgPoolV2_006) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc({3, 16, 16, 64}, ge::DT_FLOAT16));
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, 1, 1, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "error");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, VerifyAvgPoolV2_007) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc({3, 16, 16, 64}, ge::DT_FLOAT16));
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, 1, 1, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "SAME");
  op.SetAttr("pads", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, VerifyAvgPoolV2_008) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc({3, 16, 16, 64}, ge::DT_FLOAT16));
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, 1, 1, 1};
  std::vector<int32_t> padVec = {4, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "SAME");
  op.SetAttr("pads", padVec);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, VerifyAvgPoolV2_009) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc({3, 16, 16, 64}, ge::DT_FLOAT16));
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, 1, 1, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "SAME");
  op.SetAttr("data_format", stridesList);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, VerifyAvgPoolV2_010) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc({3, 16, 16, 64}, ge::DT_FLOAT16));
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, -1, -1, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "SAME");
  op.SetAttr("data_format", "NHWC");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, VerifyAvgPoolV2_011) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc({3, 16, 16, 64}, ge::DT_FLOAT16));
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, 1, -1, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "SAME");
  op.SetAttr("data_format", "NCHW");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, InfershapeAvgPoolV2_001) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x1", create_desc({3, 16, 16, 64}, ge::DT_FLOAT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, InfershapeAvgPoolV2_002) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc({3, 16, 16, 64}, ge::DT_FLOAT16));
  op.SetAttr("ksize", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, InfershapeAvgPoolV2_003) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc({3, 16, 16, 64}, ge::DT_FLOAT16));
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, InfershapeAvgPoolV2_004) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc({3, 16, 16, 64}, ge::DT_FLOAT16));
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, 1, 1, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", stridesList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, InfershapeAvgPoolV2_005) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc({3, 16, 16, 64}, ge::DT_FLOAT16));
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, 1, 1, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "SAME");
  op.SetAttr("pads", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, InfershapeAvgPoolV2_006) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc({3, 16, 16, 64}, ge::DT_FLOAT16));
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, 1, 1, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "SAME");
  op.SetAttr("data_format", stridesList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, InfershapeAvgPoolV2_007) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc({3, 16, 16, 64}, ge::DT_FLOAT16));
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, 1, 1, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "SAME");
  op.SetAttr("data_format", "NHWC");
  op.SetAttr("global_pooling", stridesList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, InfershapeAvgPoolV2_008) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc({3, 16, 16, 64}, ge::DT_FLOAT16));
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, 1, 1, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "SAME");
  op.SetAttr("data_format", "NHWC");
  op.SetAttr("ceil_mode", stridesList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, InfershapeAvgPoolV2_009) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc("x", create_desc({3, 16, 16, 64}, ge::DT_FLOAT16));
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, 1, 1, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "SAME");
  op.SetAttr("data_format", "NHWC");
  op.SetAttr("ceil_mode", stridesList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, InfershapeAvgPoolV2_010) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 128, 32, 32}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 128, 32, 32}, ge::FORMAT_NCHW));
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, 1, 1, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "SAME");
  op.SetAttr("data_format", "NHWC");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, InfershapeAvgPoolV2_011) {
  ge::op::AvgPoolV2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 128, 32}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 128, 32}, ge::FORMAT_NCHW));
  std::vector<int32_t> ksizeList = {1, 2, 2, 1};
  std::vector<int32_t> stridesList = {1, 1, 1, 1};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding_mode", "SAME");
  op.SetAttr("data_format", "NCHW");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, avg_poolv2SplicDataTest) {
    ge::op::AvgPoolV2 op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT16,
                       ge::FORMAT_NC1HWC0, {4, 3, 64, 64}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT16, 
                        ge::FORMAT_NC1HWC0, {4, 3, 64, 64}, ge::FORMAT_NCHW));
    op.SetAttr("ksize", {1, 1, 3, 3});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {1, 1, 1, 1});
    op.SetAttr("padding_mode", "SAME");
    op.SetAttr("data_format", "NCHW");
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {1, 62}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    std::vector<std::vector<int64_t>> tt;
    ge::AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, tt);

    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);

    std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {}, {}, {0, 63}, {}};
    EXPECT_EQ(expect_x_data_slice, x_data_slice);

    std::vector<int> pads;
    op.GetAttr("pads", pads);
    std::vector<int> expect_pads = {1, 1, 0, 0};
    EXPECT_EQ(expect_pads, pads);
}

TEST_F(avg_poolv2, avg_poolv2SplicDataTest02) {
    ge::op::AvgPoolV2 op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT16,
                       ge::FORMAT_NC1HWC0, {4, 3, 64, 64}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT16, 
                        ge::FORMAT_NC1HWC0, {4, 3, 64, 64}, ge::FORMAT_NCHW));
    op.SetAttr("ksize", {1, 1, 3, 3});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {1, 1, 1, 1});
    op.SetAttr("padding_mode", "CALCULATED");
    op.SetAttr("ceil_mode", true);
    op.SetAttr("data_format", "NCHW");
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {0, 63}, {1, 62}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    std::vector<std::vector<int64_t>> tt;
    ge::AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, tt);

    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);

    std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {}, {0, 63}, {0, 63}, {}};
    EXPECT_EQ(expect_x_data_slice, x_data_slice);

    std::vector<int> pads;
    op.GetAttr("pads", pads);
    std::vector<int> expect_pads = {1, 1, 0, 0};
    EXPECT_EQ(expect_pads, pads);
}

TEST_F(avg_poolv2, avg_poolv2SplicDataTest03) {
    ge::op::AvgPoolV2 op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT16,
                       ge::FORMAT_NC1HWC0, {4, 3, 64, 64}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT16, 
                        ge::FORMAT_NC1HWC0, {4, 3, 64, 64}, ge::FORMAT_NCHW));
    op.SetAttr("ksize", {1, 1, 3, 3});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {1, 1, 1, 1});
    op.SetAttr("padding_mode", "SAME");
    op.SetAttr("data_format", "NCHW");
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {0, 31}, {1, 30}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    std::vector<std::vector<int64_t>> tt;
    ge::AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, tt);

    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);

    std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {}, {0, 63}, {2, 62}, {}};
    EXPECT_EQ(expect_x_data_slice, x_data_slice);

    std::vector<int> pads;
    op.GetAttr("pads", pads);
    std::vector<int> expect_pads = {0, 1, 0, 0};
    EXPECT_EQ(expect_pads, pads);
}

TEST_F(avg_poolv2, avg_poolv2SplicDataTest04) {
    ge::op::AvgPoolV2 op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT16,
                       ge::FORMAT_NC1HWC0, {4, 3, 64, 64}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT16, 
                        ge::FORMAT_NC1HWC0, {4, 3, 64, 64}, ge::FORMAT_NCHW));
    op.SetAttr("ksize", {1, 3, 3, 1});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {1, 1, 1, 1});
    op.SetAttr("padding_mode", "CALCULATED");
    op.SetAttr("ceil_mode", true);
    op.SetAttr("data_format", "NHWC");
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {0, 63}, {1, 62}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    std::vector<std::vector<int64_t>> tt;
    ge::AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, tt);

    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);

    std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {}, {0, 63}, {0, 63}, {}};
    EXPECT_EQ(expect_x_data_slice, x_data_slice);

    std::vector<int> pads;
    op.GetAttr("pads", pads);
    std::vector<int> expect_pads = {1, 1, 0, 0};
    EXPECT_EQ(expect_pads, pads);
}

TEST_F(avg_poolv2, avg_poolv2SplicDataTest05) {
    ge::op::AvgPoolV2 op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT16,
                       ge::FORMAT_NC1HWC0, {4, 3, 64, 64}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT16, 
                        ge::FORMAT_NC1HWC0, {4, 3, 64, 64}, ge::FORMAT_NCHW));
    op.SetAttr("ksize", {1, 3, 3});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {1, 1, 1, 1});
    op.SetAttr("ceil_mode", true);
    op.SetAttr("data_format", "NHWC");
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {0, 63}, {1, 62}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    std::vector<std::vector<int64_t>> tt;
    ge::AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, tt);

    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);

    std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {}, {0, 63}, {0, 63}, {}};
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, avg_poolv2SplicDataTest06) {
    ge::op::AvgPoolV2 op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT16,
                       ge::FORMAT_NC1HWC0, {4, 3, 64, 64}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT16, 
                        ge::FORMAT_NC1HWC0, {4, 3, 64, 64}, ge::FORMAT_NCHW));
    op.SetAttr("ksize", {1, 3, 3, 1});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("padding_mode", "CALCULATED");
    op.SetAttr("ceil_mode", true);
    op.SetAttr("data_format", "NHWC");

    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, avg_poolv2SplicDataTest07) {
    ge::op::AvgPoolV2 op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT16,
                       ge::FORMAT_NC1HWC0, {4, 3, 64, 64}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT16, 
                        ge::FORMAT_NC1HWC0, {4, 3, 64, 64}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {1, 1, 1, 1});
    op.SetAttr("padding_mode", "CALCULATED");
    op.SetAttr("ceil_mode", true);
    op.SetAttr("data_format", "NHWC");

    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, avg_poolv2SplicDataTest08) {
    ge::op::AvgPoolV2 op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT16,
                       ge::FORMAT_NC1HWC0, {4, 3, 64, 64}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT16, 
                        ge::FORMAT_NC1HWC0, {4, 3, 64, 64}, ge::FORMAT_NCHW));
    op.SetAttr("ksize", {1, 3, 3, 1});
    op.SetAttr("pads", {1, 1, 1, 1});
    op.SetAttr("padding_mode", "CALCULATED");
    op.SetAttr("ceil_mode", true);
    op.SetAttr("data_format", "NHWC");

    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}
TEST_F(avg_poolv2, avg_poolv2SplicDataTest09) {
    ge::op::AvgPoolV2 op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT16,
                       ge::FORMAT_NC1HWC0, {4, 3, 64, 64}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT16, 
                        ge::FORMAT_NC1HWC0, {4, 3, 64, 64}, ge::FORMAT_NCHW));
    op.SetAttr("ksize", {1, 1, 3, 3});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {1, 1, 1, 1});
    op.SetAttr("padding_mode", "SAME");
    op.SetAttr("data_format", "NCHW");
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {1, 62}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    std::vector<std::vector<int64_t>> tt;
    ge::AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, tt);

    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);

    std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {}, {0, 63}, {}, {}};
    EXPECT_EQ(expect_x_data_slice, x_data_slice);

    std::vector<int> pads;
    op.GetAttr("pads", pads);
    std::vector<int> expect_pads = {0, 0, 1, 1};
    EXPECT_EQ(expect_pads, pads);
}

TEST_F(avg_poolv2, avg_poolv2SplicDataTest10) {
    ge::op::AvgPoolV2 op;
    op.UpdateInputDesc("x", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT16,
                       ge::FORMAT_NC1HWC0, {4, 3, 64, 64}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT16, 
                        ge::FORMAT_NC1HWC0, {4, 3, 64, 64}, ge::FORMAT_NCHW));
    op.SetAttr("ksize", {1, 3, 3, 1});
    op.SetAttr("strides", {1, 0, 0, 1});
    op.SetAttr("pads", {1, 1, 1, 1});
    op.SetAttr("padding_mode", "CALCULATED");
    op.SetAttr("ceil_mode", true);
    op.SetAttr("data_format", "NHWC");

    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_poolv2, avg_poolv2SplicDataTest11) {
    ge::op::AvgPoolV2 op;
    op.UpdateInputDesc("x", create_desc_with_ori({8, 12, 35, 35, 16}, ge::DT_FLOAT16,
                       ge::FORMAT_NC1HWC0, {8, 192, 35, 35}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({8, 12, 35, 35, 16}, ge::DT_FLOAT16, 
                        ge::FORMAT_NC1HWC0, {8, 192, 35, 35}, ge::FORMAT_NCHW));
    op.SetAttr("ksize", {1, 1, 3, 3});
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("pads", {1, 1, 1, 1});
    op.SetAttr("padding_mode", "CALCULATED");
    op.SetAttr("ceil_mode", false);
    op.SetAttr("data_format", "NCHW");
    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {0, 34}, {0, 33}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    std::vector<std::vector<int64_t>> tt;
    ge::AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, tt);

    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);

    std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {}, {0, 34}, {0, 34}, {}};
    EXPECT_EQ(expect_x_data_slice, x_data_slice);

    std::vector<int> pads;
    op.GetAttr("pads", pads);
    std::vector<int> expect_pads = {1, 1, 1, 0};
    EXPECT_EQ(expect_pads, pads);
}