/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_pooling_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class dilation2d_filter_input_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "dilation2d_filter_input_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "dilation2d_filter_input_test TearDown" << std::endl;
  }
};

TEST_F(dilation2d_filter_input_test, dilation2d_filter_input_test_1) {
  std::vector<int64_t> stride_list = vector<int64_t>({1, 2, 3, 1});
  std::vector<int64_t> rate_list = vector<int64_t>({1, 3, 2, 1});
  std::vector<int64_t> pads = vector<int64_t>({0, 0, 0, 0});
  std::string padding_mode = "SAME";
  bool ceil_mode = false;
  std::string data_format = "NHWC";

  // expect result
  std::vector<int64_t> expected_shape = {3, 16, 16, 64};

  // new op and do infershape
  ge::op::Dilation2DBackpropInput op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({3, 16, 16, 64}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {3, 16, 16, 64}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("filter",
                     create_desc_with_ori({3, 3, 64}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {3, 3, 64}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("out_backprop", create_desc_with_ori({3, 8, 6, 64}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {3, 8, 6, 64},
                                                          ge::FORMAT_NHWC));
  op.set_attr_strides(stride_list);
  op.set_attr_rates(rate_list);
  op.set_attr_padding_mode(padding_mode);
  op.set_attr_pads(pads);
  op.set_attr_ceil_mode(ceil_mode);
  op.set_attr_data_format(data_format);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}

TEST_F(dilation2d_filter_input_test, dilation2d_filter_input_test_2) {
  std::vector<int64_t> stride_list = vector<int64_t>({1, 1, 1, 1});
  std::vector<int64_t> rate_list = vector<int64_t>({1, 1, 1, 1});
  std::vector<int64_t> pads = vector<int64_t>({0, 0, 0, 0});
  std::string padding_mode = "VALID";
  bool ceil_mode = false;
  std::string data_format = "NHWC";

  // expect result
  std::vector<int64_t> expected_shape = {1, 3, 6, 16};

  // new op and do infershape
  ge::op::Dilation2DBackpropInput op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 3, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 3, 6, 16}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("filter",
                     create_desc_with_ori({3, 3, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {3, 3, 16}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 1, 4, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 1, 4, 16},
                                                          ge::FORMAT_NHWC));
  op.set_attr_strides(stride_list);
  op.set_attr_rates(rate_list);
  op.set_attr_padding_mode(padding_mode);
  op.set_attr_pads(pads);
  op.set_attr_ceil_mode(ceil_mode);
  op.set_attr_data_format(data_format);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}

TEST_F(dilation2d_filter_input_test, dilation2d_filter_input_test_3) {
  std::vector<int64_t> stride_list = vector<int64_t>({1, 1, 1, 1});
  std::vector<int64_t> rate_list = vector<int64_t>({1, 1, 1, 1});
  std::vector<int64_t> pads = vector<int64_t>({1, 1, 1, 1});
  std::string padding_mode = "CALCULATED";
  bool ceil_mode = false;
  std::string data_format = "NHWC";

  // expect result
  std::vector<int64_t> expected_shape = {1, 17, 17, 1024};

  // new op and do infershape
  ge::op::Dilation2DBackpropInput op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 17, 17, 1024}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 17, 17, 1024},
                                               ge::FORMAT_NHWC));
  op.UpdateInputDesc(
      "filter", create_desc_with_ori({3, 3, 1024}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {3, 3, 1024}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 17, 17, 1024}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
                                                          {1, 17, 17, 1024}, ge::FORMAT_NHWC));
  op.set_attr_strides(stride_list);
  op.set_attr_rates(rate_list);
  op.set_attr_padding_mode(padding_mode);
  op.set_attr_pads(pads);
  op.set_attr_ceil_mode(ceil_mode);
  op.set_attr_data_format(data_format);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}

TEST_F(dilation2d_filter_input_test, dilation2d_filter_input_test_4) {
  std::vector<int64_t> stride_list = vector<int64_t>({1, 1, 1, 1});
  std::vector<int64_t> rate_list = vector<int64_t>({1, 1, 1, 1});
  std::vector<int64_t> pads = vector<int64_t>({0, 0, 0, 0});
  std::string padding_mode = "VALID";
  bool ceil_mode = false;
  std::string data_format = "NCHW";

  // expect result
  std::vector<int64_t> expected_shape = {1, 1024, 17, 17};

  // new op and do infershape
  ge::op::Dilation2DBackpropInput op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 1024, 17, 17},
                                               ge::FORMAT_NCHW));
  op.UpdateInputDesc(
      "filter", create_desc_with_ori({1024, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1024, 3, 3}, ge::FORMAT_NCHW));
  op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 1024, 15, 15}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                          {1, 1024, 15, 15}, ge::FORMAT_NCHW));
  op.set_attr_strides(stride_list);
  op.set_attr_rates(rate_list);
  op.set_attr_padding_mode(padding_mode);
  op.set_attr_pads(pads);
  op.set_attr_ceil_mode(ceil_mode);
  op.set_attr_data_format(data_format);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}

TEST_F(dilation2d_filter_input_test, VerifierDilation2DBackpropInput_001) {
  ge::op::Dilation2DBackpropInput op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 1024, 17, 17},
                                               ge::FORMAT_NCHW));
  op.UpdateInputDesc("filter",
                     create_desc_with_ori({1024, 3, 3}, ge::DT_FLOAT, ge::FORMAT_NCHW, {1024, 3, 3}, ge::FORMAT_NCHW));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(dilation2d_filter_input_test, VerifierDilation2DBackpropInput_002) {
  ge::op::Dilation2DBackpropInput op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 1024, 17, 17},
                                               ge::FORMAT_NCHW));
  op.UpdateInputDesc("filter",
                     create_desc_with_ori({1024, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1024, 3}, ge::FORMAT_NCHW));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(dilation2d_filter_input_test, VerifierDilation2DBackpropInput_003) {
  ge::op::Dilation2DBackpropInput op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 1024, 17}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 1024, 17}, ge::FORMAT_NCHW));
  op.UpdateInputDesc(
      "filter", create_desc_with_ori({1024, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1024, 3, 3}, ge::FORMAT_NCHW));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(dilation2d_filter_input_test, VerifierDilation2DBackpropInput_004) {
  ge::op::Dilation2DBackpropInput op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 1024, 17}, ge::FORMAT_NCHW));
  op.UpdateInputDesc(
      "filter", create_desc_with_ori({1024, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1024, 3, 3}, ge::FORMAT_NCHW));
  op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 1024, 17}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 1024, 17},
                                                          ge::FORMAT_NCHW));
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(dilation2d_filter_input_test, VerifierDilation2DBackpropInput_005) {
  ge::op::Dilation2DBackpropInput op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 1024, 17, 17},
                                               ge::FORMAT_NCHW));
  op.UpdateInputDesc(
      "filter", create_desc_with_ori({1024, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1024, 3, 3}, ge::FORMAT_NCHW));
  op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                          {1, 1024, 17}, ge::FORMAT_NCHW));
  op.SetAttr("data_format", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(dilation2d_filter_input_test, VerifierDilation2DBackpropInput_006) {
  ge::op::Dilation2DBackpropInput op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 1024, 17, 17},
                                               ge::FORMAT_NCHW));
  op.UpdateInputDesc(
      "filter", create_desc_with_ori({1024, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1024, 3, 3}, ge::FORMAT_NCHW));
  op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                          {1, 1024, 17}, ge::FORMAT_NCHW));
  op.SetAttr("data_format", "ND");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(dilation2d_filter_input_test, VerifierDilation2DBackpropInput_007) {
  ge::op::Dilation2DBackpropInput op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 1024, 17, 17},
                                               ge::FORMAT_NHWC));
  op.UpdateInputDesc(
      "filter", create_desc_with_ori({1024, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1024, 3, 3}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
                                                          {1, 1024, 17}, ge::FORMAT_NHWC));
  op.SetAttr("data_format", "NCHW");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(dilation2d_filter_input_test, VerifierDilation2DBackpropInput_008) {
  ge::op::Dilation2DBackpropInput op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 1024, 17, 17},
                                               ge::FORMAT_NHWC));
  op.UpdateInputDesc(
      "filter", create_desc_with_ori({1024, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1024, 3, 3}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                          {1, 1024, 17}, ge::FORMAT_NCHW));
  op.SetAttr("data_format", "NCHW");

  std::vector<int64_t> stride_list = vector<int64_t>({1, 1, 1});
  op.SetAttr("strides", stride_list);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(dilation2d_filter_input_test, VerifierDilation2DBackpropInput_009) {
  ge::op::Dilation2DBackpropInput op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 1024, 17, 17},
                                               ge::FORMAT_NHWC));
  op.UpdateInputDesc(
      "filter", create_desc_with_ori({1024, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1024, 3, 3}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                          {1, 1024, 17}, ge::FORMAT_NCHW));
  op.SetAttr("data_format", "NCHW");
  std::vector<int64_t> stride_list = vector<int64_t>({1, 1, 1, 1});
  op.SetAttr("strides", stride_list);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(dilation2d_filter_input_test, VerifierDilation2DBackpropInput_010) {
  ge::op::Dilation2DBackpropInput op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 1024, 17, 17},
                                               ge::FORMAT_NHWC));
  op.UpdateInputDesc(
      "filter", create_desc_with_ori({1024, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1024, 3, 3}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                          {1, 1024, 17}, ge::FORMAT_NCHW));
  op.SetAttr("data_format", "NCHW");
  std::vector<int64_t> stride_list = vector<int64_t>({1, 1, 1, 1});
  op.SetAttr("strides", stride_list);
  std::vector<int64_t> rate_list = vector<int64_t>({1, 3, 2});
  op.SetAttr("rates", rate_list);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(dilation2d_filter_input_test, VerifierDilation2DBackpropInput_011) {
  ge::op::Dilation2DBackpropInput op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 1024, 17, 17},
                                               ge::FORMAT_NHWC));
  op.UpdateInputDesc(
      "filter", create_desc_with_ori({1024, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1024, 3, 3}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                          {1, 1024, 17}, ge::FORMAT_NCHW));
  op.SetAttr("data_format", "NCHW");
  std::vector<int64_t> stride_list = vector<int64_t>({1, 1, 1, 1});
  op.SetAttr("strides", stride_list);
  std::vector<int64_t> rate_list = vector<int64_t>({1, 3, 2, 1});
  op.SetAttr("rates", rate_list);
  op.SetAttr("padding_mode", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(dilation2d_filter_input_test, VerifierDilation2DBackpropInput_012) {
  ge::op::Dilation2DBackpropInput op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 1024, 17, 17},
                                               ge::FORMAT_NHWC));
  op.UpdateInputDesc(
      "filter", create_desc_with_ori({1024, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1024, 3, 3}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                          {1, 1024, 17}, ge::FORMAT_NCHW));
  op.SetAttr("data_format", "NCHW");
  std::vector<int64_t> stride_list = vector<int64_t>({1, 1, 1, 1});
  op.SetAttr("strides", stride_list);
  std::vector<int64_t> rate_list = vector<int64_t>({1, 3, 2, 1});
  op.SetAttr("rates", rate_list);
  op.SetAttr("padding_mode", "error");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(dilation2d_filter_input_test, VerifierDilation2DBackpropInput_013) {
  ge::op::Dilation2DBackpropInput op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 1024, 17, 17},
                                               ge::FORMAT_NHWC));
  op.UpdateInputDesc(
      "filter", create_desc_with_ori({1024, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1024, 3, 3}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                          {1, 1024, 17}, ge::FORMAT_NCHW));
  op.SetAttr("data_format", "NCHW");
  std::vector<int64_t> stride_list = vector<int64_t>({1, 1, 1, 1});
  op.SetAttr("strides", stride_list);
  std::vector<int64_t> rate_list = vector<int64_t>({1, 3, 2, 1});
  op.SetAttr("rates", rate_list);
  op.SetAttr("padding_mode", "SAME");
  op.SetAttr("pads", true);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(dilation2d_filter_input_test, VerifierDilation2DBackpropInput_014) {
  ge::op::Dilation2DBackpropInput op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 1024, 17, 17},
                                               ge::FORMAT_NHWC));
  op.UpdateInputDesc(
      "filter", create_desc_with_ori({1024, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1024, 3, 3}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                          {1, 1024, 17}, ge::FORMAT_NCHW));
  op.SetAttr("data_format", "NCHW");
  std::vector<int64_t> stride_list = vector<int64_t>({1, 1, 1, 1});
  op.SetAttr("strides", stride_list);
  std::vector<int64_t> rate_list = vector<int64_t>({1, 3, 2, 1});
  op.SetAttr("rates", rate_list);
  op.SetAttr("padding_mode", "SAME");
  std::vector<int64_t> pads = vector<int64_t>({0, 0, 0});
  op.SetAttr("pads", pads);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(dilation2d_filter_input_test, VerifierDilation2DBackpropInput_015) {
  ge::op::Dilation2DBackpropInput op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 1024, 17, 17},
                                               ge::FORMAT_NHWC));
  op.UpdateInputDesc(
      "filter", create_desc_with_ori({1024, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1024, 3, 3}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                          {1, 1024, 17}, ge::FORMAT_NCHW));
  op.SetAttr("data_format", "NCHW");
  std::vector<int64_t> stride_list = vector<int64_t>({1, 1, 1, 1});
  op.SetAttr("strides", stride_list);
  std::vector<int64_t> rate_list = vector<int64_t>({1, 3, 2, 1});
  op.SetAttr("rates", rate_list);
  op.SetAttr("padding_mode", "SAME");
  std::vector<int64_t> pads = vector<int64_t>({0, 0, 0, 0});
  op.SetAttr("pads", pads);
  op.SetAttr("ceil_mode", "ceil_mode");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(dilation2d_filter_input_test, VerifierDilation2DBackpropInput_016) {
  ge::op::Dilation2DBackpropInput op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 1024, 17, 17},
                                               ge::FORMAT_NHWC));
  op.UpdateInputDesc(
      "filter", create_desc_with_ori({1024, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1024, 3, 3}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                          {1, 1024, 17}, ge::FORMAT_NCHW));
  op.SetAttr("data_format", "NCHW");
  std::vector<int64_t> stride_list = vector<int64_t>({1, 1, 1, 1});
  op.SetAttr("strides", stride_list);
  std::vector<int64_t> rate_list = vector<int64_t>({2, 3, 2, 1});
  op.SetAttr("rates", rate_list);
  op.SetAttr("padding_mode", "SAME");
  std::vector<int64_t> pads = vector<int64_t>({0, 0, 0, 0});
  op.SetAttr("pads", pads);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(dilation2d_filter_input_test, VerifierDilation2DBackpropInput_017) {
  ge::op::Dilation2DBackpropInput op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 1024, 17, 17},
                                               ge::FORMAT_NHWC));
  op.UpdateInputDesc(
      "filter", create_desc_with_ori({1024, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1024, 3, 3}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                          {1, 1024, 17}, ge::FORMAT_NCHW));
  op.SetAttr("data_format", "NCHW");
  std::vector<int64_t> stride_list = vector<int64_t>({2, 1, 1, 1});
  op.SetAttr("strides", stride_list);
  std::vector<int64_t> rate_list = vector<int64_t>({1, 1, 1, 1});
  op.SetAttr("rates", rate_list);
  op.SetAttr("padding_mode", "SAME");
  std::vector<int64_t> pads = vector<int64_t>({0, 0, 0, 0});
  op.SetAttr("pads", pads);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(dilation2d_filter_input_test, VerifierDilation2DBackpropInput_018) {
  ge::op::Dilation2DBackpropInput op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 1024, 17, 17},
                                               ge::FORMAT_NHWC));
  op.UpdateInputDesc(
      "filter", create_desc_with_ori({1024, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1024, 3, 3}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                          {1, 1024, 17}, ge::FORMAT_NCHW));
  op.SetAttr("data_format", "NCHW");
  std::vector<int64_t> stride_list = vector<int64_t>({1, 1, 1, 1});
  op.SetAttr("strides", stride_list);
  std::vector<int64_t> rate_list = vector<int64_t>({1, 1, 1, 1});
  op.SetAttr("rates", rate_list);
  op.SetAttr("padding_mode", "CALCULATED");
  std::vector<int64_t> pads = vector<int64_t>({10000, 0, 0, 0});
  op.SetAttr("pads", pads);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(dilation2d_filter_input_test, InferShapeDilation2DBackpropInput_001) {
  std::vector<int64_t> stride_list = vector<int64_t>({1, 2, 3, 1});
  std::vector<int64_t> rate_list = vector<int64_t>({1, 3, 2, 1});
  std::vector<int64_t> pads = vector<int64_t>({0, 0, 0, 0});
  std::string padding_mode = "SAME";
  bool ceil_mode = false;
  std::string data_format = "NHWC";

  // expect result
  std::vector<int64_t> expected_shape = {3, -1, -1, 64};

  // new op and do infershape
  ge::op::Dilation2DBackpropInput op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({3, -1, 16, 64}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {3, 16, 16, 64}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("filter",
                     create_desc_with_ori({3, 3, 64}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {3, 3, 64}, ge::FORMAT_NHWC));
  op.set_attr_strides(stride_list);
  op.set_attr_rates(rate_list);
  op.set_attr_padding_mode(padding_mode);
  op.set_attr_pads(pads);
  op.set_attr_ceil_mode(ceil_mode);
  op.set_attr_data_format(data_format);

  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}