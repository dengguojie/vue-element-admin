/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/op_desc_utils.h"
#include "graph/common_error_codes.h"

class MaxPoolExt2_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MaxPoolExt2_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MaxPoolExt2_UT TearDown" << std::endl;
  }
};

TEST_F(MaxPoolExt2_UT, InfershapeMaxPoolExt2_001) {
  ge::op::MaxPoolExt2 op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 6, 6, 6, 16},
                                               ge::FORMAT_NDHWC));
  op.SetAttr("ksize", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MaxPoolExt2_UT, InfershapeMaxPoolExt2_002) {
  ge::op::MaxPoolExt2 op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 6, 6, 6, 16},
                                               ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5};
  op.SetAttr("ksize", ksizeList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolExt2_UT, InfershapeMaxPoolExt2_003) {
  ge::op::MaxPoolExt2 op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 6, 6, 6, 16},
                                               ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolExt2_UT, InfershapeMaxPoolExt2_004) {
  ge::op::MaxPoolExt2 op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 6, 6, 6, 16},
                                               ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolExt2_UT, InfershapeMaxPoolExt2_005) {
  ge::op::MaxPoolExt2 op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 6, 6, 6, 16},
                                               ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("data_format", stridesList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolExt2_UT, InfershapeMaxPoolExt2_006) {
  ge::op::MaxPoolExt2 op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 6, 6, 6, 16},
                                               ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", ksizeList);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolExt2_UT, InfershapeMaxPoolExt2_007) {
  ge::op::MaxPoolExt2 op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NDHWC, {1, 6, 6, 6, 16},
                                               ge::FORMAT_NDHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "error");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolExt2_UT, InfershapeMaxPoolExt2_008) {
  ge::op::MaxPoolExt2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {6, 6, 6, 16}, ge::FORMAT_NHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "SAME");
  op.SetAttr("data_format", "NHWC");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {6, 2, 2, 16};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MaxPoolExt2_UT, InfershapeMaxPoolExt2_009) {
  ge::op::MaxPoolExt2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {6, 6, 6, 16}, ge::FORMAT_NHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "SAME");
  op.SetAttr("data_format", "NCHW");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {6, 2, 2, 16};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MaxPoolExt2_UT, InfershapeMaxPoolExt2_010) {
  ge::op::MaxPoolExt2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {6, 6, 6, 16}, ge::FORMAT_NHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "VALID");
  op.SetAttr("data_format", "NHWC");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {6, 1, 1, 16};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MaxPoolExt2_UT, InfershapeMaxPoolExt2_011) {
  ge::op::MaxPoolExt2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {6, 6, 6, 16}, ge::FORMAT_NHWC));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "VALID");
  op.SetAttr("data_format", "NCHW");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {6, 1, 1, 16};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MaxPoolExt2_UT, InfershapeMaxPoolExt2_012) {
  ge::op::MaxPoolExt2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {6, 6, 6, 16}, ge::FORMAT_NCHW));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "SAME");
  op.SetAttr("data_format", "NHWC");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {6, 6, 2, 6};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MaxPoolExt2_UT, InfershapeMaxPoolExt2_013) {
  ge::op::MaxPoolExt2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {6, 6, 6, 16}, ge::FORMAT_NCHW));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "SAME");
  op.SetAttr("data_format", "NCHW");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {6, 6, 2, 6};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MaxPoolExt2_UT, InfershapeMaxPoolExt2_014) {
  ge::op::MaxPoolExt2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {6, 6, 6, 16}, ge::FORMAT_NCHW));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "VALID");
  op.SetAttr("data_format", "NHWC");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {6, 6, 1, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MaxPoolExt2_UT, InfershapeMaxPoolExt2_015) {
  ge::op::MaxPoolExt2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {6, 6, 6, 16}, ge::FORMAT_NCHW));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "VALID");
  op.SetAttr("data_format", "NCHW");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {6, 6, 1, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MaxPoolExt2_UT, DataSliceMaxPoolExt2_001) {
  ge::op::MaxPoolExt2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {6, 6, 6, 16}, ge::FORMAT_NCHW));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "SAME");
  op.SetAttr("data_format", "NHWC");

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::NO_OVERLAP_DIM);
}

TEST_F(MaxPoolExt2_UT, DataSliceMaxPoolExt2_002) {
  ge::op::MaxPoolExt2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {6, 6, 6, 16}, ge::FORMAT_NCHW));
  std::vector<int32_t> ksizeList = {6, 6, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "SAME");
  op.SetAttr("data_format", "NHWC");

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::NO_OVERLAP_DIM);
}

TEST_F(MaxPoolExt2_UT, DataSliceMaxPoolExt2_003) {
  ge::op::MaxPoolExt2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {6, 6, 6, 16}, ge::FORMAT_NCHW));
  std::vector<int32_t> ksizeList = {6, 16, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "SAME");
  op.SetAttr("data_format", "NCHW");

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::NO_OVERLAP_DIM);
}

TEST_F(MaxPoolExt2_UT, DataSliceMaxPoolExt2_004) {
  ge::op::MaxPoolExt2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {6, 6, 6, 16}, ge::FORMAT_NCHW));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "VALID");
  op.SetAttr("data_format", "NCHW");

  std::vector<std::vector<int64_t>> output_data_slice = {{10, 20}, {}, {}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(output_tensor_desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::NO_OVERLAP_DIM);
}

TEST_F(MaxPoolExt2_UT, DataSliceMaxPoolExt2_005) {
  ge::op::MaxPoolExt2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {6, 6, 6, 16}, ge::FORMAT_NCHW));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "VALID");
  op.SetAttr("data_format", "NCHW");

  std::vector<std::vector<int64_t>> output_data_slice = {{}, {10, 20}, {}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(output_tensor_desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::NOT_SUPPORT_SLICE);
}

TEST_F(MaxPoolExt2_UT, DataSliceMaxPoolExt2_006) {
  ge::op::MaxPoolExt2 op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({6, 6, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {6, 6, 6, 16}, ge::FORMAT_NCHW));
  std::vector<int32_t> ksizeList = {5, 5, 5, 5};
  std::vector<int32_t> stridesList = {3, 3, 3, 3};
  op.SetAttr("ksize", ksizeList);
  op.SetAttr("strides", stridesList);
  op.SetAttr("padding", "VALID");
  op.SetAttr("data_format", "NCHW");

  std::vector<std::vector<int64_t>> output_data_slice = {{}, {}, {10, 20}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(output_tensor_desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::NO_OVERLAP_DIM);
}