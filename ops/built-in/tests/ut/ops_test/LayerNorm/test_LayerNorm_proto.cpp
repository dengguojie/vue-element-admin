/**
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_LayerNorm_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"
#include "utils/attr_utils.h"
#include "utils/op_desc_utils.h"
#include "util/common_shape_fns.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/common_error_codes.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/node_utils.h"
#include "op_log.h"
#include "op_desc.h"
#include "all_ops.h"
#include "common/utils/ut_op_util.h"

class LayerNormTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "LayerNorm SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "LayerNorm TearDown" << std::endl;
  }
};

TEST_F(LayerNormTest, layer_norm_test_1) {
  ge::op::LayerNorm op;

  op.UpdateInputDesc("x", create_desc({30, 256, 512}, ge::DT_FLOAT));

  int begin_norm_axis = -4;
  op.SetAttr("begin_norm_axis", begin_norm_axis);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, 0);
}

TEST_F(LayerNormTest, layer_norm_test_2) {
  ge::op::LayerNorm op;

  op.UpdateInputDesc("x", create_desc({30, 256, 512}, ge::DT_FLOAT));
  op.UpdateInputDesc("mean", create_desc({512}, ge::DT_FLOAT));
  op.UpdateInputDesc("variance", create_desc({512}, ge::DT_FLOAT));

  int begin_norm_axis = -1;
  op.SetAttr("begin_norm_axis", begin_norm_axis);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_y_desc = op.GetOutputDesc("y");
  auto output_mean_desc = op.GetOutputDesc("mean");
  auto output_var_desc = op.GetOutputDesc("variance");

  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_mean_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_var_desc.GetDataType(), ge::DT_FLOAT);

  std::vector<int64_t> expected_y_shape = {30, 256, 512};
  std::vector<int64_t> expected_mv_shape = {30, 256, 1};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
  EXPECT_EQ(output_mean_desc.GetShape().GetDims(), expected_mv_shape);
  EXPECT_EQ(output_var_desc.GetShape().GetDims(), expected_mv_shape);
}

TEST_F(LayerNormTest, layer_norm_data_slice_test_001) {
  ge::op::LayerNorm op;

  op.UpdateInputDesc("x",
                     create_desc_with_ori({30, 256, 512}, ge::DT_FLOAT, ge::FORMAT_ND, {30, 256, 512}, ge::FORMAT_ND));
  op.UpdateInputDesc("gamma",
                     create_desc_with_ori({512}, ge::DT_FLOAT, ge::FORMAT_ND, {512}, ge::FORMAT_ND));
  op.UpdateInputDesc("beta",
                     create_desc_with_ori({512}, ge::DT_FLOAT, ge::FORMAT_ND, {512}, ge::FORMAT_ND));
  op.UpdateOutputDesc("y",
                     create_desc_with_ori({30, 256, 512}, ge::DT_FLOAT, ge::FORMAT_ND, {30, 256, 512}, ge::FORMAT_ND));

  int begin_params_axis = -1;
  op.SetAttr("begin_params_axis", begin_params_axis);

  std::vector<std::vector<int64_t>> y_data_slice = {{0, 15}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(output_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  ge::GeTensorDescPtr input_x = op_desc->MutableInputDesc("x");
  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(input_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
  ge::GeTensorDescPtr input_gamma = op_desc->MutableInputDesc("gamma");
  std::vector<std::vector<int64_t>> gamma_data_slice;
  ge::AttrUtils::GetListListInt(input_gamma, ge::ATTR_NAME_DATA_SLICE, gamma_data_slice);
  ge::GeTensorDescPtr input_beta = op_desc->MutableInputDesc("beta");
  std::vector<std::vector<int64_t>> beta_data_slice;
  ge::AttrUtils::GetListListInt(input_beta, ge::ATTR_NAME_DATA_SLICE, beta_data_slice);
  std::vector<std::vector<int64_t>> expect_x_data_slice = {{0, 15}, {}, {}};
  std::vector<std::vector<int64_t>> expect_gamma_data_slice = {{}};
  std::vector<std::vector<int64_t>> expect_beta_data_slice = {{}};
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  EXPECT_EQ(expect_x_data_slice, x_data_slice);
  EXPECT_EQ(expect_gamma_data_slice, gamma_data_slice);
  EXPECT_EQ(expect_beta_data_slice, beta_data_slice);
}

TEST_F(LayerNormTest, layer_norm_data_slice_test_002) {
  ge::op::LayerNorm op;

  op.UpdateInputDesc("x",
                     create_desc_with_ori({4, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ, {32, 62}, ge::FORMAT_ND));
  op.UpdateInputDesc("gamma",
                     create_desc_with_ori({62}, ge::DT_FLOAT, ge::FORMAT_ND, {62}, ge::FORMAT_ND));
  op.UpdateInputDesc("beta",
                     create_desc_with_ori({62}, ge::DT_FLOAT, ge::FORMAT_ND, {62}, ge::FORMAT_ND));
  op.UpdateOutputDesc("y",
                     create_desc_with_ori({4, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ, {32, 62}, ge::FORMAT_ND));

  int begin_params_axis = -1;
  op.SetAttr("begin_params_axis", begin_params_axis);

  std::vector<std::vector<int64_t>> y_data_slice = {{0, 2}, {}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_y = op_desc->MutableOutputDesc("y");
  vector<int64_t> shape_ori = {32, 62};
  output_y->SetOriginShape(ge::GeShape(shape_ori));
  output_y->SetOriginFormat(ge::FORMAT_ND);
  ge::GeTensorDescPtr input_x = op_desc->MutableInputDesc("x");
  input_x->SetOriginShape(ge::GeShape(shape_ori));
  input_x->SetOriginFormat(ge::FORMAT_ND);
  ge::AttrUtils::SetListListInt(output_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();

  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(input_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
  ge::GeTensorDescPtr input_gamma = op_desc->MutableInputDesc("gamma");
  std::vector<std::vector<int64_t>> gamma_data_slice;
  ge::AttrUtils::GetListListInt(input_gamma, ge::ATTR_NAME_DATA_SLICE, gamma_data_slice);
  ge::GeTensorDescPtr input_beta = op_desc->MutableInputDesc("beta");
  std::vector<std::vector<int64_t>> beta_data_slice;
  ge::AttrUtils::GetListListInt(input_beta, ge::ATTR_NAME_DATA_SLICE, beta_data_slice);
  std::vector<std::vector<int64_t>> expect_x_data_slice = {{0, 2}, {}, {}, {}};
  std::vector<std::vector<int64_t>> expect_gamma_data_slice = {{}};
  std::vector<std::vector<int64_t>> expect_beta_data_slice = {{}};
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  EXPECT_EQ(expect_x_data_slice, x_data_slice);
  EXPECT_EQ(expect_gamma_data_slice, gamma_data_slice);
  EXPECT_EQ(expect_beta_data_slice, beta_data_slice);
}

TEST_F(LayerNormTest, layer_norm_data_slice_test_003) {
  ge::op::LayerNorm op;
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  op.UpdateInputDesc("x",
                     create_desc_with_ori({4, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NCHW, {4, 2, 16, 16}, ge::FORMAT_NCHW));
  op.UpdateInputDesc("gamma",
                     create_desc_with_ori({16}, ge::DT_FLOAT, ge::FORMAT_ND, {16}, ge::FORMAT_ND));
  op.UpdateInputDesc("beta",
                     create_desc_with_ori({16}, ge::DT_FLOAT, ge::FORMAT_ND, {16}, ge::FORMAT_ND));
  op.UpdateOutputDesc("y",
                     create_desc_with_ori({4, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NCHW, {4, 2, 16, 16}, ge::FORMAT_NCHW));

  int begin_params_axis = -1;
  op.SetAttr("begin_params_axis", begin_params_axis);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(LayerNormTest, layer_norm_data_slice_test_004) {
  ge::op::LayerNorm op;
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  op.UpdateInputDesc("x",
                     create_desc_with_ori({4, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NHWC, {4, 2, 16, 16}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("gamma",
                     create_desc_with_ori({16}, ge::DT_FLOAT, ge::FORMAT_ND, {16}, ge::FORMAT_ND));
  op.UpdateInputDesc("beta",
                     create_desc_with_ori({16}, ge::DT_FLOAT, ge::FORMAT_ND, {16}, ge::FORMAT_ND));
  op.UpdateOutputDesc("y",
                     create_desc_with_ori({4, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NHWC, {4, 2, 16, 16}, ge::FORMAT_NHWC));

  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(LayerNormTest, layer_norm_data_slice_test_005) {
  ge::op::LayerNorm op;
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  op.UpdateInputDesc("x",
                     create_desc_with_ori({4, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, {32, 62}, ge::FORMAT_ND));
  op.UpdateInputDesc("gamma",
                     create_desc_with_ori({62}, ge::DT_FLOAT, ge::FORMAT_ND, {62}, ge::FORMAT_ND));
  op.UpdateInputDesc("beta",
                     create_desc_with_ori({62}, ge::DT_FLOAT, ge::FORMAT_ND, {62}, ge::FORMAT_ND));
  op.UpdateOutputDesc("y",
                     create_desc_with_ori({4, 2, 16, 16}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ, {32, 62}, ge::FORMAT_ND));

  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(LayerNormTest, layer_norm_data_slice_test_006) {
  ge::op::LayerNorm op;

  op.UpdateInputDesc("x",
                     create_desc_with_ori({30, 256, 512}, ge::DT_FLOAT, ge::FORMAT_ND, {30, 256, 512}, ge::FORMAT_ND));
  op.UpdateInputDesc("gamma",
                     create_desc_with_ori({30, 256, 512}, ge::DT_FLOAT, ge::FORMAT_ND, {30, 256, 512}, ge::FORMAT_ND));
  op.UpdateInputDesc("beta",
                     create_desc_with_ori({30, 256, 512}, ge::DT_FLOAT, ge::FORMAT_ND, {30, 256, 512}, ge::FORMAT_ND));
  op.UpdateOutputDesc("y",
                     create_desc_with_ori({30, 256, 512}, ge::DT_FLOAT, ge::FORMAT_ND, {30, 256, 512}, ge::FORMAT_ND));

  int begin_params_axis = 0;
  op.SetAttr("begin_params_axis", begin_params_axis);

  std::vector<std::vector<int64_t>> y_data_slice = {{0, 15}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(output_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  ge::GeTensorDescPtr input_x = op_desc->MutableInputDesc("x");
  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(input_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
  ge::GeTensorDescPtr input_gamma = op_desc->MutableInputDesc("gamma");
  std::vector<std::vector<int64_t>> gamma_data_slice;
  ge::AttrUtils::GetListListInt(input_gamma, ge::ATTR_NAME_DATA_SLICE, gamma_data_slice);
  ge::GeTensorDescPtr input_beta = op_desc->MutableInputDesc("beta");
  std::vector<std::vector<int64_t>> beta_data_slice;
  ge::AttrUtils::GetListListInt(input_beta, ge::ATTR_NAME_DATA_SLICE, beta_data_slice);
  std::vector<std::vector<int64_t>> expect_x_data_slice = {{0, 15}, {}, {}};
  std::vector<std::vector<int64_t>> expect_gamma_data_slice = {{0, 15}, {}, {}};
  std::vector<std::vector<int64_t>> expect_beta_data_slice = {{0, 15}, {}, {}};
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  EXPECT_EQ(expect_x_data_slice, x_data_slice);
  EXPECT_EQ(expect_gamma_data_slice, gamma_data_slice);
  EXPECT_EQ(expect_beta_data_slice, beta_data_slice);
}

TEST_F(LayerNormTest, layer_norm_data_slice_test_007) {
  ge::op::LayerNorm op;

  op.UpdateInputDesc("x",
                     create_desc_with_ori({30, 256, 512}, ge::DT_FLOAT, ge::FORMAT_ND, {30, 256, 512}, ge::FORMAT_ND));
  op.UpdateInputDesc("gamma",
                     create_desc_with_ori({256, 512}, ge::DT_FLOAT, ge::FORMAT_ND, {256, 512}, ge::FORMAT_ND));
  op.UpdateInputDesc("beta",
                     create_desc_with_ori({256, 512}, ge::DT_FLOAT, ge::FORMAT_ND, {256, 512}, ge::FORMAT_ND));
  op.UpdateOutputDesc("y",
                     create_desc_with_ori({30, 256, 512}, ge::DT_FLOAT, ge::FORMAT_ND, {30, 256, 512}, ge::FORMAT_ND));

  int begin_params_axis = 1;
  op.SetAttr("begin_params_axis", begin_params_axis);

  std::vector<std::vector<int64_t>> y_data_slice = {{0, 15}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(output_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  ge::GeTensorDescPtr input_x = op_desc->MutableInputDesc("x");
  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(input_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
  ge::GeTensorDescPtr input_gamma = op_desc->MutableInputDesc("gamma");
  std::vector<std::vector<int64_t>> gamma_data_slice;
  ge::AttrUtils::GetListListInt(input_gamma, ge::ATTR_NAME_DATA_SLICE, gamma_data_slice);
  ge::GeTensorDescPtr input_beta = op_desc->MutableInputDesc("beta");
  std::vector<std::vector<int64_t>> beta_data_slice;
  ge::AttrUtils::GetListListInt(input_beta, ge::ATTR_NAME_DATA_SLICE, beta_data_slice);
  std::vector<std::vector<int64_t>> expect_x_data_slice = {{0, 15}, {}, {}};
  std::vector<std::vector<int64_t>> expect_gamma_data_slice = {{}, {}};
  std::vector<std::vector<int64_t>> expect_beta_data_slice = {{}, {}};
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  EXPECT_EQ(expect_x_data_slice, x_data_slice);
  EXPECT_EQ(expect_gamma_data_slice, gamma_data_slice);
  EXPECT_EQ(expect_beta_data_slice, beta_data_slice);
}

TEST_F(LayerNormTest, layer_norm_test_unknow_dim) {
  // input x info
  using namespace ge;
  auto input_x_shape = vector<int64_t>({-1, -1, -1});
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{100, 200}, {1, -1}, {1, -1}};
  auto input_x_dtype = DT_FLOAT;

  // attr value
  int begin_norm_axis = -1;

  // expect result
  std::vector<int64_t> expected_y_shape = input_x_shape;
  std::vector<int64_t> expected_mv_shape = {-1, -1, 1};
  std::vector<std::pair<int64_t, int64_t>> expected_y_range = shape_range;
  std::vector<std::pair<int64_t, int64_t>> expected_mv_range = {{100, 200}, {1, -1}, {1, 1}};

  // gen LayerNorm op
  auto test_op = op::LayerNorm("LayerNorm");
  TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_ND, shape_range);
  TENSOR_INPUT_WITH_SHAPE(test_op, gamma, input_x_shape, input_x_dtype, FORMAT_ND, shape_range);
  TENSOR_INPUT_WITH_SHAPE(test_op, beta, input_x_shape, input_x_dtype, FORMAT_ND, shape_range);
  test_op.SetAttr("begin_norm_axis", begin_norm_axis);

  auto ret = test_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_y_desc = test_op.GetOutputDesc("y");
  auto output_mean_desc = test_op.GetOutputDesc("mean");
  auto output_var_desc = test_op.GetOutputDesc("variance");

  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_mean_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_var_desc.GetDataType(), ge::DT_FLOAT);

  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
  EXPECT_EQ(output_mean_desc.GetShape().GetDims(), expected_mv_shape);
  EXPECT_EQ(output_var_desc.GetShape().GetDims(), expected_mv_shape);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_y_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape_range, expected_y_range);

  std::vector<std::pair<int64_t, int64_t>> output_mean_shape_range;
  EXPECT_EQ(output_mean_desc.GetShapeRange(output_mean_shape_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_mean_shape_range, expected_mv_range);
}

TEST_F(LayerNormTest, layer_norm_test_unknow_rank) {
  // input x info
  using namespace ge;
  auto input_x_shape = vector<int64_t>({-2});
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  auto input_x_dtype = DT_FLOAT;

  // attr value
  int begin_norm_axis = -1;

  // expect result
  std::vector<int64_t> expected_y_shape = input_x_shape;
  std::vector<int64_t> expected_mv_shape = input_x_shape;
  std::vector<std::pair<int64_t, int64_t>> expected_y_range = shape_range;
  std::vector<std::pair<int64_t, int64_t>> expected_mv_range = shape_range;

  // gen LayerNorm op
  auto test_op = op::LayerNorm("LayerNorm");
  TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_ND, shape_range);
  TENSOR_INPUT_WITH_SHAPE(test_op, gamma, input_x_shape, input_x_dtype, FORMAT_ND, shape_range);
  TENSOR_INPUT_WITH_SHAPE(test_op, beta, input_x_shape, input_x_dtype, FORMAT_ND, shape_range);
  test_op.SetAttr("begin_norm_axis", begin_norm_axis);

  auto ret = test_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_y_desc = test_op.GetOutputDesc("y");
  auto output_mean_desc = test_op.GetOutputDesc("mean");
  auto output_var_desc = test_op.GetOutputDesc("variance");

  EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_mean_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_var_desc.GetDataType(), ge::DT_FLOAT);

  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
  EXPECT_EQ(output_mean_desc.GetShape().GetDims(), expected_mv_shape);
  EXPECT_EQ(output_var_desc.GetShape().GetDims(), expected_mv_shape);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_y_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape_range, expected_y_range);

  std::vector<std::pair<int64_t, int64_t>> output_mean_shape_range;
  EXPECT_EQ(output_mean_desc.GetShapeRange(output_mean_shape_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_mean_shape_range, expected_mv_range);
}
