/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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

/*!
 * \file cluster.cc
 * \brief
 */
#include "inc/cluster.h"
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/error_util.h"
#include "util/util.h"

namespace ge {
// ------------ KMeansCentroids -----------------
IMPLEMT_INFERFUNC(KMeansCentroids, KMeansCentroidsInfer) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  OP_LOGI(op_name.GetString(), "Enter KMeansCentroids inferfunction.");

  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::ConstGeTensorDescPtr input_x_desc = op_desc->GetInputDescPtr(0);
  if (input_x_desc == nullptr) {
    OP_LOGE(op_name.GetString(), "get input x failed.");
    return GRAPH_FAILED;
  }

  ge::ConstGeTensorDescPtr input_y_desc = op_desc->GetInputDescPtr(1);
  if (input_y_desc == nullptr) {
    OP_LOGE(op_name.GetString(), "get input y failed.");
    return GRAPH_FAILED;
  }

  ge::ConstGeTensorDescPtr input_ssy_desc = op_desc->GetInputDescPtr(2);
  if (input_ssy_desc == nullptr) {
    OP_LOGE(op_name.GetString(), "get input sum_square_y failed.");
    return GRAPH_FAILED;
  }

  bool use_actual_distance = false;
  if (op.GetAttr("use_actual_distance", use_actual_distance) != GRAPH_SUCCESS) {
    OP_LOGW(op_name.GetString(), "Failed to get attr[use_actual_distance]. Set it to false.");
  }

  auto output_segment_sum = op_desc->MutableOutputDesc(0);
  if (output_segment_sum == nullptr) {
    OP_LOGE(op_name.GetString(), "get output segment_sum failed.");
    return GRAPH_FAILED;
  }

  auto output_segment_count = op_desc->MutableOutputDesc(1);
  if (output_segment_count == nullptr) {
    OP_LOGE(op_name.GetString(), "get output segment_count failed.");
    return GRAPH_FAILED;
  }

  auto output_kmean_total_sum = op_desc->MutableOutputDesc(2);
  if (output_kmean_total_sum == nullptr) {
    OP_LOGE(op_name.GetString(), "get output kmean_total_sum failed.");
    return GRAPH_FAILED;
  }

  const GeShape &input_x_shape = input_x_desc->GetShape();
  const GeShape &input_y_shape = input_y_desc->GetShape();
  const GeShape &input_ssy_shape = input_ssy_desc->GetShape();
  GeShape &output_segment_sum_shape = output_segment_sum->MutableShape();
  GeShape &output_segment_count_shape = output_segment_count->MutableShape();
  GeShape &output_kmean_total_sum_shape = output_kmean_total_sum->MutableShape();

  int64_t x_m = 0;
  int64_t x_d = 0;
  int64_t y_n = 0;
  int64_t y_d = 0;
  int64_t n_ssy = 0;
  int64_t m_ssx = 0;

  x_m = input_x_shape.GetDim(0);
  x_d = input_x_shape.GetDim(1);
  y_n = input_y_shape.GetDim(0);
  y_d = input_y_shape.GetDim(1);
  n_ssy = input_ssy_shape.GetDim(1);

  if (x_d != y_d){
    OP_LOGE(op_name.GetString(),
            "The second dimension of input x should be equal to the second dimension of input y");
    return GRAPH_FAILED;
  }

  if (y_n != n_ssy){
    OP_LOGE(op_name.GetString(),
            "The first dimension of input y should be equal to the second dimension of input sum_square_y");
    return GRAPH_FAILED;
  }

  if (use_actual_distance == true){
    ge::ConstGeTensorDescPtr input_ssx_desc = op_desc->GetInputDescPtr(3);
    if (input_ssx_desc == nullptr) {
      OP_LOGE(op_name.GetString(), "get input sum_square_x failed.");
      return GRAPH_FAILED;
    }
    const GeShape &input_ssx_shape = input_ssx_desc->GetShape();
    m_ssx = input_ssx_shape.GetDim(0);
    if (x_m != m_ssx){
      OP_LOGE(op_name.GetString(),
              "The first dimension of input x should be equal to the first dimension of input sum_square_x");
      return GRAPH_FAILED;
    }
  }

  int64_t m_tail = x_m % 16;
  int64_t d_tail = x_d % 16;
  int64_t n_tail = y_n % 16;
  if (m_tail != 0 || n_tail != 0 || d_tail != 0){
    OP_LOGE(op_name.GetString(), "input shape non-16 alignment");
    return GRAPH_FAILED;
  }

  output_segment_sum_shape.SetDim(0, y_n);
  output_segment_sum_shape.SetDim(1, y_d);
  output_segment_count_shape.SetDim(0, y_n);
  output_segment_count_shape.SetDim(1, 1);
  output_kmean_total_sum_shape.SetDim(0, 1);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(KMeansCentroids, KMeansCentroidsInfer);
// ------------ KMeansCentroids END -----------------
}  // namespace ge
