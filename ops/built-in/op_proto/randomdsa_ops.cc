/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file randomdsa_ops.cpp
 * \brief
 */
#include <string>
#include <vector>
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/random_ops_shape_fns.h"
#include "util/util.h"
#include "util/error_util.h"
#include "util/vector_proto_profiling.h"
#include "graph/utils/node_utils.h"
#include "register/infer_data_slice_registry.h"
#include "graph/debug/ge_attr_define.h"
#include "inc/randomdsa_ops.h"
#include "op_const.h"

namespace {
constexpr int32_t TRUNCATE_MEAN_IDX = 2;
constexpr int32_t UNIFORM_LOW_IDX = 2;
constexpr int32_t NORMAL_MEAN_IDX = 2;
}  // namespace

namespace ge {
IMPLEMT_INFERFUNC(DSAGenBitMask, DSAGenBitMaskInfer) {
  const vector<string> depend_names = {"count"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  auto node_arg = NodeUtils::GetNodeFromOperator(op);
  auto op_info_arg = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc_count = op_info_arg->GetInputDescPtr(0);
  auto out_desc = op_info_arg->MutableOutputDesc(0);

  if (input_desc_count == nullptr || out_desc == nullptr) {
    OP_LOGE(TbeGetName(op), "[TBE Compiler] Get null node ptr");
    return GRAPH_FAILED;
  }

  // get count shape
  const auto& count_shape = input_desc_count->GetShape();
  if (count_shape.IsUnknownShape()) {
    out_desc->SetShape(count_shape);
    return GRAPH_SUCCESS;
  }

  // get const count value
  vector<int64_t> count_value;
  if (ops::GetConstIntData(op, 0, count_value)) {
    auto const_dtype = input_desc_count->GetDataType();
    // verify dimension_value
    if (count_value.size() != 1) {
      string error_msg =
          ConcatString("the element size of input[dimension] should be equal to 1, but get ", count_value.size(), ".");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
      return GRAPH_FAILED;
    }
    vector<int64_t> output_shape;
    output_shape = count_value;
    out_desc->SetShape(GeShape(output_shape));

    // decide if dynamic and change range
    if (IsUnknown(output_shape)) {
      std::vector<std::pair<int64_t, int64_t>> input_range;
      input_desc_count->GetShapeRange(input_range);
      out_desc->SetShapeRange(input_range);
      return GRAPH_SUCCESS;
    }
  }

  // get and set output dtype
  out_desc->SetDataType(DT_UINT8);
  // set shape range
  std::vector<std::pair<int64_t, int64_t>> input_range;
  input_desc_count->GetShapeRange(input_range);
  out_desc->SetShapeRange(input_range);
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(DSAGenBitMask, DSAGenBitMaskInfer);

IMPLEMT_INFERFUNC(DSARandomUniform, DSARandomUniformInfer) {
  const vector<string> depend_names = {"count"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  auto node_arg = NodeUtils::GetNodeFromOperator(op);
  auto op_info_arg = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc_count = op_info_arg->GetInputDescPtr(0);
  auto input_desc_low = op_info_arg->GetInputDescPtr(UNIFORM_LOW_IDX);
  auto out_desc = op_info_arg->MutableOutputDesc(0);
  auto out_dst_dtype = input_desc_low->GetDataType();

  // get and set output dtype
  if (input_desc_low != nullptr) {
    out_desc->SetDataType(out_dst_dtype);
  } else {
    OP_LOGW(TbeGetName(op).c_str(), "get dtype failed.");
    out_desc->SetDataType(DT_INT64);
  }
  // get count shape
  const auto& count_shape = input_desc_count->GetShape();
  if (count_shape.IsUnknownShape()) {
    out_desc->SetShape(count_shape);
    return GRAPH_SUCCESS;
  }

  // get const count value
  vector<int64_t> count_value;
  if (ops::GetConstIntData(op, 0, count_value)) {
    auto const_dtype = input_desc_count->GetDataType();
    // verify dimension_value
    if (count_value.size() != 1) {
      string error_msg =
          ConcatString("the element size of input[dimension] should be equal to 1, but get ", count_value.size(), ".");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
      return GRAPH_FAILED;
    }
    vector<int64_t> output_shape;
    output_shape = count_value;
    out_desc->SetShape(GeShape(output_shape));

    // decide if dynamic and change range
    if (IsUnknown(output_shape)) {
      std::vector<std::pair<int64_t, int64_t>> input_range;
      input_desc_count->GetShapeRange(input_range);
      out_desc->SetShapeRange(input_range);
      return GRAPH_SUCCESS;
    }
  }

  // set shape range
  std::vector<std::pair<int64_t, int64_t>> input_range;
  input_desc_count->GetShapeRange(input_range);
  out_desc->SetShapeRange(input_range);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DSARandomUniform, DSARandomUniformInfer);

IMPLEMT_INFERFUNC(DSARandomTruncatedNormal, DSARandomTruncatedNormalInfer) {
  const vector<string> depend_names = {"count"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  auto node_arg = NodeUtils::GetNodeFromOperator(op);
  auto op_info_arg = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc_count = op_info_arg->GetInputDescPtr(0);
  auto input_desc_low = op_info_arg->GetInputDescPtr(TRUNCATE_MEAN_IDX);
  auto out_desc = op_info_arg->MutableOutputDesc(0);
  auto out_dst_dtype = input_desc_low->GetDataType();

  // get and set output dtype
  if (input_desc_low != nullptr) {
    out_desc->SetDataType(out_dst_dtype);
  } else {
    OP_LOGW(TbeGetName(op).c_str(), "get dtype failed.");
    out_desc->SetDataType(DT_INT64);
  }
  // get count shape
  const auto& count_shape = input_desc_count->GetShape();
  if (count_shape.IsUnknownShape()) {
    out_desc->SetShape(count_shape);
    return GRAPH_SUCCESS;
  }

  // get const count value
  vector<int64_t> count_value;
  if (ops::GetConstIntData(op, 0, count_value)) {
    auto const_dtype = input_desc_count->GetDataType();
    // verify dimension_value
    if (count_value.size() != 1) {
      string error_msg =
          ConcatString("the element size of input[dimension] should be equal to 1, but get ", count_value.size(), ".");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
      return GRAPH_FAILED;
    }

    vector<int64_t> output_shape;
    output_shape = count_value;
    out_desc->SetShape(GeShape(output_shape));

    // decide if dynamic and change range
    if (IsUnknown(output_shape)) {
      std::vector<std::pair<int64_t, int64_t>> input_range;
      input_desc_count->GetShapeRange(input_range);
      out_desc->SetShapeRange(input_range);
      return GRAPH_SUCCESS;
    }
  }

  // set shape range
  std::vector<std::pair<int64_t, int64_t>> input_range;
  input_desc_count->GetShapeRange(input_range);
  out_desc->SetShapeRange(input_range);
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(DSARandomTruncatedNormal, DSARandomTruncatedNormalInfer);

IMPLEMT_INFERFUNC(DSARandomNormal, DSARandomNormalInfer) {
  const vector<string> depend_names = {"count"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  auto node_arg = NodeUtils::GetNodeFromOperator(op);
  auto op_info_arg = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc_count = op_info_arg->GetInputDescPtr(0);
  auto input_desc_low = op_info_arg->GetInputDescPtr(NORMAL_MEAN_IDX);
  auto out_desc = op_info_arg->MutableOutputDesc(0);
  auto out_dst_dtype = input_desc_low->GetDataType();

  // get and set output dtype
  if (input_desc_low != nullptr) {
    out_desc->SetDataType(out_dst_dtype);
  } else {
    OP_LOGW(TbeGetName(op).c_str(), "get dtype failed.");
    out_desc->SetDataType(DT_INT64);
  }
  // get count shape
  const auto& count_shape = input_desc_count->GetShape();
  if (count_shape.IsUnknownShape()) {
    out_desc->SetShape(count_shape);
    return GRAPH_SUCCESS;
  }

  // get const count value
  vector<int64_t> count_value;
  if (ops::GetConstIntData(op, 0, count_value)) {
    auto const_dtype = input_desc_count->GetDataType();
    // verify dimension_value
    if (count_value.size() != 1) {
      string error_msg =
          ConcatString("the element size of input[dimension] should be equal to 1, but get ", count_value.size(), ".");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
      return GRAPH_FAILED;
    }

    vector<int64_t> output_shape;
    output_shape = count_value;
    out_desc->SetShape(GeShape(output_shape));

    // decide if dynamic and change range
    if (IsUnknown(output_shape)) {
      std::vector<std::pair<int64_t, int64_t>> input_range;
      input_desc_count->GetShapeRange(input_range);
      out_desc->SetShapeRange(input_range);
      return GRAPH_SUCCESS;
    }
  }

  // set shape range
  std::vector<std::pair<int64_t, int64_t>> input_range;
  input_desc_count->GetShapeRange(input_range);
  out_desc->SetShapeRange(input_range);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DSARandomNormal, DSARandomNormalInfer);

}  // namespace ge
