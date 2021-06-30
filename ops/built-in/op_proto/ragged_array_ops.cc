/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file ragged_array_ops.cpp
 * \brief
 */
#include "inc/ragged_array_ops.h"
#include <utility>
#include "graph/operator.h"
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/util.h"
#include "util/error_util.h"

namespace ge {
IMPLEMT_INFERFUNC(RaggedGather, RaggedGatherInfer) {
  int num_splits;
  int64_t PARAMS_RAGGED_RANK;
  if (op.GetAttr("PARAMS_RAGGED_RANK", PARAMS_RAGGED_RANK) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  if (op.GetAttr("OUTPUT_RAGGED_RANK", num_splits) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  Shape indices;
  if (WithRank(op.get_input_desc_indices(), num_splits - PARAMS_RAGGED_RANK + 1, indices, op.GetName().c_str()) !=
      GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
      DebugString(op.get_input_desc_indices().GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  for (int64_t i = 0; i < PARAMS_RAGGED_RANK; ++i) {
    Shape indices;
    if (WithRank(op.GetDynamicInputDesc("params_nested_splits", i), 1, indices, op.GetName().c_str()) !=
        GRAPH_SUCCESS) {
      std::string err_msg = GetShapeErrMsg(0,
        DebugString(op.GetDynamicInputDesc("params_nested_splits", i).GetShape().GetDims()), "1D");
      err_msg = string("failed to call WithRank, ") + err_msg;
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }
 
  Shape params_dense_values;
  if (WithRankAtLeast(op.get_input_desc_params_dense_values(), 1, params_dense_values, op.GetName().c_str()) !=
      GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        0, DebugString(op.get_input_desc_params_dense_values().GetShape().GetDims()), "at least 1D");
    err_msg = string("failed to call WithRankAtLeast function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  DataType Tsplits_type;
  if (op.GetAttr("Tsplits", Tsplits_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
      op.GetName(), string("failed to get attr[Tsplits]"));
  }
  DataType Tvalues_type = op.GetInputDesc("params_dense_values").GetDataType();
  std::vector<int64_t> unknow_dim_vec(1, UNKNOWN_DIM);
  for (int64_t i = 0; i < num_splits; ++i) {
    TensorDesc y_tensor = op.GetDynamicOutputDesc("output_nested_splits", i);
    y_tensor.SetDataType(Tsplits_type);
    y_tensor.SetShape(Shape(unknow_dim_vec));
    if (op.UpdateDynamicOutputDesc("output_nested_splits", i, y_tensor) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("fail to update output[output_nested_splits] desc."));
      return GRAPH_FAILED;
    }
  }

  Shape value(ge::UNKNOWN_SHAPE);
  Shape values(ge::UNKNOWN_SHAPE);
  if (SubShape(params_dense_values, 1, params_dense_values.GetDimNum(), 1, value, op.GetName().c_str()) !=
      GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call SubShape function to subshape from 1 to",
      params_dense_values.GetDimNum(), " of input[params_dense_values] shape", DebugString(params_dense_values.GetDims()));
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (Concatenate(Shape(unknow_dim_vec), value, values) != GRAPH_SUCCESS) {
    std:: string err_msg = ConcatString("failed to call Concatenate function, output shape",
      DebugString(Shape(unknow_dim_vec).GetDims())," and another shape", DebugString(value.GetDims()));
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc outputDesc = op.GetOutputDesc("output_dense_values");

  if(op.GetInputDesc("params_dense_values").GetShape().GetShapeSize() == UNKNOWN_DIM){
    outputDesc.SetShape(Shape({UNKNOWN_DIM}));
    outputDesc.SetDataType(Tvalues_type);
    op.UpdateOutputDesc("output_dense_values",outputDesc);
    return GRAPH_SUCCESS;
  }else{
    std::vector<std::pair<int64_t,int64_t>> range;
    int64_t min_dim = 1;
    std::vector<int64_t> max_dims = op.GetInputDesc("params_dense_values").GetShape().GetDims();
    for(const auto &maxdim:max_dims){
      auto p = std::make_pair(min_dim,maxdim);
      range.push_back(p);
    }
    std::vector<int64_t> output_dims;
    output_dims.push_back(ge::UNKNOWN_DIM);

    outputDesc.SetShape(Shape(output_dims));
    outputDesc.SetShapeRange(range);
    outputDesc.SetDataType(Tvalues_type);
  }

  if (op.UpdateOutputDesc("output_dense_values", outputDesc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), string("fail to update output[output_dense_values] desc."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(RaggedGather, RaggedGatherInfer);

}  // namespace ge
