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
 * \file ragged_math_ops.cpp
 * \brief
 */
#include "inc/ragged_math_ops.h"
#include <utility>
#include "graph/operator.h"
#include "op_log.h"
#include "util/error_util.h"
#include "util/common_shape_fns.h"
#include "util/util.h"

namespace ge {
IMPLEMT_INFERFUNC(RaggedRange, RaggedRangeInfer) {
  Shape starts;
  Shape limits;
  Shape deltas;
  if (WithRankAtMost(op.GetInputDesc(0), 1, starts, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call WithRankAtMost function, ",
                                       "input[starts] rank must be at most 1D, got rank[",
                                       op.GetInputDesc(0).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRankAtMost(op.GetInputDesc(1), 1, limits, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call WithRankAtMost function, ",
                                       "input[limits] rank must be at most 1D, got rank[",
                                       op.GetInputDesc(1).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRankAtMost(op.GetInputDesc(2), 1, deltas, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call WithRankAtMost function, input[deltas] ", 
                                       "rank must be at most 1D, got rank[", 
                                       op.GetInputDesc(2).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t dim = ge::UNKNOWN_DIM;
  int64_t starts_dim = starts.GetDim(0);
  int64_t limits_dim = limits.GetDim(0);
  int64_t deltas_dim = deltas.GetDim(0);
  if (op.GetInputDesc(0).GetShape().GetDimNum() == 1) {
    if (Merge(starts_dim, dim, dim) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString("failed to call Merge function, the 0th dim[",
                                         starts_dim, "] of input[starts] not equal UNKNOWN_DIM");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }
  if (op.GetInputDesc(1).GetShape().GetDimNum() == 1) {
    if (Merge(limits_dim, dim, dim) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString("failed to call Merge function, the 0th dim[",
                                         limits_dim, "] of input[limits] not equal UNKNOWN_DIM");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }
  if (op.GetInputDesc(2).GetShape().GetDimNum() == 1) {
    if (Merge(deltas_dim, dim, dim) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString("failed to call Merge function, the 0th dim[",
                                         deltas_dim, "] of input[deltas] not equal UNKNOWN_DIM");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }

  int64_t rt_nested_splits_dim = ge::UNKNOWN_DIM;
  if (dim != ge::UNKNOWN_DIM) {
    rt_nested_splits_dim = dim + 1;
  } else if (op.GetInputDesc(0).GetShape().GetDimNum() == 0 && op.GetInputDesc(1).GetShape().GetDimNum() == 0 &&
             op.GetInputDesc(2).GetShape().GetDimNum() == 0) {
    rt_nested_splits_dim = 2;
  }

  DataType Tsplits_type;
  if (op.GetAttr("Tsplits", Tsplits_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("get attr[Tsplits] failed"));
    return GRAPH_FAILED;
  }
  TensorDesc rt_nested_desc = op.GetOutputDesc("rt_nested_splits");
  rt_nested_desc.SetShape(Shape({rt_nested_splits_dim}));
  rt_nested_desc.SetDataType(Tsplits_type);
  (void)op.UpdateOutputDesc("rt_nested_splits", rt_nested_desc);

  DataType T_type = op.GetInputDesc("starts").GetDataType();
  std::vector<int64_t> unknow_dim_vec(1, UNKNOWN_DIM);
  TensorDesc dense_desc = op.GetOutputDesc("rt_dense_values");
  dense_desc.SetShape(Shape(unknow_dim_vec));
  dense_desc.SetDataType(T_type);
  (void)op.UpdateOutputDesc("rt_dense_values", dense_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(RaggedRange, RaggedRangeInfer);

}  // namespace ge
