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
#include "util/common_shape_fns.h"
#include "util/util.h"

namespace ge {
IMPLEMT_INFERFUNC(RaggedRange, RaggedRangeInfer) {
  Shape starts;
  Shape limits;
  Shape deltas;
  if (WithRankAtMost(op.GetInputDesc(0), 1, starts, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input starts must be at most 1D.");
    return GRAPH_FAILED;
  }
  if (WithRankAtMost(op.GetInputDesc(1), 1, limits, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input limits must be at most 1D.");
    return GRAPH_FAILED;
  }
  if (WithRankAtMost(op.GetInputDesc(2), 1, deltas, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input deltas must be at most 1D.");
    return GRAPH_FAILED;
  }

  int64_t dim = UNKNOWN_DIM;
  int64_t starts_dim = starts.GetDim(0);
  int64_t limits_dim = limits.GetDim(0);
  int64_t deltas_dim = deltas.GetDim(0);
  if (op.GetInputDesc(0).GetShape().GetDimNum() == 1) {
    if (Merge(starts_dim, dim, dim) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "Merge starts_dim failed.");
      return GRAPH_FAILED;
    }
  }
  if (op.GetInputDesc(1).GetShape().GetDimNum() == 1) {
    if (Merge(limits_dim, dim, dim) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "Merge limits_dim failed.");
      return GRAPH_FAILED;
    }
  }
  if (op.GetInputDesc(2).GetShape().GetDimNum() == 1) {
    if (Merge(deltas_dim, dim, dim) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "Merge deltas_dim failed.");
      return GRAPH_FAILED;
    }
  }

  int64_t rt_nested_splits_dim = UNKNOWN_DIM;
  if (dim != UNKNOWN_DIM) {
    rt_nested_splits_dim = dim + 1;
  } else if (op.GetInputDesc(0).GetShape().GetDimNum() == 0 && op.GetInputDesc(1).GetShape().GetDimNum() == 0 &&
             op.GetInputDesc(2).GetShape().GetDimNum() == 0) {
    rt_nested_splits_dim = 2;
  }

  DataType Tsplits_type;
  if (op.GetAttr("Tsplits", Tsplits_type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr Tsplits error.");
    return GRAPH_FAILED;
  }
  TensorDesc nested_desc = op.GetOutputDesc("rt_nested_splits");
  nested_desc.SetShape(Shape({rt_nested_splits_dim}));
  nested_desc.SetDataType(Tsplits_type);
  if (op.UpdateOutputDesc("rt_nested_splits", nested_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "fail to update output rt_nested_splits.");
    return GRAPH_FAILED;
  }

  DataType T_type = op.GetInputDesc("starts").GetDataType();
  std::vector<int64_t> unknow_dim_vec(1, UNKNOWN_DIM);
  TensorDesc dense_desc = op.GetOutputDesc("rt_dense_values");
  dense_desc.SetShape(Shape(unknow_dim_vec));
  dense_desc.SetDataType(T_type);
  if (op.UpdateOutputDesc("rt_dense_values", dense_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "fail to update output rt_dense_values.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(RaggedRange, RaggedRangeInfer);

}  // namespace ge
