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
 * \file ragged_conversion_ops.cpp
 * \brief
 */
#include "inc/ragged_conversion_ops.h"
#include <cmath>
#include <string>
#include <string.h>
#include <vector>
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/ragged_conversion_ops_shape_fns.h"
#include "util/util.h"
#include "util/error_util.h"

namespace ge {
IMPLEMT_INFERFUNC(RaggedTensorToSparse, RaggedTensorToSparseInfer) {
  int64_t num_splits;
  std::string err_msg;
  if (op.GetAttr("RAGGED_RANK", num_splits) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("get attr[RAGGED_RANK] failed"));
    return GRAPH_FAILED;
  }
  if (num_splits < 1) {
    err_msg = ConcatString("check attr[RAGGED_RANK] failed, must >= 1, got ",
                           num_splits);
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  TensorDesc rt_dense_values_desc = op.GetInputDesc(num_splits);
  Shape rt_dense_values_shape;
  if (WithRankAtLeast(rt_dense_values_desc, 1, rt_dense_values_shape,
                      TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRankAtLeast function, input[rt_dense_values] rank "
        "must >= 1, got rank[",
        rt_dense_values_desc.GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  for (int64_t i = 0; i < num_splits; ++i) {
    TensorDesc splits_desc = op.GetDynamicInputDesc("rt_nested_splits", i);
    Shape splits_shape;
    if (WithRank(splits_desc, 1, splits_shape, TbeGetName(op).c_str()) !=
        GRAPH_SUCCESS) {
      err_msg =
          ConcatString("failed to call WithRank function, input[",
                       splits_desc.GetName(), "] rank must be 1, got rank[",
                       splits_desc.GetShape().GetDimNum(), "]");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }

  int64_t dense_dims;
  if (RankKnown(rt_dense_values_shape)) {
    dense_dims = num_splits + rt_dense_values_shape.GetDimNum();
  } else {
    dense_dims = UNKNOWN_DIM;
  }
  int64_t num_values;
  if (RankKnown(rt_dense_values_shape)) {
    bool found_unknown = false;
    int64_t size = 1;
    for (size_t i = 0; i < rt_dense_values_shape.GetDimNum(); ++i) {
      int64_t dim_val = rt_dense_values_shape.GetDim(i);
      if (dim_val == UNKNOWN_DIM) {
        found_unknown = true;
      } else {
        size *= dim_val;
      }
    }
    if (found_unknown) {
      num_values = UNKNOWN_DIM;
    } else {
      num_values = size;
    }
  } else {
    num_values = UNKNOWN_DIM;
  }

  Shape sparse_indices_output;
  if (Matrix(num_values, dense_dims, sparse_indices_output) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(
        TbeGetName(op),
        string("failed to call Matrix for num_values and dense_dims"));
    return GRAPH_FAILED;
  }
  TensorDesc sparse_indices_desc = op.GetOutputDesc("sparse_indices");
  sparse_indices_desc.SetDataType(DT_INT64);
  sparse_indices_desc.SetShape(sparse_indices_output);
  if (op.UpdateOutputDesc("sparse_indices", sparse_indices_desc) !=
      GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("update output[sparse_indices] desc failed"));
    return GRAPH_FAILED;
  }

  TensorDesc sparse_values_desc = op.GetOutputDesc("sparse_values");
  sparse_values_desc.SetDataType(rt_dense_values_desc.GetDataType());
  sparse_values_desc.SetShape(Shape({num_values}));
  if (op.UpdateOutputDesc("sparse_values", sparse_values_desc) !=
      GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("update output[sparse_values] desc failed"));
    return GRAPH_FAILED;
  }

  TensorDesc sparse_dense_shape_desc = op.GetOutputDesc("sparse_dense_shape");
  sparse_dense_shape_desc.SetDataType(DT_INT64);
  sparse_dense_shape_desc.SetShape(Shape({dense_dims}));
  if (op.UpdateOutputDesc("sparse_dense_shape", sparse_dense_shape_desc) !=
      GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("update output[sparse_dense_shape] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(RaggedTensorToSparse, RaggedTensorToSparseInfer);

IMPLEMT_INFERFUNC(RaggedTensorToTensor, RaggedTensorToTensorInfer) {
  if (RaggedTensorToTensorShapeFn(op) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
      std::string("failed to call RaggedTensorToTensorShapeFn function"));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(RaggedTensorToTensor, RaggedTensorToTensorInfer);

}  // namespace ge
