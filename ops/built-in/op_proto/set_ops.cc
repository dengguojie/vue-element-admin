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
 * \file set_ops.cpp
 * \brief
 */
#include "inc/set_ops.h"
#include "graph/operator.h"
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/util.h"
#include "util/error_util.h"

namespace ge {
IMPLEMT_INFERFUNC(DenseToDenseSetOperation, DenseToDenseSetOperationInfer) {
  std::string err_msg;
  if (op.GetInputsSize() != 2) {
    err_msg =
        ConcatString("input size should be 2, got[", op.GetInputsSize(), "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  auto x1_tensor = op.GetInputDesc(0);
  auto x2_tensor = op.GetInputDesc(1);
  Shape x1_shape = x1_tensor.GetShape();
  int64_t output_rank_dim = 0;
  if (WithRankAtLeast(x1_tensor, 2, x1_shape, op.GetName().c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(0, DebugString(x1_tensor.GetShape().GetDims()),
                             "at least 2D");
    err_msg = string("failed to call WithRankAtLeast function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (RankKnown(x1_shape)) {
    const int64_t x1_rank = static_cast<int64_t>(x1_shape.GetDimNum());
    Shape x2_shape;
    if (WithRank(x2_tensor, x1_rank, x2_shape, op.GetName().c_str()) !=
        GRAPH_SUCCESS) {
      err_msg =
          ConcatString("input[x2] rank[", x2_tensor.GetShape().GetDimNum(),
                       "] must match input[x1] rank[", x1_rank, "]");
      err_msg = string("failed to call WithRank function, ") + err_msg;
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    if (RankKnown(x2_shape)) {
      const int64_t x2_rank = static_cast<int64_t>(x2_shape.GetDimNum());
      Shape group_x1_shape;
      graphStatus status = SubShape(x1_shape, 0, x2_rank - 1, 1, group_x1_shape,
                                    op.GetName().c_str());
      if (status != GRAPH_SUCCESS) {
        err_msg = ConcatString(
            "failed to call SubShape function to get subshape from 0 to",
            x2_rank - 1, " of input[x1] shape",
            DebugString(x1_shape.GetDims()));
        AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
      }
      Shape group_x2_shape;
      status = SubShape(x2_shape, 0, x2_rank - 1, 1, group_x2_shape,
                        op.GetName().c_str());
      if (status != GRAPH_SUCCESS) {
        err_msg = ConcatString(
            "failed to call SubShape function to get subshape from 0 to",
            x2_rank - 1, " of input[x2] shape",
            DebugString(x2_shape.GetDims()));
        AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
      }
      Shape unused_shape;
      status = Merge(group_x1_shape, group_x2_shape, unused_shape,
                     op.GetName().c_str());
      if (status != GRAPH_SUCCESS) {
        err_msg = ConcatString(
            "failed to call Merge function to merge input[x1] subshape",
            DebugString(group_x1_shape.GetDims()), "and input[x2] subshape",
            DebugString(group_x2_shape.GetDims()));
        AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
      }
    }
    output_rank_dim = x1_rank;
  } else {
    Shape x2_shape = x2_tensor.GetShape();
    if (WithRankAtLeast(x2_tensor, 2, x2_shape, op.GetName().c_str()) !=
        GRAPH_SUCCESS) {
      err_msg = GetShapeErrMsg(1, DebugString(x2_tensor.GetShape().GetDims()),
                               "at least 2D");
      err_msg = string("failed to call WithRankAtLeast function, ") + err_msg;
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    if (RankKnown(x2_shape)) {
      output_rank_dim = static_cast<int64_t>(x2_shape.GetDimNum());
    } else {
      output_rank_dim = ge::UNKNOWN_DIM;
    }
  }

  DataType output_type = x1_tensor.GetDataType();
  if (output_type != x2_tensor.GetDataType()) {
    err_msg = ConcatString("input[x1] data type[", DTypeStr(output_type),
                           "] must match input[x2] data type[",
                           DTypeStr(x2_tensor.GetDataType()), "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  Shape output_indices;
  Shape output_values;
  Shape output_shape;
  (void)Matrix(ge::UNKNOWN_DIM, output_rank_dim, output_indices);
  (void)Vector(ge::UNKNOWN_DIM, output_values);
  (void)Vector(output_rank_dim, output_shape);

  TensorDesc y_indices_tensor = op.GetOutputDesc("y_indices");
  y_indices_tensor.SetShape(output_indices);
  y_indices_tensor.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("y_indices", y_indices_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("fail to update output[y_indices]."));
    return GRAPH_FAILED;
  }
  TensorDesc y_values_tensor = op.GetOutputDesc("y_values");
  y_values_tensor.SetShape(output_values);
  y_values_tensor.SetDataType(output_type);
  if (op.UpdateOutputDesc("y_values", y_values_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("fail to update output[y_values]."));
    return GRAPH_FAILED;
  }
  TensorDesc y_shape_tensor = op.GetOutputDesc("y_shape");
  y_shape_tensor.SetShape(output_shape);
  y_shape_tensor.SetDataType(DT_INT64);
  return op.UpdateOutputDesc("y_shape", y_shape_tensor);
}

INFER_FUNC_REG(DenseToDenseSetOperation, DenseToDenseSetOperationInfer);

IMPLEMT_INFERFUNC(SetSize, SetSizeInfer) {
  Shape unknown_shape(ge::UNKNOWN_SHAPE);

  TensorDesc output_desc = op.GetOutputDesc("size");
  output_desc.SetShape(unknown_shape);
  output_desc.SetDataType(DT_INT32);
  return op.UpdateOutputDesc("size", output_desc);
}

INFER_FUNC_REG(SetSize, SetSizeInfer);

IMPLEMT_INFERFUNC(DenseToSparseSetOperation, DenseToSparseSetOperationInfer) {
  std::string err_msg;
  if (op.GetInputsSize() != 4) {
    err_msg =
        ConcatString("input size should be 4, got[", op.GetInputsSize(), "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  auto x2_indices_tensor = op.GetInputDesc(1);
  auto x2_values_tensor = op.GetInputDesc(2);
  auto x2_shape_tensor = op.GetInputDesc(3);
  graphStatus status =
      ValidateSparseTensor(x2_indices_tensor, x2_values_tensor, x2_shape_tensor,
                           op.GetName().c_str());
  if (status != GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call ValidateSparseTensor function, check validate sparse "
        "parameters failed");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x1_desc = op_desc->MutableInputDesc(0);
  GeShape x1_shape;
  Shape x2_shape = x2_shape_tensor.GetShape();
  int64_t output_rank_dim = 0;
  const int64_t input_rank_dim = x2_shape.GetDim(0);
  if (WithRankAtLeast(x1_desc, 2, x1_shape, op.GetName().c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(0, DebugString(x1_desc->GetShape().GetDims()),
                             "at least 2D");
    err_msg = string("failed to call WithRankAtLeast function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (RankKnown(x1_shape)) {
    int64_t x1_rank = static_cast<int64_t>(x1_shape.GetDimNum());
    if (input_rank_dim != x1_rank) {
      err_msg =
          ConcatString("input[x1] rank[", x1_rank,
                       "] must match input[x2] dim0[", input_rank_dim, "]");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    output_rank_dim = x1_rank;
  } else if (ValueKnown(x2_shape, 0)) {
    output_rank_dim = input_rank_dim;
  } else {
    output_rank_dim = ge::UNKNOWN_DIM;
  }

  Shape output_indices;
  Shape output_values;
  Shape output_shape;
  (void)Matrix(ge::UNKNOWN_DIM, output_rank_dim, output_indices);
  (void)Vector(ge::UNKNOWN_DIM, output_values);
  (void)Vector(output_rank_dim, output_shape);

  DataType x1_type = op.GetInputDesc("x1").GetDataType();

  TensorDesc y_indices_tensor = op.GetOutputDesc("y_indices");
  y_indices_tensor.SetShape(output_indices);
  y_indices_tensor.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("y_indices", y_indices_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("fail to update output[y_indices]."));
    return GRAPH_FAILED;
  }
  TensorDesc y_values_tensor = op.GetOutputDesc("y_values");
  y_values_tensor.SetShape(output_values);
  y_values_tensor.SetDataType(x1_type);
  if (op.UpdateOutputDesc("y_values", y_values_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("fail to update output[y_values]."));
    return GRAPH_FAILED;
  }
  TensorDesc y_shape_tensor = op.GetOutputDesc("y_shape");
  y_shape_tensor.SetShape(output_shape);
  y_shape_tensor.SetDataType(DT_INT64);
  return op.UpdateOutputDesc("y_shape", y_shape_tensor);
}

INFER_FUNC_REG(DenseToSparseSetOperation, DenseToSparseSetOperationInfer);

IMPLEMT_INFERFUNC(SparseToSparseSetOperation, SparseToSparseSetOperationInfer) {
  int64_t inputs_size = op.GetInputsSize();
  std::string err_msg;
  if (inputs_size != 6) {
    err_msg = ConcatString("inputs size must be 6, got ", inputs_size);
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  auto x1_indices_tensor = op.GetInputDesc(0);
  auto x1_values_tensor = op.GetInputDesc(1);
  auto x1_shape_tensor = op.GetInputDesc(2);
  auto x2_indices_tensor = op.GetInputDesc(3);
  auto x2_values_tensor = op.GetInputDesc(4);
  auto x2_shape_tensor = op.GetInputDesc(5);
  graphStatus status =
      ValidateSparseTensor(x1_indices_tensor, x1_values_tensor, x1_shape_tensor,
                           op.GetName().c_str());
  if (status != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(
        op.GetName(),
        string("failed to call ValidateSparseTensor function for "
               "input[x1_indices], input[x1_values], input[x1_shape]."));
    return GRAPH_FAILED;
  }
  status = ValidateSparseTensor(x2_indices_tensor, x2_values_tensor,
                                x2_shape_tensor, op.GetName().c_str());
  if (status != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(
        op.GetName(),
        string("failed to call ValidateSparseTensor function for "
               "input[x2_indices], input[x2_values], input[x2_shape]."));
    return GRAPH_FAILED;
  }

  Shape x1_shape = x1_shape_tensor.GetShape();
  Shape x2_shape = x2_shape_tensor.GetShape();
  int64_t output_rank_dim = 0;
  const int64_t input_rank_dim2 = x2_shape.GetDim(0);
  if (ValueKnown(x1_shape, 0)) {
    const int64_t input_rank_dim1 = x1_shape.GetDim(0);
    if (input_rank_dim1 < 2) {
      err_msg = ConcatString("input[x1_shape] rank must > = 2, got rank[",
                             input_rank_dim1, "]");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    if (input_rank_dim1 != input_rank_dim2) {
      err_msg = ConcatString("input[x1_shape] rank[", input_rank_dim1,
                             "] is not equal to input[x2_shape] rank[",
                             input_rank_dim2, "]");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    output_rank_dim = input_rank_dim1;
  } else if (ValueKnown(x2_shape, 0)) {
    if (input_rank_dim2 < 2) {
      err_msg = ConcatString("input[x2_shape] rank must > = 2, got rank[",
                             input_rank_dim2, "]");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    output_rank_dim = input_rank_dim2;
  } else {
    output_rank_dim = ge::UNKNOWN_DIM;
  }

  Shape output_indices;
  Shape output_values;
  Shape output_shape;
  (void)Matrix(ge::UNKNOWN_DIM, output_rank_dim, output_indices);
  (void)Vector(ge::UNKNOWN_DIM, output_values);
  (void)Vector(output_rank_dim, output_shape);

  DataType x1_values_type = op.GetInputDesc("x1_values").GetDataType();

  TensorDesc y_indices_tensor = op.GetOutputDesc("y_indices");
  y_indices_tensor.SetShape(output_indices);
  y_indices_tensor.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("y_indices", y_indices_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[y_indices] failed."));
    return GRAPH_FAILED;
  }
  TensorDesc y_values_tensor = op.GetOutputDesc("y_values");
  y_values_tensor.SetShape(output_values);
  y_values_tensor.SetDataType(x1_values_type);
  if (op.UpdateOutputDesc("y_values", y_values_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[y_values] failed."));
    return GRAPH_FAILED;
  }
  TensorDesc y_shape_tensor = op.GetOutputDesc("y_shape");
  y_shape_tensor.SetShape(output_shape);
  y_shape_tensor.SetDataType(DT_INT64);
  return op.UpdateOutputDesc("y_shape", y_shape_tensor);
}

INFER_FUNC_REG(SparseToSparseSetOperation, SparseToSparseSetOperationInfer);

}  // namespace ge
