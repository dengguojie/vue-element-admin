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
 * \file sparse_ops.cpp
 * \brief
 */
#include "inc/sparse_ops.h"
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/error_util.h"
#include "graph/utils/op_desc_utils.h"
#include "util/error_util.h"
#include "util/util.h"

namespace ge {
IMPLEMT_INFERFUNC(SparseSoftmax, SparseSoftmaxInfer) {
  Shape values;
  Shape unused;
  std::string err_msg;
  if (WithRank(op.GetInputDesc(0), 2, unused, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[indices] rank must be 2, got "
        "rank[", op.GetInputDesc(0).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 1, values, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[values] rank must be 1, got "
        "rank[", op.GetInputDesc(1).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(2), 1, unused, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[shape] rank must be 1, got "
        "rank[", op.GetInputDesc(2).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto type = op.GetInputDesc("values").GetDataType();
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(values);
  output_desc.SetDataType(type);
  return op.UpdateOutputDesc("y", output_desc);
}

INFER_FUNC_REG(SparseSoftmax, SparseSoftmaxInfer);

IMPLEMT_INFERFUNC(SparseTensorDenseAdd, SparseTensorDenseAddInfer) {
  TensorDesc desc = op.GetInputDesc(3);
  Shape shape = desc.GetShape();

  auto type = op.GetInputDesc("x1_values").GetDataType();
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(shape);
  output_desc.SetDataType(type);
  return op.UpdateOutputDesc("y", output_desc);
}

INFER_FUNC_REG(SparseTensorDenseAdd, SparseTensorDenseAddInfer);

IMPLEMT_INFERFUNC(AddSparseToTensorsMap, AddSparseToTensorsMapInfer) {
  Shape unused;
  if (WithRank(op.GetInputDesc(0), 2, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(op.GetInputDesc(0).GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 1, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(op.GetInputDesc(1).GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(2), 1, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2,
        DebugString(op.GetInputDesc(2).GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  Shape shape;
  if (Scalar(shape) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString("failed to call function Scalar to create a scalar shape");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc output_desc = op.GetOutputDesc("handle");
  output_desc.SetShape(shape);
  output_desc.SetDataType(DT_INT64);
  return op.UpdateOutputDesc("handle", output_desc);
}

INFER_FUNC_REG(AddSparseToTensorsMap, AddSparseToTensorsMapInfer);

IMPLEMT_INFERFUNC(SparseSliceGrad, SparseSliceGradInfer) {
  Shape indices_shape;
  if (WithRank(op.GetInputDesc(1), 2, indices_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call WithRank function, input[indices] rank must be 2, got "
        "rank[", op.GetInputDesc(1).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_PARAM_INVALID;
  }
  TensorDesc desc = op.GetOutputDesc("y_grad");
  std::vector<int64_t> new_dims;
  new_dims.push_back(indices_shape.GetDim(0));
  desc.SetShape(Shape(new_dims));
  DataType type = op.GetInputDesc("backprop_val_grad").GetDataType();
  desc.SetDataType(type);
  return op.UpdateOutputDesc("y_grad", desc);
}

INFER_FUNC_REG(SparseSliceGrad, SparseSliceGradInfer);

IMPLEMT_INFERFUNC(SparseSlice, SparseSliceInfer) {
  Shape shape_shape = op.GetInputDesc(2).GetShape();
  TensorDesc y_indices_desc = op.GetOutputDesc("y_indices");
  std::vector<int64_t> new_dims;
  new_dims.push_back(UNKNOWN_DIM);
  new_dims.push_back(shape_shape.GetDim(0));
  y_indices_desc.SetShape(Shape(new_dims));
  y_indices_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("y_indices", y_indices_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("update output[y_indices] desc failed."));
    return GRAPH_FAILED;
  }
  TensorDesc y_value_desc = op.GetOutputDesc("y_values");
  new_dims.clear();
  new_dims.push_back(UNKNOWN_DIM);
  y_value_desc.SetShape(Shape(new_dims));
  DataType type = op.GetInputDesc("values").GetDataType();
  y_value_desc.SetDataType(type);
  if (op.UpdateOutputDesc("y_values", y_value_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("update output[y_values] desc failed."));
    return GRAPH_FAILED;
  }
  TensorDesc y_shape_desc = op.GetOutputDesc("y_shape");
  y_shape_desc.SetShape(shape_shape);
  y_shape_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("y_shape", y_shape_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("update output[y_shape] desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseSlice, SparseSliceInfer);

IMPLEMT_INFERFUNC(SparseConcat, SparseConcatInfer) {
  int64_t n;
  std::string err_msg;
  if (op.GetAttr("N", n) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("get attr[N] value failed."));
    return GRAPH_FAILED;
  }
  int64_t output_row_count = 0;
  int64_t output_indice_cols = -1;
  Shape output_shape(ge::UNKNOWN_SHAPE);

  if (n < 2) {
    err_msg = GetAttrValueErrMsg("N", ConcatString(n), "must be large than 1");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int length_nodes = 0;
  while (true) {
    TensorDesc unused_desc = op.GetInputDesc(length_nodes);
    if (unused_desc.GetShape().GetDims().size() == 0) {
      break;
    } else {
      length_nodes += 1;
    }
  }

  vector<int> list_length(3);
  vector<string> list_input_names({"indices", "values", "shapes"});
  for (int i = 0; i < 3; i++) {
    while (true) {
      TensorDesc unused_desc =
          op.GetDynamicInputDesc(list_input_names[i], list_length[i]);
      if (unused_desc.GetShape().GetDims().size() == 0) {
        break;
      } else {
        list_length[i] += 1;
      }
    }
    if (list_length[i] != n) {
      err_msg =
          ConcatString("length of ", list_input_names[i], "(", list_length[i],
                       ")", " should be equal to N(", n, ")");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }

  for (int i = 0; i < n; i++) {
    // Get parameters
    TensorDesc ind = op.GetInputDesc(i);
    TensorDesc val = op.GetInputDesc(i + n);
    TensorDesc shape = op.GetInputDesc(i + 2 * n);

    // Check shape
    Shape unused_shape;
    if (WithRank(ind, 2, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
      err_msg = ConcatString(
          "failed to call WithRank function, input[indices] rank must be 2-D, "
          "got rank[", ind.GetShape().GetDimNum(), "]");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
    if (WithRank(val, 1, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
      err_msg = ConcatString(
          "failed to call WithRank function, input[values] rank must be 1-D, "
          "got rank[", val.GetShape().GetDimNum(), "]");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
    if (WithRank(shape, 1, unused_shape, TbeGetName(op).c_str()) !=
        GRAPH_SUCCESS) {
      err_msg = ConcatString(
          "failed to call WithRank function, input[shapes] rank must be 1-D, "
          "got rank[", shape.GetShape().GetDimNum(), "]");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }

    // Add to output_ind_rows.
    auto ind_shape = ind.GetShape();
    auto val_shape = val.GetShape();
    auto shape_shape = shape.GetShape();
    int64_t numDim = 0;
    if (Merge(ind_shape.GetDim(0), val_shape.GetDim(0), numDim) !=
        GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(
          TbeGetName(op),
          string(
              "failed to call Merge function for dim of indices and values."));
      return GRAPH_FAILED;
    }
    if (Add(output_row_count, numDim, output_row_count) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(
          TbeGetName(op), string("failed to call Merge function to add dim."));
      return GRAPH_FAILED;
    }

    // Merge into output_ind_cols and output_shape.
    if (Merge(output_indice_cols, ind_shape.GetDim(1), output_indice_cols) !=
        GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(
          TbeGetName(op),
          string(
              "failed to call Merge function for dim of output and indice."));
      return GRAPH_FAILED;
    }
    if (Merge(output_shape, shape_shape, output_shape, TbeGetName(op).c_str()) !=
        GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(
          TbeGetName(op),
          string(
              "failed to call Merge function for shape of output and shape."));
      return GRAPH_FAILED;
    }
  }

  // Set output
  TensorDesc y_indices_desc = op.GetOutputDesc("y_indices");
  Shape y_indices_shape({output_row_count, output_indice_cols});
  y_indices_desc.SetDataType(DT_INT64);
  y_indices_desc.SetShape(y_indices_shape);
  if (op.UpdateOutputDesc("y_indices", y_indices_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("update output[y_indices] desc failed."));
    return GRAPH_FAILED;
  }

  TensorDesc y_values_desc = op.GetOutputDesc("y_values");
  Shape y_values_shape({output_row_count});
  DataType values_type = op.GetDynamicInputDesc("values", 0).GetDataType();
  y_values_desc.SetDataType(values_type);
  y_values_desc.SetShape(y_values_shape);
  if (op.UpdateOutputDesc("y_values", y_values_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("update output[y_values] desc failed."));
    return GRAPH_FAILED;
  }

  TensorDesc y_shape_desc = op.GetOutputDesc("y_shape");
  y_shape_desc.SetDataType(DT_INT64);
  y_shape_desc.SetShape(output_shape);
  if (op.UpdateOutputDesc("y_shape", y_shape_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("update output[y_shape] desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseConcat, SparseConcatInfer);

IMPLEMT_INFERFUNC(SparseCross, SparseCrossInfer) {
  DataType out_type;
  if (op.GetAttr("out_type", out_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("fail to get attr[out_type]."));
    return GRAPH_FAILED;
  }
  std::string err_msg;
  if ((out_type != DT_INT64) && (out_type != DT_STRING)) {
    err_msg = ConcatString(
        "attr[out_type] data type should be DT_INT64 or DT_STRING, got[",
        DTypeStr(out_type), "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_PARAM_INVALID;
  }

  DataType internal_type;
  if (op.GetAttr("internal_type", internal_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("fail to get attr[internal_type]."));
    return GRAPH_FAILED;
  }
  if ((internal_type != DT_INT64) && (internal_type != DT_STRING)) {
    err_msg = ConcatString(
        "attr[internal_type] data type should be DT_INT64 or DT_STRING, got[",
        DTypeStr(internal_type), "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_PARAM_INVALID;
  }

  int indices_num = op.GetDynamicInputNum("indices");
  int start_num = 0;
  int end_num = indices_num;
  for (int i = start_num; i < end_num; i++) {
    Shape input_indices_shape;
    auto indice_desc = op.GetInputDesc(i);
    if (WithRank(indice_desc, 2, input_indices_shape, TbeGetName(op).c_str()) !=
        GRAPH_SUCCESS) {
      err_msg = GetShapeErrMsg(
          i, DebugString(indice_desc.GetShape().GetDims()), "2D");
      err_msg = string("failed to call WithRank function, ") + err_msg;
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }

  int values_num = op.GetDynamicInputNum("values");
  start_num = end_num;
  end_num = start_num + values_num;
  for (int i = start_num; i < end_num; i++) {
    Shape input_values_shape;
    auto value_desc = op.GetInputDesc(i);
    if (WithRank(value_desc, 1, input_values_shape, TbeGetName(op).c_str()) !=
        GRAPH_SUCCESS) {
      err_msg = GetShapeErrMsg(i, DebugString(value_desc.GetShape().GetDims()),
                               "1D");
      err_msg = string("failed to call WithRank function, ") + err_msg;
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }

  int shapes_num = op.GetDynamicInputNum("shapes");
  start_num = end_num;
  end_num = start_num + shapes_num;
  for (int i = start_num; i < end_num; i++) {
    Shape input_shapes_shape;
    auto shape_desc = op.GetInputDesc(i);
    if (WithRank(shape_desc, 1, input_shapes_shape, TbeGetName(op).c_str()) !=
        GRAPH_SUCCESS) {
      err_msg = GetShapeErrMsg(i, DebugString(shape_desc.GetShape().GetDims()),
                               "1D");
      err_msg = string("failed to call WithRank function, ") + err_msg;
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }

  int denses_num = op.GetDynamicInputNum("dense_inputs");
  start_num = end_num;
  end_num = start_num + denses_num;
  for (int i = start_num; i < end_num; i++) {
    Shape input_denses_shape;
    auto dense_desc = op.GetInputDesc(i);
    if (WithRank(dense_desc, 2, input_denses_shape, TbeGetName(op).c_str()) !=
        GRAPH_SUCCESS) {
      err_msg = GetShapeErrMsg(i, DebugString(dense_desc.GetShape().GetDims()),
                               "2D");
      err_msg = string("failed to call WithRank function, ") + err_msg;
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }

  Shape indices_shape;
  (void)Matrix(ge::UNKNOWN_DIM, 2, indices_shape);
  TensorDesc indices_desc = op.GetOutputDesc("output_indices");
  indices_desc.SetShape(indices_shape);
  indices_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("output_indices", indices_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("fail to update output[output_indices] desc."));
    return GRAPH_FAILED;
  }

  Shape values_shape;
  (void)Vector(ge::UNKNOWN_DIM, values_shape);
  TensorDesc values_desc = op.GetOutputDesc("output_values");
  values_desc.SetShape(values_shape);
  values_desc.SetDataType(out_type);

  if (op.UpdateOutputDesc("output_values", values_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("fail to update output[output_values] desc."));
    return GRAPH_FAILED;
  }

  Shape shape;
  (void)Vector(2, shape);
  TensorDesc shape_desc = op.GetOutputDesc("output_shape");
  shape_desc.SetShape(shape);
  shape_desc.SetDataType(DT_INT64);

  if (op.UpdateOutputDesc("output_shape", shape_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("fail to update output[output_shape] desc."));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseCross, SparseCrossInfer);

IMPLEMT_INFERFUNC(SparseToDense, SparseToDenseInfer) {
  DYNAMIC_SHAPE_NOT_SUPPORTED(op);
  GeShape shape;
  if (MakeShapeFromShapeTensor(op, "output_shape", shape,
                               TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call MakeShapeFromShapeTensor function to make shape from "
        "input[output_shape]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  DataType values_type = op.GetInputDesc("values").GetDataType();

  auto output_desc = op_desc->MutableOutputDesc(0);
  output_desc->SetShape(shape);
  output_desc->SetDataType(values_type);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseToDense, SparseToDenseInfer);

IMPLEMT_INFERFUNC(SparseAdd, SparseAddInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  GeShape x1_shape;
  if (WithRank(op_desc->MutableInputDesc(2), 1, x1_shape,
               TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        2, DebugString(op_desc->MutableInputDesc(2)->GetShape().GetDims()),
        "1D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  DataType x1_indices_type = DT_INT64;
  DataType x1_values_type = op_desc->MutableInputDesc(1)->GetDataType();
  DataType x1_shape_type = DT_INT64;

  auto sum_indices_desc = op_desc->MutableOutputDesc(0);
  std::vector<int64_t> new_dims;
  new_dims.push_back(ge::UNKNOWN_DIM);
  new_dims.push_back(x1_shape.GetDim(0));
  sum_indices_desc->SetShape(GeShape(new_dims));
  sum_indices_desc->SetDataType(x1_indices_type);

  auto sum_value_desc = op_desc->MutableOutputDesc(1);
  new_dims.clear();
  new_dims.push_back(ge::UNKNOWN_DIM);
  sum_value_desc->SetShape(GeShape(new_dims));
  sum_value_desc->SetDataType(x1_values_type);

  auto sum_shape_desc = op_desc->MutableOutputDesc(2);
  sum_shape_desc->SetShape(x1_shape);
  sum_shape_desc->SetDataType(x1_shape_type);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseAdd, SparseAddInfer);

IMPLEMT_INFERFUNC(SparseFillEmptyRows, SparseFillEmptyRowsInfer) {
  std::vector<std::string> input_infer_depends = {"dense_shape"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(input_infer_depends);

  GeShape indices_shape;
  const char* op_name = TbeGetName(op).c_str();
  auto indices_desc = op_desc->MutableInputDesc(0);
  std::string err_msg;
  if (WithRank(indices_desc, 2, indices_shape, op_name) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(0, DebugString(indices_desc->GetShape().GetDims()),
                             "2D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  GeShape values_shape;
  auto values_desc = op_desc->MutableInputDesc(1);
  if (WithRank(values_desc, 1, values_shape, op_name) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(1, DebugString(values_desc->GetShape().GetDims()),
                             "1D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  GeShape dense_shape;
  auto dense_desc = op_desc->MutableInputDesc(2);
  if (WithRank(dense_desc, 1, dense_shape, op_name) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(2, DebugString(dense_desc->GetShape().GetDims()),
                             "1D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  GeShape default_value_shape;
  auto default_value_desc = op_desc->MutableInputDesc(3);
  if (WithRank(default_value_desc, 0, default_value_shape, op_name) !=
      GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(3, DebugString(dense_desc->GetShape().GetDims()),
                             "scalar");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t num_dim = 0;
  if (indices_shape.GetDim(0) == UNKNOWN_DIM) {
    num_dim = UNKNOWN_DIM;
  } else {
    if (Merge(indices_shape.GetDim(0), values_shape.GetDim(0), num_dim) !=
        GRAPH_SUCCESS) {
      err_msg = ConcatString(
          "failed to call Merge function to merge input[indices] dim[0] and "
          "input[value] dim[0]. ",
          indices_shape.GetDim(0), " and ", values_shape.GetDim(0));
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }

  int64_t unuse_dim = 0;
  if (Merge(indices_shape.GetDim(1), dense_shape.GetDim(0), unuse_dim) !=
      GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call Merge function to merge input[indices] dim[0] and "
        "input[dense_shape] dim[0].");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  DataType indices_type = DT_INT64;
  DataType values_type = op.GetInputDesc("values").GetDataType();
  DataType empty_row_indicator_type = DT_BOOL;
  DataType reverse_index_map_type = DT_INT64;

  auto y_indices_desc = op_desc->MutableOutputDesc(0);
  std::vector<int64_t> new_dims;
  new_dims.push_back(ge::UNKNOWN_DIM);
  new_dims.push_back(dense_shape.GetDim(0));
  GeShape y_indices_shape(new_dims);
  if (!ShapeFullyDefined(y_indices_shape)) {
    std::vector<std::pair<int64_t, int64_t>> indices_range;
    for (const int64_t& indices_dim : y_indices_shape.GetDims()) {
      indices_range.push_back(
          indices_dim == UNKNOWN_DIM
              ? std::pair<int64_t, int64_t>{1, -1}
              : std::pair<int64_t, int64_t>{indices_dim, indices_dim});
    }
    y_indices_desc->SetShapeRange(indices_range);
  }
  y_indices_desc->SetShape(y_indices_shape);
  y_indices_desc->SetDataType(indices_type);

  new_dims.clear();
  new_dims.push_back(ge::UNKNOWN_DIM);
  GeShape y_value_shape(new_dims);
  if (!ShapeFullyDefined(y_value_shape)) {
    std::vector<std::pair<int64_t, int64_t>> value_range;
    for (const int64_t& value_dim : y_value_shape.GetDims()) {
      value_range.push_back(
          value_dim == UNKNOWN_DIM
              ? std::pair<int64_t, int64_t>{1, -1}
              : std::pair<int64_t, int64_t>{value_dim, value_dim});
    }
    y_indices_desc->SetShapeRange(value_range);
  }
  auto y_value_desc = op_desc->MutableOutputDesc(1);
  y_value_desc->SetShape(y_value_shape);
  y_value_desc->SetDataType(values_type);

  Shape const_dense_shape;
  Tensor dense_shape_tensor;
  if (op.GetInputConstData("dense_shape", dense_shape_tensor) ==
      GRAPH_SUCCESS) {
    if (MakeShapeFromShapeTensor(dense_shape_tensor, const_dense_shape,
                                 TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
      err_msg = ConcatString(
          "failed to call MakeShapeFromShapeTensor function to make shape from "
          "input[dense_shape]");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGI(TbeGetName(op).c_str(),
            "Get dense_shape from const data failed, using unknown shape");
    std::vector<int64_t> const_dense_dims{ge::UNKNOWN_DIM};
    const_dense_shape = Shape(const_dense_dims);
  }

  new_dims.clear();
  new_dims.push_back(const_dense_shape.GetDim(0));
  GeShape indicator_shape(new_dims);
  if (!ShapeFullyDefined(indicator_shape)) {
    std::vector<std::pair<int64_t, int64_t>> indicator_range;
    for (const int64_t& indicator_dim : indicator_shape.GetDims()) {
      indicator_range.push_back(
          indicator_dim == UNKNOWN_DIM
              ? std::pair<int64_t, int64_t>{1, -1}
              : std::pair<int64_t, int64_t>{indicator_dim, indicator_dim});
    }
    y_indices_desc->SetShapeRange(indicator_range);
  }
  auto empty_row_indicator_desc = op_desc->MutableOutputDesc(2);
  empty_row_indicator_desc->SetShape(indicator_shape);
  empty_row_indicator_desc->SetDataType(empty_row_indicator_type);

  new_dims.clear();
  new_dims.push_back(num_dim);
  auto reverse_index_map_desc = op_desc->MutableOutputDesc(3);
  reverse_index_map_desc->SetShape(GeShape(new_dims));
  reverse_index_map_desc->SetDataType(reverse_index_map_type);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseFillEmptyRows, SparseFillEmptyRowsInfer);

IMPLEMT_INFERFUNC(SparseSparseMaximum, SparseSparseMaximumInfer) {
  Shape unuse_shape;
  std::string err_msg;
  if (WithRank(op.GetInputDesc(0), 2, unuse_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[x1_indices] rank must be 2-D, "
        "got rank[", op.GetInputDesc(0).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 1, unuse_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[x1_values] rank must be 1-D, "
        "got rank[", op.GetInputDesc(1).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(2), 1, unuse_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[x1_shape] rank must be 1-D, "
        "got rank[", op.GetInputDesc(2).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(3), 2, unuse_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[x2_indices] rank must be 2-D, "
        "got rank[", op.GetInputDesc(3).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(4), 1, unuse_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[x2_values] rank must be 1-D, "
        "got rank[", op.GetInputDesc(4).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(5), 1, unuse_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[x2_shape] rank must be 1-D, "
        "got rank[", op.GetInputDesc(5).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  DataType x1_indices_type = DT_INT64;
  DataType x1_values_type = op.GetInputDesc("x1_values").GetDataType();

  TensorDesc y_indices_desc = op.GetOutputDesc("y_indices");
  std::vector<int64_t> new_dims;
  new_dims.push_back(ge::UNKNOWN_DIM);
  new_dims.push_back(ge::UNKNOWN_DIM);
  y_indices_desc.SetShape(Shape(new_dims));
  y_indices_desc.SetDataType(x1_indices_type);
  if (op.UpdateOutputDesc("y_indices", y_indices_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("update output[y_indices] desc failed"));
    return GRAPH_FAILED;
  }

  TensorDesc y_values_desc = op.GetOutputDesc("y_values");
  new_dims.clear();
  new_dims.push_back(ge::UNKNOWN_DIM);
  y_values_desc.SetShape(Shape(new_dims));
  y_values_desc.SetDataType(x1_values_type);
  if (op.UpdateOutputDesc("y_values", y_values_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("update output[y_values] desc failed"));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseSparseMaximum, SparseSparseMaximumInfer);

IMPLEMT_INFERFUNC(SparseSparseMinimum, SparseSparseMinimumInfer) {
  Shape unuse_shape;
  std::string err_msg;
  if (WithRank(op.GetInputDesc(0), 2, unuse_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[x1_indices] rank must be 2-D, "
        "got rank[", op.GetInputDesc(0).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 1, unuse_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[x1_values] rank must be 1-D, "
        "got rank[", op.GetInputDesc(1).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(2), 1, unuse_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[x1_shape] rank must be 1-D, "
        "got rank[", op.GetInputDesc(2).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(3), 2, unuse_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[x2_indices] rank must be 2-D, "
        "got rank[", op.GetInputDesc(3).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(4), 1, unuse_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[x2_values] rank must be 1-D, "
        "got rank[", op.GetInputDesc(4).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(5), 1, unuse_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[x2_shape] rank must be 1-D, "
        "got rank[", op.GetInputDesc(5).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  DataType x1_indices_type = DT_INT64;
  DataType x1_values_type = op.GetInputDesc("x1_values").GetDataType();

  TensorDesc y_indices_desc = op.GetOutputDesc("y_indices");
  std::vector<int64_t> newDims;
  newDims.push_back(ge::UNKNOWN_DIM);
  newDims.push_back(ge::UNKNOWN_DIM);
  y_indices_desc.SetShape(Shape(newDims));
  y_indices_desc.SetDataType(x1_indices_type);
  if (op.UpdateOutputDesc("y_indices", y_indices_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("update output[y_indices] desc failed"));
    return GRAPH_FAILED;
  }

  TensorDesc y_values_desc = op.GetOutputDesc("y_values");
  newDims.clear();
  newDims.push_back(ge::UNKNOWN_DIM);
  y_values_desc.SetShape(Shape(newDims));
  y_values_desc.SetDataType(x1_values_type);
  if (op.UpdateOutputDesc("y_values", y_values_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("update output[y_values] desc failed"));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseSparseMinimum, SparseSparseMinimumInfer);

IMPLEMT_INFERFUNC(SparseReduceMax, SparseReduceMaxInfer) {
  bool keep_dim = false;
  if (op.GetAttr("keep_dims", keep_dim) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("get attr[keep_dims] failed"));
    return GRAPH_FAILED;
  }
  DataType x_values_type = op.GetInputDesc("x_values").GetDataType();
  Tensor x_shape_tensor;
  Tensor reduction_axes_tensor;
  auto result_x = op.GetInputConstData("x_shape", x_shape_tensor);
  auto result_reduction = op.GetInputConstData("reduction_axes", reduction_axes_tensor);
  if (result_x != GRAPH_SUCCESS || result_reduction != GRAPH_SUCCESS) {
    Shape unknown_shape(ge::UNKNOWN_RANK);
    TensorDesc y_desc = op.GetOutputDesc("y");
    y_desc.SetShape(unknown_shape);
    y_desc.SetDataType(x_values_type);
    op.UpdateOutputDesc("y", y_desc);
  } else {
    const int64_t* x_shape_data = reinterpret_cast<const int64_t *>(x_shape_tensor.GetData());
    const int32_t* reduction_axes_data = reinterpret_cast<const int32_t *>(reduction_axes_tensor.GetData());
    int64_t ndims = x_shape_tensor.GetTensorDesc().GetShape().GetShapeSize();
    int32_t count = reduction_axes_tensor.GetTensorDesc().GetShape().GetShapeSize();
    std::unordered_set<int64_t> axes;
    for (int32_t i = 0; i < count; ++i) {
      axes.insert((static_cast<int64_t>(reduction_axes_data[i]) + ndims) % ndims);
    }
    std::vector<int64_t> dims;
    if (keep_dim) {
      dims.reserve(ndims);
      for (int d = 0; d < ndims; ++d) {
        if (axes.find(d) == axes.end()) {
          dims.push_back(x_shape_data[d]);
        } else {
          dims.push_back(1);
        }
      }
    } else {
      for (int d = 0; d < ndims; ++d) {
        if (axes.find(d) == axes.end()) {
          dims.push_back(x_shape_data[d]);
        }
      }
    }
    TensorDesc y_desc = op.GetOutputDesc("y");
    ge::Shape shape(dims);
    y_desc.SetShape(shape);
    y_desc.SetDataType(x_values_type);
    op.UpdateOutputDesc("y", y_desc);
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseReduceMax, SparseReduceMaxInfer);

IMPLEMT_INFERFUNC(SparseReduceMaxSparse, SparseReduceMaxSparseInfer) {
  Shape unknown_shape(ge::UNKNOWN_SHAPE);

  DataType y_indices_type = DT_INT64;
  DataType y_values_type = op.GetInputDesc("x_values").GetDataType();
  DataType y_shape_type = DT_INT64;

  TensorDesc y_indices_desc = op.GetOutputDesc("y_indices");
  y_indices_desc.SetShape(unknown_shape);
  y_indices_desc.SetDataType(y_indices_type);
  op.UpdateOutputDesc("y_indices", y_indices_desc);

  TensorDesc y_values_desc = op.GetOutputDesc("y_values");
  y_values_desc.SetShape(unknown_shape);
  y_values_desc.SetDataType(y_values_type);
  op.UpdateOutputDesc("y_values", y_values_desc);

  TensorDesc y_shape_desc = op.GetOutputDesc("y_shape");
  y_shape_desc.SetShape(unknown_shape);
  y_shape_desc.SetDataType(y_shape_type);
  op.UpdateOutputDesc("y_shape", y_shape_desc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseReduceMaxSparse, SparseReduceMaxSparseInfer);

IMPLEMT_INFERFUNC(SparseReduceSum, SparseReduceSumInfer) {
  GeShape unknown_shape(ge::UNKNOWN_SHAPE);
  DataType y_type = op.GetInputDesc("x_values").GetDataType();

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto y_desc = op_desc->MutableOutputDesc(0);
  y_desc->SetShape(unknown_shape);
  y_desc->SetDataType(y_type);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseReduceSum, SparseReduceSumInfer);

IMPLEMT_INFERFUNC(SparseReduceSumSparse, SparseReduceSumSparseInfer) {
  Shape unknown_shape(ge::UNKNOWN_SHAPE);

  DataType y_indices_type = DT_INT64;
  DataType y_values_type = op.GetInputDesc("x_values").GetDataType();
  DataType y_shape_type = DT_INT64;

  TensorDesc y_indices_desc = op.GetOutputDesc("y_indices");
  y_indices_desc.SetShape(unknown_shape);
  y_indices_desc.SetDataType(y_indices_type);
  op.UpdateOutputDesc("y_indices", y_indices_desc);

  TensorDesc y_values_desc = op.GetOutputDesc("y_values");
  y_values_desc.SetShape(unknown_shape);
  y_values_desc.SetDataType(y_values_type);
  op.UpdateOutputDesc("y_values", y_values_desc);

  TensorDesc y_shape_desc = op.GetOutputDesc("y_shape");
  y_shape_desc.SetShape(unknown_shape);
  y_shape_desc.SetDataType(y_shape_type);
  op.UpdateOutputDesc("y_shape", y_shape_desc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseReduceSumSparse, SparseReduceSumSparseInfer);

IMPLEMT_INFERFUNC(SparseSplit, SparseSplitInfer) {
  auto shape_tensor = op.get_input_desc_shape();
  auto values_tensor = op.get_input_desc_values();
  auto shape_shape = shape_tensor.GetShape();
  Shape output_indices;
  Shape output_values;
  auto output_shape = shape_shape;
  Matrix(ge::UNKNOWN_DIM, shape_shape.GetShapeSize(), output_indices);
  Vector(ge::UNKNOWN_DIM, output_values);

  uint32_t num_splits = 0;
  if (op.GetAttr("num_split", num_splits) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("get attr[num_split] failed"));
    return GRAPH_FAILED;
  }
  for (uint32_t i = 0; i < num_splits; i++) {
    TensorDesc y_tensor = op.GetDynamicOutputDesc("y_indices", i);
    y_tensor.SetShape(output_indices);
    y_tensor.SetDataType(DT_INT64);
    if (op.UpdateDynamicOutputDesc("y_indices", i, y_tensor) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString(
          "update description for output y_indices[", i, "] failed");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }
  for (uint32_t i = 0; i < num_splits; i++) {
    TensorDesc y_tensor = op.GetDynamicOutputDesc("y_values", i);
    y_tensor.SetShape(output_values);
    y_tensor.SetDataType(values_tensor.GetDataType());
    if (op.UpdateDynamicOutputDesc("y_values", i, y_tensor) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString("update description for output y_values[", i, "] failed");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }
  for (uint32_t i = 0; i < num_splits; i++) {
    TensorDesc y_tensor = op.GetDynamicOutputDesc("y_shape", i);
    y_tensor.SetShape(output_shape);
    y_tensor.SetDataType(DT_INT64);
    if (op.UpdateDynamicOutputDesc("y_shape", i, y_tensor) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString("update description for output y_shape[", i, "] failed");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseSplit, SparseSplitInfer);

IMPLEMT_INFERFUNC(AddManySparseToTensorsMap, AddManySparseToTensorsMapInfer) {
  Shape unused_shape;
  auto indices_tensor = op.get_input_desc_indices();
  auto values_tensor = op.get_input_desc_values();
  auto shape_tensor = op.get_input_desc_shape();
  if (WithRank(indices_tensor, 2, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(indices_tensor.GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_PARAM_INVALID;
  }
  if (WithRank(values_tensor, 1, unused_shape, TbeGetName(op).c_str())
      != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(values_tensor.GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_PARAM_INVALID;
  }
  if (WithRank(shape_tensor, 1, unused_shape, TbeGetName(op).c_str())
      != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2,
        DebugString(shape_tensor.GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_PARAM_INVALID;
  }
  auto output_shape = Shape({ge::UNKNOWN_DIM});
  TensorDesc y_desc = op.GetOutputDesc("handles");
  y_desc.SetShape(output_shape);
  y_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("handles", y_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
                                      string("update output[handles] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AddManySparseToTensorsMap, AddManySparseToTensorsMapInfer);

IMPLEMT_INFERFUNC(SparseDenseCwiseAdd, SparseDenseCwiseAddInfer) {
  Shape shape;
  auto x1_indices_tensor = op.get_input_desc_x1_indices();
  if (WithRank(x1_indices_tensor, 2, shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call WithRank function, input[x1_indices] rank must be 2-D, "
        "got rank[", x1_indices_tensor.GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_PARAM_INVALID;
  }

  auto x_dim = x1_indices_tensor.GetShape().GetDim(0);
  auto vector_shape = Shape({x_dim});
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(vector_shape);
  auto x1_values_tensor = op.get_input_desc_x1_values();
  y_desc.SetDataType(x1_values_tensor.GetDataType());
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("update output[y] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseDenseCwiseAdd, SparseDenseCwiseAddInfer);

IMPLEMT_INFERFUNC(SparseDenseCwiseDiv, SparseDenseCwiseDivInfer) {
  Shape shape;
  auto x1_indices_tensor = op.get_input_desc_x1_indices();
  if (WithRank(x1_indices_tensor, 2, shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call WithRank function, input[x1_indices] rank must be 2-D, "
        "got rank[", x1_indices_tensor.GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_PARAM_INVALID;
  }

  auto x_dim = x1_indices_tensor.GetShape().GetDim(0);
  auto vector_shape = Shape({x_dim});
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(vector_shape);
  auto x1_values_tensor = op.get_input_desc_x1_values();
  y_desc.SetDataType(x1_values_tensor.GetDataType());
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("update output[y] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseDenseCwiseDiv, SparseDenseCwiseDivInfer);

IMPLEMT_INFERFUNC(SparseDenseCwiseMul, SparseDenseCwiseMulInfer) {
  Shape shape;
  auto x1_indices_tensor = op.get_input_desc_x1_indices();
  if (WithRank(x1_indices_tensor, 2, shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call WithRank function, input[x1_indices] rank must be 2-D, "
        "got rank[", x1_indices_tensor.GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_PARAM_INVALID;
  }

  auto x_dim = x1_indices_tensor.GetShape().GetDim(0);
  auto vector_shape = Shape({x_dim});
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(vector_shape);
  auto x1_values_tensor = op.get_input_desc_x1_values();
  y_desc.SetDataType(x1_values_tensor.GetDataType());
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("update output[y] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseDenseCwiseMul, SparseDenseCwiseMulInfer);

IMPLEMT_INFERFUNC(SparseReorder, SparseReorderInfer) {
  auto tensor_indices = op.get_input_desc_indices();
  auto tensor_values = op.get_input_desc_values();
  auto tensor_shape = op.get_input_desc_shape();

  Shape indices;
  Shape values;
  Shape unused;
  std::string err_msg;
  if (WithRank(tensor_indices, 2, indices, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[indices] rank must be 2-D, "
        "got rank[",
        tensor_indices.GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(tensor_values, 1, values, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[values] rank must be 1-D, "
        "got rank[",
        tensor_values.GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(tensor_shape, 1, unused, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[shape] rank must be 1-D, "
        "got rank[",
        tensor_shape.GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  DataType y_indices_type = op.GetInputDesc("indices").GetDataType();
  DataType y_values_type = op.GetInputDesc("values").GetDataType();

  TensorDesc indices_desc = op.GetOutputDesc("y_indices");
  indices_desc.SetShape(Shape(indices));
  indices_desc.SetDataType(y_indices_type);
  if (op.UpdateOutputDesc("y_indices", indices_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("update output[y_indices] desc failed."));
    return GRAPH_FAILED;
  }

  TensorDesc values_desc = op.GetOutputDesc("y_values");
  values_desc.SetShape(Shape(values));
  values_desc.SetDataType(y_values_type);
  if (op.UpdateOutputDesc("y_values", values_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("update output[y_values] desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseReorder, SparseReorderInfer);

IMPLEMT_INFERFUNC(SparseReshape, SparseReshapeInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  auto indices_desc = op_desc->MutableInputDesc(0);
  auto shape_desc = op_desc->MutableInputDesc(1);
  auto new_shape_desc = op_desc->MutableInputDesc(2);
  std::string err_msg;
  if ((indices_desc->GetDataType() != DT_INT64) ||
      (shape_desc->GetDataType() != DT_INT64) ||
      (new_shape_desc->GetDataType() != DT_INT64)) {
    err_msg = ConcatString(
        "all input data type should be DT_INT64, input[indices] data type[",
        DTypeStr(indices_desc->GetDataType()), "], input[shape] data type[",
        DTypeStr(shape_desc->GetDataType()), "], input[new_shape] data type[",
        DTypeStr(new_shape_desc->GetDataType()), "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_PARAM_INVALID;
  }

  const char* op_name = TbeGetName(op).c_str();
  GeShape indices;
  if (WithRank(indices_desc, 2, indices, op_name) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(0, DebugString(indices_desc->GetShape().GetDims()),
                             "2D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  GeShape unused;
  if (WithRank(shape_desc, 1, unused, op_name) != GRAPH_SUCCESS) {
    err_msg =
        GetShapeErrMsg(1, DebugString(shape_desc->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  GeShape new_shape;
  if (WithRank(new_shape_desc, 1, new_shape, op_name) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(
        2, DebugString(new_shape_desc->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  Shape result;
  auto indices_shape_dims = indices_desc->GetShape().GetDims();
  auto new_shape_dims = new_shape_desc->GetShape().GetDims();

  Matrix(indices_shape_dims[0], new_shape_dims[0], result);
  GeShape y_indices_shape(result.GetDims());

  auto y_indices_desc = op_desc->MutableOutputDesc(0);
  y_indices_desc->SetShape(y_indices_shape);
  y_indices_desc->SetDataType(indices_desc->GetDataType());

  auto y_shape_desc = op_desc->MutableOutputDesc(1);
  y_shape_desc->SetShape(new_shape);
  y_shape_desc->SetDataType(shape_desc->GetDataType());

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseReshape, SparseReshapeInfer);

IMPLEMT_INFERFUNC(SparseAddGrad, SparseAddGradInfer) {
  Shape shape;
  std::string err_msg;
  if (WithRank(op.GetInputDesc(1), 2, shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[x1_indices] rank must be 2-D, "
        "got rank[",
        op.GetInputDesc(1).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(2), 2, shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[x2_indices] rank must be 2-D, "
        "got rank[",
        op.GetInputDesc(2).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  DataType backprop_val_grad_type;
  backprop_val_grad_type = op.GetInputDesc("backprop_val_grad").GetDataType();

  auto x1_indices_dims = op.GetInputDesc(1).GetShape().GetDims();
  auto x2_indices_dims = op.GetInputDesc(2).GetShape().GetDims();

  graphStatus output_status;

  TensorDesc x1_val_grad_desc = op.GetOutputDesc("x1_val_grad");
  x1_val_grad_desc.SetShape(Shape({x1_indices_dims[0]}));
  x1_val_grad_desc.SetDataType(backprop_val_grad_type);
  output_status = op.UpdateOutputDesc("x1_val_grad", x1_val_grad_desc);
  if (output_status != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("update output[x1_val_grad] desc failed"));
    return GRAPH_FAILED;
  }

  TensorDesc x2_val_grad_desc = op.GetOutputDesc("x2_val_grad");
  x2_val_grad_desc.SetShape(Shape({x2_indices_dims[0]}));
  x2_val_grad_desc.SetDataType(backprop_val_grad_type);
  output_status = op.UpdateOutputDesc("x2_val_grad", x2_val_grad_desc);
  if (output_status != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("update output[x2_val_grad] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseAddGrad, SparseAddGradInfer);

IMPLEMT_INFERFUNC(SparseFillEmptyRowsGrad, SparseFillEmptyRowsGradInfer) {
  Shape shape;
  std::string err_msg;
  if (WithRank(op.GetInputDesc(0), 1, shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(
        0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 1, shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(
        0, DebugString(op.GetInputDesc(1).GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  DataType grad_values_type = op.GetInputDesc("grad_values").GetDataType();
  auto reverse_index_map_shape = op.GetInputDesc(0).GetShape();

  graphStatus output_status;

  TensorDesc y_value_desc = op.GetOutputDesc("y_value");
  y_value_desc.SetShape(reverse_index_map_shape);
  y_value_desc.SetDataType(grad_values_type);
  output_status = op.UpdateOutputDesc("y_value", y_value_desc);
  if (output_status != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("fail to update output[y_value] desc."));
    return GRAPH_FAILED;
  }

  TensorDesc y_default_value_desc = op.GetOutputDesc("y_default_value");
  y_default_value_desc.SetShape(Shape());
  y_default_value_desc.SetDataType(grad_values_type);
  output_status = op.UpdateOutputDesc("y_default_value", y_default_value_desc);
  if (output_status != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("fail to update output[y_default_value] desc."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseFillEmptyRowsGrad, SparseFillEmptyRowsGradInfer);

IMPLEMT_INFERFUNC(SparseTensorDenseMatMul, SparseTensorDenseMatMulInfer) {
  int64_t unused_dim = 0;
  Shape unused_shape;
  GeShape x1_shape;
  Shape x2_shape;
  auto x1_indices_tensor = op.get_input_desc_x1_indices();
  auto x1_values_tensor = op.get_input_desc_x1_values();
  auto x1_shape_tensor = op.get_input_desc_x1_shape();
  auto x2_tensor = op.get_input_desc_x2();
  std::string err_msg;
  if (WithRank(x1_indices_tensor, 2, unused_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(
        0, DebugString(x1_indices_tensor.GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(x1_values_tensor, 1, unused_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(
        1, DebugString(x1_values_tensor.GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (MakeShapeFromShapeTensor(op, "x1_shape", x1_shape,
                               TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call MakeShapeFromShapeTensor function to make shape from "
        "input[x1_shape]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRankShape(x1_shape, 2, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(2, DebugString(x1_shape.GetDims()), "2D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(x2_tensor, 2, x2_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    err_msg =
        GetShapeErrMsg(3, DebugString(x2_tensor.GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  bool adjoint_a, adjoint_b;
  if (op.GetAttr("adjoint_a", adjoint_a) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("failed to get attr[adjoint_a]."));
    return GRAPH_FAILED;
  }
  if (op.GetAttr("adjoint_b", adjoint_b) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("failed to get attr[adjoint_b]."));
    return GRAPH_FAILED;
  }

  int64_t output_right_shape = x2_shape.GetDim(adjoint_b ? 0 : 1);
  int64_t output_left_shape = x1_shape.GetDim(adjoint_a ? 1 : 0);
  int64_t inner_left_shape = x1_shape.GetDim(adjoint_a ? 0 : 1);
  int64_t inner_right_shape = x2_shape.GetDim(adjoint_b ? 1 : 0);
  graphStatus status = Merge(inner_left_shape, inner_right_shape, unused_dim);
  if (status != GRAPH_SUCCESS) {
    err_msg = ConcatString("failed to call Merge function to merge ",
                           inner_left_shape, " and ", inner_right_shape);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  Shape output_shape;
  status = Matrix(output_left_shape, output_right_shape, output_shape);
  if (status != GRAPH_SUCCESS) {
    err_msg = ConcatString("failed to call Matrix function to create matrix ",
                           output_left_shape, " and ", output_right_shape);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc y_tensor = op.GetOutputDesc("y");
  y_tensor.SetDataType(x1_values_tensor.GetDataType());
  y_tensor.SetShape(output_shape);
  return op.UpdateOutputDesc("y", y_tensor);
}

INFER_FUNC_REG(SparseTensorDenseMatMul, SparseTensorDenseMatMulInfer);

IMPLEMT_INFERFUNC(SerializeSparse, SerializeSparseInfer) {
  Shape unused_shape;
  auto indices_tensor = op.get_input_desc_indices();
  auto values_tensor = op.get_input_desc_values();
  auto shape_tensor = op.get_input_desc_shape();
  if (WithRank(indices_tensor, 2, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(indices_tensor.GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(values_tensor, 1, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(values_tensor.GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(shape_tensor, 1, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2,
        DebugString(shape_tensor.GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  DataType output_type;
  if (op.GetAttr("out_type", output_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("get attr[out_type] failed."));
    return GRAPH_FAILED;
  }
  Shape output_shape;
  (void)Vector(3, output_shape);
  TensorDesc output_tensor = op.GetOutputDesc("serialized_sparse");
  output_tensor.SetDataType(output_type);
  output_tensor.SetShape(output_shape);
  return op.UpdateOutputDesc("serialized_sparse", output_tensor);
}

INFER_FUNC_REG(SerializeSparse, SerializeSparseInfer);

IMPLEMT_INFERFUNC(SerializeManySparse, SerializeManySparseInfer) {
  Shape unused_shape;
  auto indices_tensor = op.get_input_desc_indices();
  auto values_tensor = op.get_input_desc_values();
  auto shape_tensor = op.get_input_desc_shape();
  if (WithRank(indices_tensor, 2, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(indices_tensor.GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(values_tensor, 1, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(values_tensor.GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(shape_tensor, 1, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2,
        DebugString(shape_tensor.GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  DataType output_type;
  if (op.GetAttr("out_type", output_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("get attr[out_type] failed."));
    return GRAPH_FAILED;
  }
  Shape output_shape;
  (void)Matrix(ge::UNKNOWN_DIM, 3, output_shape);
  TensorDesc output_tensor = op.GetOutputDesc("serialized_sparse");
  output_tensor.SetDataType(output_type);
  output_tensor.SetShape(output_shape);
  return op.UpdateOutputDesc("serialized_sparse", output_tensor);
}

INFER_FUNC_REG(SerializeManySparse, SerializeManySparseInfer);

IMPLEMT_INFERFUNC(DeserializeSparse, DeserializeSparseInfer) {
  Shape input_shape = op.get_input_desc_serialized_sparse().GetShape();
  int64_t existing = input_shape.GetDimNum();
  if (input_shape.GetDim(existing - 1) != 3) {
    std::string err_msg = ConcatString("input[serialized_sparse] last dim value[",
        input_shape.GetDim(existing - 1), "] must be 3");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  Shape indices_shape;
  Shape values_shape;
  Shape sparse_shape;
  (void)Matrix(ge::UNKNOWN_DIM, ge::UNKNOWN_DIM, indices_shape);
  (void)Vector(ge::UNKNOWN_DIM, values_shape);
  (void)Vector(ge::UNKNOWN_DIM, sparse_shape);

  TensorDesc indices_tensor = op.GetOutputDesc("indices");
  indices_tensor.SetDataType(DT_INT64);
  indices_tensor.SetShape(indices_shape);
  if (op.UpdateOutputDesc("indices", indices_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("update output[indices] desc failed."));
    return GRAPH_FAILED;
  }

  DataType values_type;
  if (op.GetAttr("dtype", values_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("get attr[dtype] failed."));
    return GRAPH_FAILED;
  }
  TensorDesc values_tensor = op.GetOutputDesc("values");
  values_tensor.SetDataType(values_type);
  values_tensor.SetShape(values_shape);
  if (op.UpdateOutputDesc("values", values_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("update output[values] desc failed."));
    return GRAPH_FAILED;
  }

  TensorDesc shape_tensor = op.GetOutputDesc("shape");
  shape_tensor.SetDataType(DT_INT64);
  shape_tensor.SetShape(sparse_shape);
  return op.UpdateOutputDesc("shape", shape_tensor);
}

INFER_FUNC_REG(DeserializeSparse, DeserializeSparseInfer);

IMPLEMT_INFERFUNC(DeserializeManySparse, DeserializeManySparseInfer) {
  Shape input_shape;
  auto serialized_sparse_tensor = op.get_input_desc_serialized_sparse();
  if (WithRank(serialized_sparse_tensor, 2, input_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(serialized_sparse_tensor.GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  int64_t existing = input_shape.GetDimNum();
  if (input_shape.GetDim(existing - 1) != 3) {
    OP_LOGE(TbeGetName(op).c_str(), "Input last dim value must be 3");
    return GRAPH_FAILED;
  }

  Shape indices_shape;
  Shape values_shape;
  Shape sparse_shape;
  (void)Matrix(ge::UNKNOWN_DIM, ge::UNKNOWN_DIM, indices_shape);
  (void)Vector(ge::UNKNOWN_DIM, values_shape);
  (void)Vector(ge::UNKNOWN_DIM, sparse_shape);

  TensorDesc indices_tensor = op.GetOutputDesc("indices");
  indices_tensor.SetDataType(DT_INT64);
  indices_tensor.SetShape(indices_shape);
  if (op.UpdateOutputDesc("indices", indices_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("update output[indices] desc failed."));
    return GRAPH_FAILED;
  }

  DataType values_type;
  if (op.GetAttr("dtype", values_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("get attr[dtype] failed."));
    return GRAPH_FAILED;
  }
  TensorDesc values_tensor = op.GetOutputDesc("values");
  values_tensor.SetDataType(values_type);
  values_tensor.SetShape(values_shape);
  if (op.UpdateOutputDesc("values", values_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("update output[values] desc failed."));
    return GRAPH_FAILED;
  }

  TensorDesc shape_tensor = op.GetOutputDesc("shape");
  shape_tensor.SetDataType(DT_INT64);
  shape_tensor.SetShape(sparse_shape);
  return op.UpdateOutputDesc("shape", shape_tensor);
}

INFER_FUNC_REG(DeserializeManySparse, DeserializeManySparseInfer);

IMPLEMT_INFERFUNC(TakeManySparseFromTensorsMap,
                  TakeManySparseFromTensorsMapInfer) {
  Shape unused_shape;
  std::string err_msg;
  if (WithRank(op.GetInputDesc(0), 1, unused_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(
        0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  Shape indices_shape;
  Shape values_shape;
  Shape sparse_shape;
  (void)Matrix(ge::UNKNOWN_DIM, ge::UNKNOWN_DIM, indices_shape);
  (void)Vector(ge::UNKNOWN_DIM, values_shape);
  (void)Vector(ge::UNKNOWN_DIM, sparse_shape);

  TensorDesc indices_tensor = op.GetOutputDesc("indices");
  indices_tensor.SetDataType(DT_INT64);
  indices_tensor.SetShape(indices_shape);
  if (op.UpdateOutputDesc("indices", indices_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("fail to update output[indices] desc."));
    return GRAPH_FAILED;
  }

  DataType values_type;
  if (op.GetAttr("dtype", values_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("fail to get attr[dtype]."));
    return GRAPH_FAILED;
  }
  TensorDesc values_tensor = op.GetOutputDesc("values");
  values_tensor.SetDataType(values_type);
  values_tensor.SetShape(values_shape);
  if (op.UpdateOutputDesc("values", values_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("fail to update output[values] desc."));
    return GRAPH_FAILED;
  }

  TensorDesc shape_tensor = op.GetOutputDesc("shape");
  shape_tensor.SetDataType(DT_INT64);
  shape_tensor.SetShape(sparse_shape);
  return op.UpdateOutputDesc("shape", shape_tensor);
}

INFER_FUNC_REG(TakeManySparseFromTensorsMap, TakeManySparseFromTensorsMapInfer);

}  // namespace ge
