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

namespace ge {
IMPLEMT_INFERFUNC(DenseToDenseSetOperation, DenseToDenseSetOperationInfer) {
  if (op.GetInputsSize() != 2) {
    OP_LOGE(op.GetName().c_str(), "Inputs size is not 2.");
    return GRAPH_FAILED;
  }

  auto x1_tensor = op.GetInputDesc(0);
  auto x2_tensor = op.GetInputDesc(1);
  Shape x1_shape = x1_tensor.GetShape();
  int64_t output_rank_dim = 0;
  if (WithRankAtLeast(x1_tensor, 2, x1_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Input x1 rank at least must be 2.");
    return GRAPH_FAILED;
  }
  if (RankKnown(x1_shape)) {
    const int64_t x1_rank = static_cast<int64_t>(x1_shape.GetDimNum());
    Shape x2_shape;
    if (WithRank(x2_tensor, x1_rank, x2_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "Input x2 rank must match x1.");
      return GRAPH_FAILED;
    }
    if (RankKnown(x2_shape)) {
      const int64_t x2_rank = static_cast<int64_t>(x2_shape.GetDimNum());
      Shape group_x1_shape;
      graphStatus status = SubShape(x1_shape, 0, x2_rank - 1, 1, group_x1_shape, op.GetName().c_str());
      if (status != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Input x1 subShape error.");
        return GRAPH_FAILED;
      }
      Shape group_x2_shape;
      status = SubShape(x2_shape, 0, x2_rank - 1, 1, group_x2_shape, op.GetName().c_str());
      if (status != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Input x2 subShape error.");
        return GRAPH_FAILED;
      }
      Shape unused_shape;
      status = Merge(group_x1_shape, group_x1_shape, unused_shape, op.GetName().c_str());
      if (status != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Input x1 x2 merge shape error.");
        return GRAPH_FAILED;
      }
    }
    output_rank_dim = x1_rank;
  } else {
    Shape x2_shape = x2_tensor.GetShape();
    if (WithRankAtLeast(x2_tensor, 2, x2_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "Input x2 rank at least must be 2.");
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
    OP_LOGE(op.GetName().c_str(), "Input x1 type must match x2 type.");
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
    OP_LOGE(op.GetName().c_str(), "Updata output y_indices error.");
    return GRAPH_FAILED;
  }
  TensorDesc y_values_tensor = op.GetOutputDesc("y_values");
  y_values_tensor.SetShape(output_values);
  y_values_tensor.SetDataType(output_type);
  if (op.UpdateOutputDesc("y_values", y_values_tensor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Updata output y_values error.");
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
  if (op.GetInputsSize() != 4) {
    OP_LOGE(op.GetName().c_str(), "Inputs size is not 4.");
    return GRAPH_FAILED;
  }

  auto x2_indices_tensor = op.GetInputDesc(1);
  auto x2_values_tensor = op.GetInputDesc(2);
  auto x2_shape_tensor = op.GetInputDesc(3);
  graphStatus status = ValidateSparseTensor(x2_indices_tensor, x2_values_tensor, x2_shape_tensor, op.GetName().c_str());
  if (status != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Validate sparse parameters error.");
    return GRAPH_FAILED;
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x1_desc = op_desc->MutableInputDesc(0);
  GeShape x1_shape;
  Shape x2_shape = x2_shape_tensor.GetShape();
  int64_t output_rank_dim = 0;
  const int64_t input_rank_dim = x2_shape.GetDim(0);
  if (WithRankAtLeast(x1_desc, 2, x1_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Input x1 rank at least must be 2.");
    return GRAPH_FAILED;
  }
  if (RankKnown(x1_shape)) {
    int64_t x1_rank = static_cast<int64_t>(x1_shape.GetDimNum());
    if (input_rank_dim != x1_rank) {
      OP_LOGE(op.GetName().c_str(), "Input x1 rank must match x2 dim0.");
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
    OP_LOGE(op.GetName().c_str(), "Updata output y_indices error.");
    return GRAPH_FAILED;
  }
  TensorDesc y_values_tensor = op.GetOutputDesc("y_values");
  y_values_tensor.SetShape(output_values);
  y_values_tensor.SetDataType(x1_type);
  if (op.UpdateOutputDesc("y_values", y_values_tensor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Updata output y_values error.");
    return GRAPH_FAILED;
  }
  TensorDesc y_shape_tensor = op.GetOutputDesc("y_shape");
  y_shape_tensor.SetShape(output_shape);
  y_shape_tensor.SetDataType(DT_INT64);
  return op.UpdateOutputDesc("y_shape", y_shape_tensor);
}

INFER_FUNC_REG(DenseToSparseSetOperation, DenseToSparseSetOperationInfer);

IMPLEMT_INFERFUNC(SparseToSparseSetOperation, SparseToSparseSetOperationInfer) {
  if (op.GetInputsSize() != 6) {
    OP_LOGE(op.GetName().c_str(), "Inputs size is not 6.");
    return GRAPH_FAILED;
  }

  auto x1_indices_tensor = op.GetInputDesc(0);
  auto x1_values_tensor = op.GetInputDesc(1);
  auto x1_shape_tensor = op.GetInputDesc(2);
  auto x2_indices_tensor = op.GetInputDesc(3);
  auto x2_values_tensor = op.GetInputDesc(4);
  auto x2_shape_tensor = op.GetInputDesc(5);
  graphStatus status = ValidateSparseTensor(x1_indices_tensor, x1_values_tensor, x1_shape_tensor, op.GetName().c_str());
  if (status != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Validate Sparse x1 parameters error.");
    return GRAPH_FAILED;
  }
  status = ValidateSparseTensor(x2_indices_tensor, x2_values_tensor, x2_shape_tensor, op.GetName().c_str());
  if (status != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Validate Sparse x2 parameters error.");
    return GRAPH_FAILED;
  }

  Shape x1_shape = x1_shape_tensor.GetShape();
  Shape x2_shape = x2_shape_tensor.GetShape();
  int64_t output_rank_dim = 0;
  const int64_t input_rank_dim2 = x2_shape.GetDim(0);
  if (ValueKnown(x1_shape, 0)) {
    const int64_t input_rank_dim1 = x1_shape.GetDim(0);
    if (input_rank_dim1 < 2) {
      OP_LOGE(op.GetName().c_str(), "Input sparse x1 rank must > = 2.");
      return GRAPH_FAILED;
    }
    if (input_rank_dim1 != input_rank_dim2) {
      OP_LOGE(op.GetName().c_str(), "Input x1 rank must match x2 rank.");
      return GRAPH_FAILED;
    }
    output_rank_dim = input_rank_dim1;
  } else if (ValueKnown(x2_shape, 0)) {
    if (input_rank_dim2 < 2) {
      OP_LOGE(op.GetName().c_str(), "Input sparse x2 rank must > = 2.");
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
    OP_LOGE(op.GetName().c_str(), "Updata output y_indices error.");
    return GRAPH_FAILED;
  }
  TensorDesc y_values_tensor = op.GetOutputDesc("y_values");
  y_values_tensor.SetShape(output_values);
  y_values_tensor.SetDataType(x1_values_type);
  if (op.UpdateOutputDesc("y_values", y_values_tensor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Updata output y_values error.");
    return GRAPH_FAILED;
  }
  TensorDesc y_shape_tensor = op.GetOutputDesc("y_shape");
  y_shape_tensor.SetShape(output_shape);
  y_shape_tensor.SetDataType(DT_INT64);
  return op.UpdateOutputDesc("y_shape", y_shape_tensor);
}

INFER_FUNC_REG(SparseToSparseSetOperation, SparseToSparseSetOperationInfer);

}  // namespace ge
