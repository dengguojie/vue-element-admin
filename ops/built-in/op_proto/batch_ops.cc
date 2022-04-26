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
 * \file batch_ops.cpp
 * \brief
 */
#include "inc/batch_ops.h"
#include "op_log.h"
#include "util/error_util.h"
#include "util/common_shape_fns.h"
#include "util/lookup_ops_shape_fns.h"

namespace ge {
IMPLEMT_INFERFUNC(Batch, BatchInfer) {
  for (size_t i = 0; i < op.GetInputsSize(); ++i) {
    Shape out_shapes;
    if (ReplaceDim(op.GetInputDesc(i).GetShape(), 0, ge::UNKNOWN_DIM, out_shapes, TbeGetName(op).c_str()) ==
        GRAPH_FAILED) {
      std::string err_msg = ConcatString(
          "failed to call ReplaceDim function,"
          " the input x_tensors[", i, "] is a real number without 0 dimension ");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
    auto y_tensor_type = op.GetDynamicInputDesc("x_tensors", i).GetDataType();
    TensorDesc output_desc = op.GetDynamicOutputDesc("y_tensors", i);
    output_desc.SetShape(out_shapes);
    output_desc.SetDataType(y_tensor_type);
    op.UpdateDynamicOutputDesc("y_tensors", i, output_desc);
  }

  Shape scalar_shape;
  Scalar(scalar_shape);
  TensorDesc y_desc = op.GetOutputDesc("y_id");
  y_desc.SetShape(scalar_shape);
  y_desc.SetDataType(DT_INT64);
  op.UpdateOutputDesc("y_id", y_desc);

  std::vector<int64_t> dims = {ge::UNKNOWN_DIM, 3};
  TensorDesc output_desc_batch_index = op.GetOutputDesc("y_index");
  output_desc_batch_index.SetShape(Shape(dims));
  output_desc_batch_index.SetDataType(DT_INT64);
  op.UpdateOutputDesc("y_index", output_desc_batch_index);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Batch, BatchInfer);

IMPLEMT_INFERFUNC(Unbatch, UnbatchInfer) {
  Shape out_shape;
  auto x_tensor = op.GetInputDesc(0);
  if (ReplaceDim(x_tensor.GetShape(), 0, ge::UNKNOWN_DIM, out_shape, TbeGetName(op).c_str()) == GRAPH_FAILED) {
     AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
        string("failed to call ReplaceDim function, create output[y_tensor] shape failed"));
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDesc("y_tensor");
  output_desc.SetShape(out_shape);
  output_desc.SetDataType(x_tensor.GetDataType());
  if (op.UpdateOutputDesc("y_tensor", output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("update output[y_tensor] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Unbatch, UnbatchInfer);

IMPLEMT_INFERFUNC(UnbatchGrad, UnbatchGradInfer) {
  auto x_input_tensor = op.GetInputDesc(0);
  auto grad_tensor = op.GetInputDesc(2);
  auto grad_rank = grad_tensor.GetShape().GetDimNum();
  if (x_input_tensor.GetDataType() != grad_tensor.GetDataType()) {
    string err_msg = ConcatString("data type of input[x_input] is not equal to input[grad], data type of input[x_input] is ",
                                  DTypeStr(x_input_tensor.GetDataType()), ", data type of input[grad] is ",
                                  DTypeStr(grad_tensor.GetDataType()));
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  auto out_shape = UnknownShapeOfRank(grad_rank);
  TensorDesc output_desc = op.GetOutputDesc("y_grad");
  output_desc.SetShape(out_shape);
  output_desc.SetDataType(x_input_tensor.GetDataType());
  if (op.UpdateOutputDesc("y_grad", output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("update output[y_grad] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(UnbatchGrad, UnbatchGradInfer);
}  // namespace ge