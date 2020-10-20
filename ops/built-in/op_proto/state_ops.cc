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
 * \file state_ops.cpp
 * \brief
 */
#include "inc/state_ops.h"
#include "common/inc/op_log.h"
#include "common_shape_fns.h"

namespace ge {

IMPLEMT_INFERFUNC(Variable, VariableInfer) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Variable, VariableInfer);

IMPLEMT_INFERFUNC(TemporaryVariable, TemporaryVariableInfer) {
  TensorDesc td = op.GetOutputDesc("y");

  std::vector<int64_t> shape_size{};
  (void)op.GetAttr("shape", shape_size);
  td.SetShape(ge::Shape(shape_size));
  uint32_t data_type = DT_FLOAT;
  (void)op.GetAttr("dtype", data_type);
  td.SetDataType((DataType)data_type);

  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TemporaryVariable, TemporaryVariableInfer);

IMPLEMT_INFERFUNC(DestroyTemporaryVariable, DestroyTemporaryVariableInfer) {
  TensorDesc input_desc = op.GetInputDesc("x");
  (void)op.UpdateOutputDesc("y", input_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DestroyTemporaryVariable, DestroyTemporaryVariableInfer);

IMPLEMT_INFERFUNC(IsVariableInitialized, IsVariableInitializedInfer) {
  TensorDesc input_desc = op.GetInputDesc("x");
  input_desc.SetShape(ge::Shape());
  input_desc.SetDataType(DT_BOOL);
  (void)op.UpdateOutputDesc("y", input_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(IsVariableInitialized, IsVariableInitializedInfer);

IMPLEMT_INFERFUNC(VarIsInitializedOp, VarIsInitializedOpInfer) {
  TensorDesc input_desc = op.GetInputDesc("x");
  input_desc.SetShape(ge::Shape());
  input_desc.SetDataType(DT_BOOL);
  (void)op.UpdateOutputDesc("y", input_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(VarIsInitializedOp, VarIsInitializedOpInfer);

IMPLEMT_INFERFUNC(CountUpTo, CountUpToInfer) {
  Shape out;
  if (WithRank(op.GetInputDesc(0), 0, out, op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  DataType type = op.GetInputDesc("ref").GetDataType();

  TensorDesc outputDesc = op.GetOutputDesc("y");
  outputDesc.SetShape(out);
  outputDesc.SetDataType(type);
  return op.UpdateOutputDesc("y", outputDesc);
}

INFER_FUNC_REG(CountUpTo, CountUpToInfer);

}  // namespace ge
