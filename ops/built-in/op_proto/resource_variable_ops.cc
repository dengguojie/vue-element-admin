/**
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file resource_variable_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "inc/resource_variable_ops.h"
#include "op_log.h"
#include "common_shape_fns.h"
#include "resource_variable_ops_shape_fns.h"
#include "util/util.h"

namespace ge {

IMPLEMT_INFERFUNC(VarHandleOp, VarHandleOpInfer) {
  TensorDesc outputDesc = op.GetOutputDesc("y");
  outputDesc.SetShape(Shape());
  outputDesc.SetDataType(DT_RESOURCE);
  if (op.UpdateOutputDesc("y", outputDesc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update outputDesc desc failed");
    return GRAPH_FAILED;
  }

  Operator::OpType type;
  if (op.GetAttr("dtype", type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr dtype failed");
    return GRAPH_FAILED;
  }

  Operator::OpListInt dims;
  if (op.GetAttr("shape", dims) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr shape failed");
    return GRAPH_FAILED;
  }

  Shape elemShape(std::move(dims));
  ShapeAndType shape_and_type(elemShape, type);
  std::vector<ShapeAndType> handle_shapes_and_types;
  handle_shapes_and_types.reserve(1);
  handle_shapes_and_types.emplace_back(shape_and_type);
  std::vector<std::vector<ShapeAndType>> shapes_and_types(2);
  shapes_and_types[0] = handle_shapes_and_types;
  auto context = op.GetInferenceContext();
  context->SetOutputHandleShapesAndTypes(shapes_and_types);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(VarHandleOp, VarHandleOpInfer);

IMPLEMT_INFERFUNC(AssignVariableOp, AssignVariableOpInfer) {
  return CreateAssignShapeFn(op);
}

INFER_FUNC_REG(AssignVariableOp, AssignVariableOpInfer);

IMPLEMT_INFERFUNC(AssignAddVariableOp, AssignAddVariableOpInfer) {
  return CreateAssignShapeFn(op);
}

INFER_FUNC_REG(AssignAddVariableOp, AssignAddVariableOpInfer);

IMPLEMT_INFERFUNC(AssignSubVariableOp, AssignSubVariableOpInfer) {
  return CreateAssignShapeFn(op);
}

INFER_FUNC_REG(AssignSubVariableOp, AssignSubVariableOpInfer);

}  // namespace ge