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
 * \file resource_variable_ops.cpp
 * \brief
 */
#include "inc/resource_variable_ops.h"
#include "common/inc/op_log.h"
#include "common_shape_fns.h"
#include "resource_variable_ops_shape_fns.h"
#include "util/util.h"
#include "util/error_util.h"

namespace ge {

IMPLEMT_INFERFUNC(VarHandleOp, VarHandleOpInfer) {
  TensorDesc outputDesc = op.GetOutputDesc("y");
  outputDesc.SetDataType(DT_RESOURCE);
  outputDesc.SetShape(Shape());
  if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", outputDesc)) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("update output[y] desc failed."));
    return GRAPH_FAILED;
  }

  Operator::OpType type;
  if (GRAPH_SUCCESS != op.GetAttr("dtype", type)) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("get attr[dtype] failed."));
    return GRAPH_FAILED;
  }

  Operator::OpListInt dims;
  if (GRAPH_SUCCESS != op.GetAttr("shape", dims)) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("get attr[shape] failed."));
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