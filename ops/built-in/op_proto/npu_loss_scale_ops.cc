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
 * @file npu_loss_scale_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
**/

#include "inc/npu_loss_scale_ops.h"
#include <string>
#include <vector>
#include <algorithm>
#include "util/util.h"
#include "op_log.h"


namespace ge
{

IMPLEMT_COMMON_INFERFUNC(NPUClearFloatStatusOperatorInferShape) {
  TensorDesc td = op.GetOutputDesc("data");
  td.SetShape(Shape({8}));
  (void)op.UpdateOutputDesc("data", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(NPUClearFloatStatusOperator, NPUClearFloatStatusOperatorVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(NPUClearFloatStatusOperator, NPUClearFloatStatusOperatorInferShape);
VERIFY_FUNC_REG(NPUClearFloatStatusOperator, NPUClearFloatStatusOperatorVerify);

IMPLEMT_COMMON_INFERFUNC(NPUGetFloatStatusOperatorInferShape) {
  TensorDesc td = op.GetOutputDesc("data");
  td.SetShape(Shape({8}));
  (void)op.UpdateOutputDesc("data", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(NPUGetFloatStatusOperator, NPUGetFloatStatusOperatorVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(NPUGetFloatStatusOperator, NPUGetFloatStatusOperatorInferShape);
VERIFY_FUNC_REG(NPUGetFloatStatusOperator, NPUGetFloatStatusOperatorVerify);

IMPLEMT_COMMON_INFERFUNC(NPUAllocFloatStatusOperatorInferShape) {
  TensorDesc td = op.GetOutputDesc("data");
  td.SetShape(Shape({8}));
  (void)op.UpdateOutputDesc("data", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(NPUAllocFloatStatusOperator, NPUAllocFloatStatusOperatorVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(NPUAllocFloatStatusOperator, NPUAllocFloatStatusOperatorInferShape);
VERIFY_FUNC_REG(NPUAllocFloatStatusOperator, NPUAllocFloatStatusOperatorVerify);

//----------------------------TBE Loss Scale------------------------------------//
IMPLEMT_COMMON_INFERFUNC(NPUClearFloatStatusInferShape) {
  TensorDesc td = op.GetOutputDesc("data");
  td.SetShape(Shape({8}));
  (void)op.UpdateOutputDesc("data", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(NPUClearFloatStatus, NPUClearFloatStatusVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(NPUClearFloatStatus, NPUClearFloatStatusInferShape);
VERIFY_FUNC_REG(NPUClearFloatStatus, NPUClearFloatStatusVerify);

IMPLEMT_COMMON_INFERFUNC(NPUGetFloatStatusInferShape) {
  TensorDesc td = op.GetOutputDesc("data");
  td.SetShape(Shape({8}));
  (void)op.UpdateOutputDesc("data", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(NPUGetFloatStatus, NPUGetFloatStatusVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(NPUGetFloatStatus, NPUGetFloatStatusInferShape);
VERIFY_FUNC_REG(NPUGetFloatStatus, NPUGetFloatStatusVerify);

IMPLEMT_COMMON_INFERFUNC(NPUAllocFloatStatusInferShape) {
  TensorDesc td = op.GetOutputDesc("data");
  td.SetShape(Shape({8}));
  (void)op.UpdateOutputDesc("data", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(NPUAllocFloatStatus, NPUAllocFloatStatusVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(NPUAllocFloatStatus, NPUAllocFloatStatusInferShape);
VERIFY_FUNC_REG(NPUAllocFloatStatus, NPUAllocFloatStatusVerify);

} // namespace ge
