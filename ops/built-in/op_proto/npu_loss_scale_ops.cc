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
 * \file npu_loss_scale_ops.cpp
 * \brief
 */
#include "inc/npu_loss_scale_ops.h"
#include <string>
#include <vector>
#include <algorithm>
#include "util/util.h"
#include "op_log.h"

namespace ge {

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

// ----------------------------TBE Loss Scale------------------------------------//
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
// ----------------------------NPUGetFloatStatusV2------------------------------------//
IMPLEMT_COMMON_INFERFUNC(NPUGetFloatStatusV2InferShape) {
  TensorDesc td = op.GetOutputDesc("data");
  td.SetShape(Shape({8}));  // 8 * 4B equals 32B
  (void)op.UpdateOutputDesc("data", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(NPUGetFloatStatusV2, NPUGetFloatStatusV2Verify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(NPUGetFloatStatusV2, NPUGetFloatStatusV2InferShape);
VERIFY_FUNC_REG(NPUGetFloatStatusV2, NPUGetFloatStatusV2Verify);
// ----------------------------NPUGetFloatStatusV2------------------------------------//
// ----------------------------NPUClearFloatStatusV2------------------------------------//
IMPLEMT_COMMON_INFERFUNC(NPUClearFloatStatusV2InferShape) {
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(NPUClearFloatStatusV2, NPUClearFloatStatusV2Verify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(NPUClearFloatStatusV2, NPUClearFloatStatusV2InferShape);
VERIFY_FUNC_REG(NPUClearFloatStatusV2, NPUClearFloatStatusV2Verify);
// ----------------------------NPUClearFloatStatusV2------------------------------------//
}  // namespace ge
