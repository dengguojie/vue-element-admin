/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file map_ops.cpp
 * \brief
 */

#include "inc/map_ops.h"
#include "graph/operator.h"
#include "op_log.h"
#include "common_shape_fns.h"
#include "graph/utils/op_desc_utils.h"
#include "util/util.h"
#include "error_util.h"

namespace ge {
IMPLEMT_INFERFUNC(TensorMapHasKey, TensorMapHasKeyInfer) {
  TensorDesc yDesc = op.GetOutputDescByName("has_key");
  Shape scalarShape;
  (void)Scalar(scalarShape);
  yDesc.SetDataType(DT_BOOL);
  yDesc.SetShape(scalarShape);
  return op.UpdateOutputDesc("has_key", yDesc);
}
INFER_FUNC_REG(TensorMapHasKey, TensorMapHasKeyInfer);

IMPLEMT_INFERFUNC(TensorMapErase, TensorMapEraseInfer) {
  TensorDesc yDesc = op.GetOutputDescByName("output_handle");
  Shape scalarShape;
  (void)Scalar(scalarShape);
  yDesc.SetDataType(DT_VARIANT);
  yDesc.SetShape(scalarShape);
  return op.UpdateOutputDesc("output_handle", yDesc);
}
INFER_FUNC_REG(TensorMapErase, TensorMapEraseInfer);

IMPLEMT_INFERFUNC(TensorMapInsert, TensorMapInsertInfer) {
  TensorDesc outputDesc = op.GetOutputDescByName("output_handle");
  Shape scalarShape;
  (void)Scalar(scalarShape);
  outputDesc.SetDataType(DT_VARIANT);
  outputDesc.SetShape(scalarShape);
  return op.UpdateOutputDesc("output_handle", outputDesc);
}
INFER_FUNC_REG(TensorMapInsert, TensorMapInsertInfer);

IMPLEMT_INFERFUNC(TensorMapLookup, TensorMapLookupInfer) {
  TensorDesc outputDesc = op.GetOutputDescByName("value");
  DataType value_type;
  if (op.GetAttr("value_dtype", value_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("Op get attr value_dtype failed."));
    return GRAPH_FAILED;
  }
  outputDesc.SetDataType(value_type);
  outputDesc.SetShape(Shape(ge::UNKNOWN_SHAPE));
  return op.UpdateOutputDesc("value", outputDesc);
}
INFER_FUNC_REG(TensorMapLookup, TensorMapLookupInfer);

IMPLEMT_INFERFUNC(TensorMapSize, TensorMapSizeInfer) {
  Shape scalar_shape;
  (void)Scalar(scalar_shape);
  TensorDesc output_desc = op.GetOutputDescByName("size");
  output_desc.SetShape(scalar_shape);
  output_desc.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("size", output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("Update size desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(TensorMapSize, TensorMapSizeInfer);

IMPLEMT_INFERFUNC(TensorMapStackKeys, TensorMapStackKeysInfer) {
  DataType key_dtype;
  if (op.GetAttr("key_dtype", key_dtype) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("Get attr key_dtype failed."));
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDescByName("keys");
  output_desc.SetShape(Shape({UNKNOWN_DIM}));
  output_desc.SetDataType(key_dtype);
  if (op.UpdateOutputDesc("keys", output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("Update keys desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(TensorMapStackKeys, TensorMapStackKeysInfer);

IMPLEMT_INFERFUNC(EmptyTensorMap, EmptyTensorMapInfer) {
  Shape shape;
  if (Scalar(shape) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("Create Scalar failed"));
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDescByName("handle");
  output_desc.SetShape(shape);
  output_desc.SetDataType(DT_VARIANT);
  if (op.UpdateOutputDesc("handle", output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("Update handle desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(EmptyTensorMap, EmptyTensorMapInfer);
}  // namespace ge
