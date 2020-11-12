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
 * \file quantize_ops.cpp
 * \brief
 */
#include "inc/quantize_ops.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "util/util.h"
#include "util/error_util.h"
#include "op_log.h"

namespace ge {
// ----------------Dequantize Op------------------------------------------------
IMPLEMT_COMMON_INFERFUNC(DequantizeInferShape) {
  std::string mode;
  if (op.GetAttr("mode", mode) == GRAPH_SUCCESS) {
    if (mode != "MIN_COMBINED" && mode != "MIN_FIRST" && mode != "SCALED") {
      string excepted_value = ConcatString("MIN_COMBINED,MIN_FIRST,SCALED");
      OpsAttrValueErrReport(op.GetName(), "mode", excepted_value, mode);
      OP_LOGE(op.GetName().c_str(), "Attr mode(%s) can only support MIN_COMBINED or MIN_FIRST or SCALED.",
              mode.c_str());
      return GRAPH_FAILED;
    }
  }
  Shape shape_x = op.GetInputDesc("x").GetShape();
  DataType min_range_dtype = op.GetInputDesc("min_range").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(shape_x);
  tensordesc_output.SetDataType(min_range_dtype);
  if (op.UpdateOutputDesc("y", tensordesc_output) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Dequantize, DequantizeInferShape);
// ----------------Dequantize End----------------------------------------------

IMPLEMT_COMMON_INFERFUNC(AscendQuantInferShape) {
  TensorDesc x_desc = op.GetInputDesc("x");
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetDataType(DT_INT8);
  Shape shape = x_desc.GetShape();
  y_desc.SetShape(shape);
  (void)op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AscendQuant, AscendQuantInferShape);

IMPLEMT_COMMON_INFERFUNC(AscendDequantInferShape) {
  TensorDesc y_desc = op.GetOutputDesc("y");
  int type;
  if (op.GetAttr("dtype", type) == GRAPH_SUCCESS) {
    y_desc.SetDataType((ge::DataType)type);
  }
  Shape shape = op.GetInputDesc("x").GetShape();
  y_desc.SetShape(shape);
  (void)op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AscendDequant, AscendDequantInferShape);

IMPLEMT_COMMON_INFERFUNC(AscendAntiQuantInferShape) {
  TensorDesc y_desc = op.GetOutputDesc("y");
  int type;
  if (op.GetAttr("dtype", type) == GRAPH_SUCCESS) {
    y_desc.SetDataType((ge::DataType)type);
  }
  Shape shape = op.GetInputDesc("x").GetShape();
  y_desc.SetShape(shape);
  (void)op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AscendAntiQuant, AscendAntiQuantInferShape);

IMPLEMT_COMMON_INFERFUNC(AscendDequantS16InferShape) {
  TensorDesc x_desc = op.GetInputDesc("x0");
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetDataType(DT_INT16);
  Shape shape = x_desc.GetShape();
  y_desc.SetShape(shape);
  (void)op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(AscendDequantS16, AscendDequantS16InferShape);

IMPLEMT_COMMON_INFERFUNC(AscendRequantInferShape) {
  TensorDesc x_desc = op.GetInputDesc("x");
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetDataType(DT_INT8);
  Shape shape = x_desc.GetShape();
  y_desc.SetShape(shape);
  (void)op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(AscendRequant, AscendRequantInferShape);

IMPLEMT_COMMON_INFERFUNC(AscendRequantS16InferShape) {
  TensorDesc x_desc = op.GetInputDesc("x");
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetDataType(DT_INT8);
  Shape shape = x_desc.GetShape();
  y_desc.SetShape(shape);
  (void)op.UpdateOutputDesc("y", y_desc);
  bool dual_output = false;
  if (op.GetAttr("dual_output", dual_output) == GRAPH_SUCCESS) {
    if (dual_output == true) {
      TensorDesc y1_desc = op.GetOutputDesc("y1");
      y1_desc.SetDataType(DT_INT16);
      y1_desc.SetShape(shape);
      (void)op.UpdateOutputDesc("y1", y1_desc);
    }
  }
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(AscendRequantS16, AscendRequantS16InferShape);
}  // namespace ge
