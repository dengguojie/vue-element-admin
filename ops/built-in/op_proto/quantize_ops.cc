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
* @file quantize_ops.cpp
*
* @brief
*
* @version 1.0
*
*/

#include "inc/quantize_ops.h"
#include <cmath>
#include <string>
#include <string.h>
#include <vector>
#include "util/util.h"
#include "op_log.h"
#include "util/error_util.h"

namespace ge {
// ----------------Dequantize Op------------------------------------------------
IMPLEMT_COMMON_INFERFUNC(DequantizeInferShape) {
  std::string mode;
  if (op.GetAttr("mode", mode) == GRAPH_SUCCESS) {
    if (mode != "MIN_COMBINED" && mode != "MIN_FIRST" && mode != "SCALED") {
      string excepted_value = Strcat("MIN_COMBINED,MIN_FIRST,SCALED");
      OpsAttrValueErrReport(op.GetName(), "mode", excepted_value, mode);
      OP_LOGE(op.GetName().c_str(),
              "attr mode(%s) can only support"
              " MIN_COMBINED or MIN_FIRST or SCALED.",
              mode.c_str());
      return GRAPH_FAILED;
    }
  }
  Shape shape_x = op.GetInputDesc("x").GetShape();
  DataType min_range_dtype = op.GetInputDesc("min_range").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(shape_x);
  tensordesc_output.SetDataType(min_range_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
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

IMPLEMT_COMMON_INFERFUNC(AscendDequantInferShape)
{
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

IMPLEMT_COMMON_INFERFUNC(AscendAntiQuantInferShape)
{
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

IMPLEMT_COMMON_INFERFUNC(AscendDequantS16InferShape)
{
  TensorDesc x_desc = op.GetInputDesc("x0");
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetDataType(DT_INT16);
  Shape shape = x_desc.GetShape();
  y_desc.SetShape(shape);
  (void)op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(AscendDequantS16, AscendDequantS16InferShape);

IMPLEMT_COMMON_INFERFUNC(AscendRequantInferShape)
{
  TensorDesc x_desc = op.GetInputDesc("x");
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetDataType(DT_INT8);
  Shape shape = x_desc.GetShape();
  y_desc.SetShape(shape);
  (void)op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(AscendRequant, AscendRequantInferShape);

IMPLEMT_COMMON_INFERFUNC(AscendRequantS16InferShape)
{
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
} // namespace ge
