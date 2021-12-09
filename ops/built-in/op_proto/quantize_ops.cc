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

#include "graph/types.h"
#include "op_log.h"
#include "util/error_util.h"
#include "util/util.h"

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
  auto op_info = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto min_range = op_info->MutableInputDesc("min_range");
  vector<int64_t> shape_x = input_desc->MutableShape().GetDims();

  auto tensordesc_output = op_info->MutableOutputDesc("y");

  if (IsUnknown(shape_x)) {
    std::vector<std::pair<int64_t, int64_t>> shape_range;
    input_desc->GetShapeRange(shape_range);
    DataType min_range_dtype = min_range->GetDataType();

    tensordesc_output->SetShape(GeShape(shape_x));
    tensordesc_output->SetDataType(min_range_dtype);
    tensordesc_output->SetShapeRange(shape_range);
  } else {
    tensordesc_output->SetShape(GeShape(shape_x));
  }

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Dequantize, DequantizeInferShape);

// ----------------Quantize Beg----------------------------------------------
IMPLEMT_COMMON_INFERFUNC(QuantizeInferShape) {
  TensorDesc output_desc_y = op.GetOutputDesc("y");
  DataType predict_dtype = DT_INT8;
  Format predict_format = op.GetInputDesc("x").GetFormat();
  ge::Shape output_shape = op.GetInputDesc("x").GetShape();
  string dtype = "torch.qint8";
  op.GetAttr("dtype", dtype);
  if (dtype == "torch.qint8") {
      predict_dtype = DT_INT8;
  } else if(dtype == "torch.quint8") {
      predict_dtype = DT_UINT8;
  } else if(dtype == "torch.qint32") {
      predict_dtype = DT_INT32;
  } else {
      OP_LOGI(op.GetName().c_str(), "The dtype is not supported.");
      return GRAPH_FAILED;
  }
  output_desc_y.SetDataType(predict_dtype);
  output_desc_y.SetFormat(predict_format);
  output_desc_y.SetShape(output_shape);
  (void)op.UpdateOutputDesc("y", output_desc_y);
  return GRAPH_SUCCESS;
  }

COMMON_INFER_FUNC_REG(Quantize, QuantizeInferShape);
// ----------------Quantize End----------------------------------------------

// ----------------Dequantize End----------------------------------------------

IMPLEMT_COMMON_INFERFUNC(AscendQuantInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    auto op_info = OpDescUtils::GetOpDescFromOperator(op);
    auto output_desc = op_info->MutableOutputDesc(0);
    output_desc->SetDataType(DT_INT8);
    int type;
    if (op.GetAttr("dst_type", type) == GRAPH_SUCCESS) {
      output_desc->SetDataType((ge::DataType)type);
    }
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(AscendQuant, AscendQuantInferShape);

IMPLEMT_COMMON_INFERFUNC(AscendDequantInferShape) {
  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  OP_LOGD(opName.GetString(), "[AscendDequant Inferformat] Finaly deq_scale format is %d", FORMAT_NCHW);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);

  auto tensordesc_input = op_desc->MutableInputDesc("deq_scale");
  tensordesc_input->SetOriginFormat(FORMAT_NCHW);
  tensordesc_input->SetFormat(FORMAT_NCHW);

  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    int type;
    if (op.GetAttr("dtype", type) == GRAPH_SUCCESS) {
      auto output_desc = op_desc->MutableOutputDesc(0);
      output_desc->SetDataType((ge::DataType)type);
    }
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(AscendDequant, AscendDequantInferShape);

IMPLEMT_COMMON_INFERFUNC(AscendAntiQuantInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    auto op_info = OpDescUtils::GetOpDescFromOperator(op);
    auto output_desc = op_info->MutableOutputDesc(0);

    int type;
    if (op.GetAttr("dtype", type) == GRAPH_SUCCESS) {
      output_desc->SetDataType((ge::DataType)type);
      return GRAPH_SUCCESS;
    }
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(AscendAntiQuant, AscendAntiQuantInferShape);

IMPLEMT_COMMON_INFERFUNC(AscendDequantS16InferShape) {
  if (OneInOneOutDynamicInfer(op, "x0", {"y"})) {
    auto op_info = OpDescUtils::GetOpDescFromOperator(op);
    auto output_desc = op_info->MutableOutputDesc(0);
    output_desc->SetDataType(DT_INT16);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(AscendDequantS16, AscendDequantS16InferShape);

IMPLEMT_COMMON_INFERFUNC(AscendRequantInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    auto op_info = OpDescUtils::GetOpDescFromOperator(op);
    auto output_desc = op_info->MutableOutputDesc(0);
    output_desc->SetDataType(DT_INT8);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(AscendRequant, AscendRequantInferShape);

IMPLEMT_COMMON_INFERFUNC(AscendRequantS16InferShape) {
  bool dual_output = false;
  if (op.GetAttr("dual_output", dual_output) == GRAPH_SUCCESS && dual_output) {
    if (OneInOneOutDynamicInfer(op, "x0", {"y0", "y1"})) {
      auto op_info = OpDescUtils::GetOpDescFromOperator(op);

      auto output_desc_y0 = op_info->MutableOutputDesc(0);
      output_desc_y0->SetDataType(DT_INT8);

      auto output_desc_y1 = op_info->MutableOutputDesc(1);
      output_desc_y1->SetDataType(DT_INT16);

      return GRAPH_SUCCESS;
    }
  } else {
    if (OneInOneOutDynamicInfer(op, "x0", {"y0"})) {
      auto op_info = OpDescUtils::GetOpDescFromOperator(op);
      auto output_desc_y0 = op_info->MutableOutputDesc(0);
      output_desc_y0->SetDataType(DT_INT8);
      return GRAPH_SUCCESS;
    }
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(AscendRequantS16, AscendRequantS16InferShape);
}  // namespace ge
