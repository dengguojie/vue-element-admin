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
 * \file parsing_ops.cpp
 * \brief
 */
#include "inc/parsing_ops.h"
#include "common/inc/op_log.h"
#include "common_shape_fns.h"
#include "graph/operator.h"
#include "util/util.h"
#include "util/common_shape_fns.h"
#include "util/error_util.h"
#include "op_log.h"
#include "graph/utils/node_utils.h"
#include <string>

namespace ge {

IMPLEMT_INFERFUNC(StringToNumber, StringToNumberInfer) {
  TensorDesc out_desc = op.GetOutputDesc("y");
  DataType out_type;
  if (op.GetAttr("out_type", out_type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attribute failed");
    return GRAPH_FAILED;
  }
  if ((out_type != DT_FLOAT) || (out_type != DT_DOUBLE) || (out_type != DT_INT32) || (out_type != DT_INT64)) {
    OP_LOGE(op.GetName().c_str(), "out_type type not supported");
    return GRAPH_FAILED;
  }

  out_desc.SetDataType(out_type);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update y failed");
    return GRAPH_FAILED;
  }
  return UnchangedShape(op, "x", "y");
}
INFER_FUNC_REG(StringToNumber, StringToNumberInfer);

IMPLEMT_INFERFUNC(DecodeRaw, DecodeRawInfer) {
  int64_t unused_dim = 0;
  auto x1_tensor = op.GetInputDesc(0);
  Shape s = x1_tensor.GetShape();
  std::vector<int64_t> dims;
  for (int i = 0; i< s.GetDimNum(); i++) {
    dims.push_back(s.GetDim(i));
  }
  dims.push_back(UNKNOWN_DIM);
  Shape output_shape(dims);
  DataType dtype;
  if (op.GetAttr("out_type", dtype) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr dtype failed");
    return GRAPH_FAILED;
  }
  TensorDesc y_tensor = op.GetOutputDesc("output");
  y_tensor.SetDataType(dtype);
  y_tensor.SetShape(output_shape);
  if (op.UpdateOutputDesc("output", y_tensor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update output failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(DecodeRaw, DecodeRawInfer);

IMPLEMT_INFERFUNC(ParseTensor, ParseTensorInfer) {
  Shape output_shape(ge::UNKNOWN_RANK);
  DataType out_type;
  if (op.GetAttr("out_type", out_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
                                       string("get attr[out_type] failed"));
    return GRAPH_FAILED;
  }

  TensorDesc output_desc = op.GetOutputDesc("output");
  output_desc.SetShape(output_shape);
  output_desc.SetDataType(out_type);
  if (op.UpdateOutputDesc("output", output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("update output[output] desc failed"));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(ParseTensor, ParseTensorInfer);

IMPLEMT_INFERFUNC(DecodeCSV, DecodeCSVInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc == nullptr) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("get op desc failed, op desc is nullptr"));
    return GRAPH_FAILED;
  }

  size_t inputs_size = op_desc->GetInputsSize();
  size_t record_defaults_size = inputs_size - 1;
  for (int i = 0; i < record_defaults_size; ++i) {
    GeShape record_default_shape;
    auto temp_record_default_desc =
        op_desc->MutableInputDesc("record_defaults" + std::to_string(i));
    if (temp_record_default_desc == nullptr) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(
          op.GetName(), string("get input[record_default] desc failed, "
                               "input[record_default] desc is nullptr"));
      return GRAPH_FAILED;
    }

    std::string err_msg;
    if (WithRankAtMost(temp_record_default_desc, 1, record_default_shape) !=
        GRAPH_SUCCESS) {
      err_msg = GetShapeErrMsg(
          i, DebugString(temp_record_default_desc->GetShape().GetDims()),
          "at most 1D");
      err_msg = string(
                    "failed to call WithRankAtMost function, dynamic input "
                    "record_defaults") +
                err_msg;
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    if (record_default_shape.GetDimNum() == 1 &&
        record_default_shape.GetDim(0) > 1) {
      err_msg =
          ConcatString("shape of dynamic input record_defaults[", i,
                       "] must be at length-0 or length-1 vector or a "
                       "scalar, got record_defaults[",
                       i, "][0] dim[", record_default_shape.GetDim(0), "]");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_PARAM_INVALID;
    }
  }

  size_t outputs_size = op_desc->GetOutputsSize();
  for (int i = 0; i < outputs_size; ++i) {
    auto temp_record_default_desc =
        op.GetDynamicInputDesc("record_defaults", i);
    auto temp_output_desc = op.GetDynamicOutputDesc("output", i);
    (void)FillOpDesc(temp_output_desc, temp_record_default_desc.GetShape(),
                     temp_record_default_desc.GetDataType());
    op.UpdateDynamicOutputDesc("output", i, temp_output_desc);
  }

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(DecodeCSV, DecodeCSVInfer);

}  // namespace ge
