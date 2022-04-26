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
 * \file logging_ops.cpp
 * \brief
 */
#include "inc/logging_ops.h"
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/error_util.h"

namespace ge {
IMPLEMT_INFERFUNC(Timestamp, TimestampInfer) {
  TensorDesc yDesc = op.GetOutputDesc("y");
  Shape scalarShape;
  (void)Scalar(scalarShape);
  yDesc.SetDataType(DT_DOUBLE);
  yDesc.SetShape(scalarShape);
  return op.UpdateOutputDesc("y", yDesc);
}

INFER_FUNC_REG(Timestamp, TimestampInfer);

IMPLEMT_INFERFUNC(Assert, AssertInfer) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Assert, AssertInfer);

IMPLEMT_INFERFUNC(Print, PrintInfer) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Print, PrintInfer);

IMPLEMT_INFERFUNC(PrintV2, PrintV2Infer) {
  Shape unused;
  if (WithRank(op.GetInputDesc(0), 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(op.GetInputDesc(0).GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(PrintV2, PrintV2Infer);

//----------------PrintV3----------------
IMPLEMT_INFERFUNC(PrintV3, PrintV3Infer) {
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(op.GetInputDesc("x").GetShape());
  output_desc.SetDataType(op.GetInputDesc("x").GetDataType());
  op.UpdateOutputDesc("y", output_desc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(PrintV3, PrintV3Infer);
//----------------PrintV3End----------------
}  // namespace ge
