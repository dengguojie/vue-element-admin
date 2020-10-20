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

}  // namespace ge
