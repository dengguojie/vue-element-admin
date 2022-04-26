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
 * \file copy_ops.cpp
 * \brief
 */
#include "inc/array_ops.h"

#include <string>
#include <vector>

#include "util/util.h"
#include "op_log.h"

namespace ge {

IMPLEMT_INFERFUNC(Copy, CopyInferShape) {
  TensorDesc tensordesc = op.GetInputDesc("x");
  Shape input_shape = tensordesc.GetShape();
  DataType input_dtype = tensordesc.GetDataType();
  Format input_format = tensordesc.GetFormat();
  TensorDesc td = op.GetOutputDesc("y");

  int64_t top_size;
  if (GRAPH_SUCCESS != op.GetAttr("N", top_size)) {
    OP_LOGE(TbeGetName(op).c_str(), "GetAttr of N failed.");
    return GRAPH_FAILED;
  }

  if (top_size < 1) {
    OP_LOGE(TbeGetName(op).c_str(), "the number of top need greater than or equals to 1.");
    return GRAPH_FAILED;
  }

  for (int64_t i = 0; i < top_size; ++i) {
    td.SetShape(input_shape);
    td.SetDataType(input_dtype);
    td.SetFormat(input_format);
    op.UpdateDynamicOutputDesc("y", i, td);
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Copy, CopyVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Copy, CopyInferShape);
VERIFY_FUNC_REG(Copy, CopyVerify);

}  // namespace ge
