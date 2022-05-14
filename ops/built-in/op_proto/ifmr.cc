/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file ifmr.cpp
 * \brief
 */
#include "math_ops.h"

#include <vector>
#include <string>

namespace ge {
IMPLEMT_VERIFIER(IFMR, IFMRVerify) {
  // verify the inputs of data type
  if (op.GetInputDescByName("data").GetDataType() != op.GetInputDescByName("data_min").GetDataType()) {
    return GRAPH_FAILED;
  }

  if (op.GetInputDescByName("data").GetDataType() != op.GetInputDescByName("data_max").GetDataType()) {
    return GRAPH_FAILED;
  }

  if (op.GetInputDescByName("cumsum").GetDataType() != ge::DT_INT32) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(IFMRInferShape) {
  Shape ret_shape({1, });

  TensorDesc scale = op.GetOutputDescByName("scale");
  scale.SetShape(ret_shape);
  scale.SetDataType(DT_FLOAT);
  (void)op.UpdateOutputDesc("scale", scale);

  TensorDesc offset = op.GetOutputDescByName("offset");
  offset.SetShape(ret_shape);
  offset.SetDataType(DT_FLOAT);
  (void)op.UpdateOutputDesc("offset", offset);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(IFMR, IFMRInferShape);

// Registered verify function
VERIFY_FUNC_REG(IFMR, IFMRVerify);
}  // namespace ge
