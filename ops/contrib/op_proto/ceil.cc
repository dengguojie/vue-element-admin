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
 * \file ceil.cpp
 * \brief
 */
#include "ceil.h"
#include <string>
#include <vector>

namespace ge {
IMPLEMT_VERIFIER(TikCeil, CeilVerify)
{
  DataType input_type_0 = op.GetInputDesc("input_gm").GetDataType();
  printf("[Plugin][INFO] Input type:{%d}, DT_FLOAT16=%d, DT_FLOAT=%d\n", input_type_0, DT_FLOAT16, DT_FLOAT);
  if (op.GetInputsSize() != 1) {
    printf("[ERROR][Plugin] number of input must be 1\n");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
IMPLEMT_COMMON_INFERFUNC(CeilInferShape)
{
  TensorDesc tensor_desc_out = op.GetOutputDesc("output_gm");
  tensor_desc_out.SetShape(op.GetInputDesc("input_gm").GetShape());
  tensor_desc_out.SetDataType(op.GetInputDesc("input_gm").GetDataType());
  tensor_desc_out.SetFormat(op.GetInputDesc("input_gm").GetFormat());
  (void)op.UpdateOutputDesc("output_gm", tensor_desc_out);
  return GRAPH_SUCCESS;
}
// Registered inferfunction
COMMON_INFER_FUNC_REG(TikCeil, CeilInferShape);

// Registered verify function
VERIFY_FUNC_REG(TikCeil, CeilVerify);
}