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
#include "mask_fill.h"

namespace ge {
// ----------------MaskedFill Begin-------------------
IMPLEMT_COMMON_INFERFUNC(InferMaskedFillShape) {
  // ge::Operator op;
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x", "mask", "y", is_dynamic_output)){
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MaskedFill, MaskedFillVerify) {
  auto input_type_mask = op.GetInputDesc("mask").GetDataType();
  if (input_type_mask != DT_BOOL) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(MaskedFill, InferMaskedFillShape);
VERIFY_FUNC_REG(MaskedFill, MaskedFillVerify);
// ----------------MaskedFill END---------------------
}  // namespace ge