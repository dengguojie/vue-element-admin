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
 * \file elewise_calculation_ops.cpp
 * \brief
 */
#include "add.h"
#include <string>
#include <vector>

namespace ge {

IMPLEMT_VERIFIER(Add, AddVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AddInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y", is_dynamic_output)) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_DATA_SLICE_FUNC_REG(Add, ElewiseTwoInputInferDataSlice);
COMMON_INFER_FUNC_REG(Add, AddInferShape);
VERIFY_FUNC_REG(Add, AddVerify);

}  // namespace ge