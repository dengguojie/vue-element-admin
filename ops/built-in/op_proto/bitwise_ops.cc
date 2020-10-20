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
 * \file bitwise_ops.cpp
 * \brief
 */
#include "inc/bitwise_ops.h"
#include "common_shape_fns.h"

namespace ge {

IMPLEMT_INFERFUNC(RightShift, RightShiftInfer) {
  DataType type = op.GetInputDesc("x").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("z");
  tensordesc_output.SetDataType(type);
  (void)op.UpdateOutputDesc("z", tensordesc_output);
  return BROADCAST_INFER("x", "y", "z")(op);
}

INFER_FUNC_REG(RightShift, RightShiftInfer);

}  // namespace ge
