/**
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file bi_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
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
