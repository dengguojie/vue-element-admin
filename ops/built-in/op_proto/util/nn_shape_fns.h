/**
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this
 * file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file nn_shape_fns.h
 *
 * @brief
 *
 * @version 1.0
 *
 */

#ifndef GE_NN_OPS_SHAPE_FN_H
#define GE_NN_OPS_SHAPE_FN_H

#include "operator.h"
#include "graph/debug/ge_log.h"

namespace ge {
#define UNCHANGED_SHAPE()                                      \
  TensorDesc outputDesc = op.GetOutputDesc("y");               \
  outputDesc.SetShape(op.GetInputDesc(0).GetShape());          \
  outputDesc.SetDataType(op.GetInputDesc(0).GetDataType());    \
  outputDesc.SetFormat(FORMAT_NCHW);                           \
  if (op.UpdateOutputDesc("y", outputDesc) != GRAPH_SUCCESS) { \
    GE_LOGE("fail to update output y.");                       \
    return GRAPH_FAILED;                                       \
  }                                                            \
  return GRAPH_SUCCESS;
}
#endif  // GE_NN_OPS_SHAPE_FN_H
