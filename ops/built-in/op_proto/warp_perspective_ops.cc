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
 * @file  warp_perspective_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "inc/warp_perspective_ops.h"
#include <unordered_set>

#include "op_log.h"
#include "common_shape_fns.h"

namespace ge {

IMPLEMT_INFERFUNC(WarpPerspective, WarpPerspectiveInfer) {
  TensorDesc input0_desc = op.GetInputDesc("x");
  input0_desc.SetDataType(DT_FLOAT);
  op.UpdateInputDesc("x", input0_desc);

  TensorDesc input1_desc = op.GetInputDesc("matrix");
  input1_desc.SetDataType(DT_FLOAT);
  op.UpdateInputDesc("matrix", input1_desc);

  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetDataType(DT_FLOAT);

  int out_height;
  int out_widht;
  if (GRAPH_SUCCESS != op.GetAttr("out_height", out_height)) {
    OP_LOGE(op.GetName().c_str(), "Get out_height failed!\n");
    return GRAPH_FAILED;
  }

  if (GRAPH_SUCCESS != op.GetAttr("out_width", out_widht)) {
    OP_LOGE(op.GetName().c_str(), "Get out_width failed!\n");
    return GRAPH_FAILED;
  }

  Shape input_shape = op.GetInputDesc("x").GetShape();
  if (input_shape.GetDimNum() != 4) {
      OP_LOGE(op.GetName().c_str(), "Input Shape dim is not 4!\n");
      return GRAPH_FAILED;
  }

  int channel = input_shape.GetDim(1);
  int N = input_shape.GetDim(0);
  output_desc.SetShape(Shape({N, channel, out_height, out_widht}));

  if (op.UpdateOutputDesc("y", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "output desc update failed!\n");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(WarpPerspective, WarpPerspectiveInfer);

}

