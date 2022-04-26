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
 * \file warp_perspective_ops.cpp
 * \brief
 */
#include "inc/warp_perspective_ops.h"
#include <unordered_set>

#include "common/inc/op_log.h"
#include "common_shape_fns.h"
#include "util/error_util.h"

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
    OP_LOGE(TbeGetName(op).c_str(), "Get out_height failed!\n");
    return GRAPH_FAILED;
  }

  if (GRAPH_SUCCESS != op.GetAttr("out_width", out_widht)) {
    OP_LOGE(TbeGetName(op).c_str(), "Get out_width failed!\n");
    return GRAPH_FAILED;
  }

  Shape input_shape = op.GetInputDesc("x").GetShape();
  if (input_shape.GetDimNum() != 4) {
    OP_LOGE(TbeGetName(op).c_str(), "Input Shape dim is not 4!\n");
    return GRAPH_FAILED;
  }

  int channel = input_shape.GetDim(1);
  int N = input_shape.GetDim(0);
  output_desc.SetShape(Shape({N, channel, out_height, out_widht}));

  if (op.UpdateOutputDesc("y", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "output desc update failed!\n");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(WarpPerspective, WarpPerspectiveInfer);

}  // namespace ge
