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
 * \target_crop_and_resize.cpp
 * \brief
 */
#include "inc/target_crop_and_resize.h"

#include <string>
#include <nlohmann/json.hpp>

#include "graph/operator_reg.h"
#include "util/util.h"
#include "op_log.h"
#include "graph/utils/graph_utils.h"
#include "./util/error_util.h"
#include "graph/utils/type_utils.h"

namespace ge {
IMPLEMT_VERIFIER(TargetCropAndResize, TargetCropAndResizeVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(TargetCropAndResizeInferShape) {
  OP_LOGI("TargetCropAndResize", "infer shape begin");
  auto x_desc = op.GetInputDesc("x");
  auto x_shape = x_desc.GetShape().GetDims();
  int64_t channel = 3;
  int64_t xDimNum = op.GetInputDesc("x").GetShape().GetDimNum();
  if (x_shape.empty() || xDimNum < 4) {
    OP_LOGE(TbeGetName(op).c_str(), "get x shape failed, or x shape is smaller than 4.");
    return GRAPH_FAILED;
  }

  auto boxes_shape = op.GetInputDesc("boxes").GetShape().GetDims();
  int64_t boxesDimNum = op.GetInputDesc("boxes").GetShape().GetDimNum();
  if (boxes_shape.empty() || boxesDimNum < 2) {
    OP_LOGE(TbeGetName(op).c_str(), "get boxes shape failed, or boxes shape is smaller than 2.");
    return GRAPH_FAILED;
  }
  int64_t batch = boxes_shape[0];
  if (x_desc.GetFormat() == FORMAT_NCHW) {
    channel = x_shape[1];
  } else if (x_desc.GetFormat() == FORMAT_NHWC) {
    channel = x_shape[3];
  }

  int64_t output_h = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("output_h", output_h)) {
    OP_LOGE(TbeGetName(op).c_str(), "get attr failed");
    return GRAPH_FAILED;
  }

  int64_t output_w = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("output_w", output_w)) {
    OP_LOGE(TbeGetName(op).c_str(), "get attr failed");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dim_vector;
  dim_vector.push_back(batch);
  dim_vector.push_back(channel);
  dim_vector.push_back(output_h);
  dim_vector.push_back(output_w);
  Shape out_shape(dim_vector);
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(out_shape);
  y_desc.SetDataType(ge::DT_UINT8);
  (void)op.UpdateOutputDesc("y", y_desc);

  return GRAPH_SUCCESS;
}


COMMON_INFER_FUNC_REG(TargetCropAndResize, TargetCropAndResizeInferShape);
VERIFY_FUNC_REG(TargetCropAndResize, TargetCropAndResizeVerify);
}  // namespace ge
