/**
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in complian
ce with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file nn_training_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "inc/rpn_ops.h"
#include <string>
#include <vector>
#include <map>
#include "util/util.h"
#include "op_log.h"

namespace ge {
// ---------------- NMSWithMask Op-------------------
IMPLEMT_COMMON_INFERFUNC(NMSWithMaskShapeAndType) {
  OP_LOGI(op.GetName().c_str(), "Enter op_proto inferfunction!");
  float iou_threshold;
  if (op.GetAttr("iou_threshold", iou_threshold) == ge::GRAPH_SUCCESS) {
    if (iou_threshold <= 0) {
      OP_LOGE(op.GetName().c_str(), "iou_threshold(%f) must > 0",
              iou_threshold);
      return GRAPH_FAILED;
    }
  }

  TensorDesc outBoxDesc = op.GetOutputDesc("selected_boxes");
  TensorDesc outIdxDesc = op.GetOutputDesc("selected_idx");
  TensorDesc outMaskDesc = op.GetOutputDesc("selected_mask");
  TensorDesc inDesc = op.GetInputDesc("box_scores");

  outBoxDesc.SetShape(inDesc.GetShape());
  outBoxDesc.SetDataType(inDesc.GetDataType());
  (void)op.UpdateOutputDesc("selected_boxes", outBoxDesc);

  std::vector<int64_t> dims_in = inDesc.GetShape().GetDims();

  outIdxDesc.SetShape(ge::Shape(std::vector<int64_t>{dims_in.front()}));
  outIdxDesc.SetDataType(DT_INT32);
  (void)op.UpdateOutputDesc("selected_idx", outIdxDesc);

  outMaskDesc.SetShape(ge::Shape(std::vector<int64_t>{dims_in.front()}));
  outMaskDesc.SetDataType(DT_UINT8);
  (void)op.UpdateOutputDesc("selected_mask", outMaskDesc);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(NMSWithMask, NMSWithMaskShapeAndType);
// ---------------- NMSWithMask Op END---------------------
} // namespace ge
