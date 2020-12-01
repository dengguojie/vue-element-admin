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
 * \file rpn_ops.cpp
 * \brief
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
  if (op.GetAttr("iou_threshold", iou_threshold) == GRAPH_SUCCESS) {
    if (iou_threshold <= 0) {
      OP_LOGE(op.GetName().c_str(), "Attr iou_threshold(%f) must > 0", iou_threshold);
      return GRAPH_FAILED;
    }
  }

  TensorDesc out_box_desc = op.GetOutputDesc("selected_boxes");
  TensorDesc out_idx_desc = op.GetOutputDesc("selected_idx");
  TensorDesc out_mask_desc = op.GetOutputDesc("selected_mask");
  TensorDesc in_desc = op.GetInputDesc("box_scores");

  out_box_desc.SetShape(in_desc.GetShape());
  out_box_desc.SetDataType(in_desc.GetDataType());
  if (op.UpdateOutputDesc("selected_boxes", out_box_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dims_in = in_desc.GetShape().GetDims();
  out_idx_desc.SetShape(Shape(std::vector<int64_t>{dims_in.front()}));
  out_idx_desc.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("selected_idx", out_idx_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  out_mask_desc.SetShape(Shape(std::vector<int64_t>{dims_in.front()}));
  out_mask_desc.SetDataType(DT_BOOL);
  if (op.UpdateOutputDesc("selected_mask", out_mask_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(NMSWithMask, NMSWithMaskShapeAndType);
// ---------------- NMSWithMask Op END---------------------
}  // namespace ge
