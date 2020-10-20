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
  if (op.GetAttr("iou_threshold", iou_threshold) == ge::GRAPH_SUCCESS) {
    if (iou_threshold <= 0) {
      OP_LOGE(op.GetName().c_str(), "iou_threshold(%f) must > 0", iou_threshold);
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
}  // namespace ge
