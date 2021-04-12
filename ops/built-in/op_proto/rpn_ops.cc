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
#include "util/error_util.h"
#include "op_log.h"

namespace ge {
// ---------------- NMSWithMask Op-------------------
IMPLEMT_COMMON_INFERFUNC(NMSWithMaskShapeAndType) {
  OP_LOGI(op.GetName().c_str(), "Enter op_proto inferfunction!");
  float iou_threshold;
  if (op.GetAttr("iou_threshold", iou_threshold) == GRAPH_SUCCESS) {
    if (iou_threshold <= 0) {
      std::string err_msg = GetAttrValueErrMsg("iou_threshold", ConcatString(iou_threshold), ConcatString("more than 0"));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }

  TensorDesc out_box_desc = op.GetOutputDesc("selected_boxes");
  TensorDesc out_idx_desc = op.GetOutputDesc("selected_idx");
  TensorDesc out_mask_desc = op.GetOutputDesc("selected_mask");
  TensorDesc in_desc = op.GetInputDesc("box_scores");

  out_box_desc.SetShape(in_desc.GetShape());
  out_box_desc.SetDataType(in_desc.GetDataType());

  std::vector<int64_t> dims_in = in_desc.GetShape().GetDims();
  out_idx_desc.SetShape(Shape(std::vector<int64_t>{dims_in.front()}));
  out_idx_desc.SetDataType(DT_INT32);

  out_mask_desc.SetShape(Shape(std::vector<int64_t>{dims_in.front()}));
  out_mask_desc.SetDataType(DT_BOOL);

  std::vector<std::pair<int64_t, int64_t>> shape_range;
  in_desc.GetShapeRange(shape_range);

  if (shape_range.size() > 0) {
    std::vector<std::pair<int64_t, int64_t>> out_range = shape_range;
    out_box_desc.SetShapeRange(out_range);
    out_idx_desc.SetShapeRange({out_range[0]});
    out_mask_desc.SetShapeRange({out_range[0]});
  }

  if (op.UpdateOutputDesc("selected_boxes", out_box_desc) != GRAPH_SUCCESS) {
    std::string err_msg = UpdateParamErrMsg("selected_boxes");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (op.UpdateOutputDesc("selected_idx", out_idx_desc) != GRAPH_SUCCESS) {
    std::string err_msg = UpdateParamErrMsg("selected_idx");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (op.UpdateOutputDesc("selected_mask", out_mask_desc) != GRAPH_SUCCESS) {
    std::string err_msg = UpdateParamErrMsg("selected_mask");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(NMSWithMask, NMSWithMaskShapeAndType);
// ---------------- NMSWithMask Op END---------------------
}  // namespace ge
