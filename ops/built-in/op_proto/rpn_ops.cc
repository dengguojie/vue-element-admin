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
  OP_LOGI(TbeGetName(op).c_str(), "Enter op_proto inferfunction!");
  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::ConstGeTensorDescPtr in_desc = op_desc->GetInputDescPtr(0);

  ge::GeTensorDescPtr out_box_desc_ptr = op_desc->MutableOutputDesc(0);
  ge::GeTensorDescPtr out_idx_desc_ptr = op_desc->MutableOutputDesc(1);
  ge::GeTensorDescPtr out_mask_desc_ptr = op_desc->MutableOutputDesc(2);

  if (in_desc == nullptr ||
      out_box_desc_ptr == nullptr || out_idx_desc_ptr == nullptr || out_mask_desc_ptr == nullptr) {
    OP_LOGE(TbeGetName(op).c_str(), "[TBE Compiler] Get null node ptr");
    return GRAPH_FAILED;
  }

  const GeShape &in_shape = in_desc->GetShape();
  GeShape &out_box_shape = out_box_desc_ptr->MutableShape();
  GeShape &out_idx_shape = out_idx_desc_ptr->MutableShape();
  GeShape &out_mask_shape = out_mask_desc_ptr->MutableShape();

  float iou_threshold;
  if (!AttrUtils::GetFloat(op_desc, "iou_threshold", iou_threshold)) {
    OP_LOGE(TbeGetName(op).c_str(), "[TBE Compiler] Get attr iou_threshold failed!");
    return GRAPH_FAILED;
  }
  if (iou_threshold <= 0) {
    std::string err_msg = GetAttrValueErrMsg("iou_threshold", ConcatString(iou_threshold), ConcatString("more than 0"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  out_idx_shape.SetDimNum(1);
  out_idx_shape.SetDim(0, in_shape.GetDim(0));

  out_mask_shape.SetDimNum(1);
  out_mask_shape.SetDim(0, in_shape.GetDim(0));

  // set output shape (N, 5)
  size_t dim_size = in_shape.GetDimNum();
  out_box_shape.SetDimNum(dim_size);
  out_box_shape.SetDim(0, in_shape.GetDim(0));
  out_box_shape.SetDim(1, 5);
  out_box_desc_ptr->SetShape(out_box_shape);

  out_box_desc_ptr->SetDataType(in_desc->GetDataType());
  out_idx_desc_ptr->SetDataType(DT_INT32);
  out_mask_desc_ptr->SetDataType(DT_BOOL);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(NMSWithMask, NMSWithMaskShapeAndType);
// ---------------- NMSWithMask Op END---------------------
}  // namespace ge
