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
 * \file roipooling_ops.cpp
 * \brief
 */
#include "inc/nn_detect_ops.h"

#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#include "util/util.h"
#include "util/error_util.h"
#include "op_log.h"
#include "axis_util.h"

namespace ge {

IMPLEMT_INFERFUNC(ROIPooling, ROIPoolingInferShape) {
  auto pooled_h = op.get_attr_pooled_h();
  auto pooled_w = op.get_attr_pooled_w();

  auto xShape = op.get_input_desc_x().GetShape().GetDims();
  auto xDtype = op.get_input_desc_x().GetDataType();
  auto roisShape = op.get_input_desc_rois().GetShape().GetDims();
  CHECK(roisShape.size() < 2,
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), GetShapeSizeErrMsg(1, ConcatString(roisShape.size()),ConcatString("smaller than 2!"))),
      return GRAPH_FAILED);

  int64_t inputN, inputC1, poolH, poolW;
  CHECK(xShape.size() < 2,
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), GetShapeSizeErrMsg(0, ConcatString(xShape.size()), ConcatString("smaller than 2!"))),
      return GRAPH_FAILED);
  inputN = xShape[0];
  inputC1 = xShape[1];

  poolH = pooled_h;
  poolW = pooled_w;
  int64_t output_n = 0;
  if(roisShape.size() == 3) {
    output_n = roisShape[2] * inputN;
  } else if(roisShape.size() == 2) {
    output_n = roisShape[0];
  }
  vector<int64_t> yShape({output_n, inputC1, poolH, poolW});

  auto outdesc = op.GetOutputDesc("y");
  outdesc.SetShape(Shape(yShape));
  outdesc.SetDataType(ge::DataType(xDtype));
  (void)op.update_output_desc_y(outdesc);

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(ROIPooling, ROIPoolingVerify) {
  int64_t xDimNum = op.get_input_desc_x().GetShape().GetDimNum();
  if (xDimNum < 4) {
    std::string err_msg = GetShapeErrMsg(0, ConcatString(xDimNum), ConcatString("more than or equal to 4!"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t roisDimNum = op.get_input_desc_rois().GetShape().GetDimNum();
  int64_t roi_max_num = 0;
  if (roisDimNum == 3) {
    auto roisShape = op.get_input_desc_rois().GetShape().GetDims();
    roi_max_num = roisShape[2];
    if (roi_max_num > 6000 || roi_max_num % 16 != 0) {
    std::string err_msg = GetShapeErrMsg(1, ConcatString(roi_max_num), ConcatString("the dim 2 of rois shape can not be greater than 6000 and can be divided by 16!"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
    }
  } else if (roisDimNum == 2) {
    auto roisShape = op.get_input_desc_rois().GetShape().GetDims();
    roi_max_num = roisShape[0];
    if (roi_max_num > 6000) {
      string excepted_shape = ConcatString("the dim 2 of rois shape can not be greater than 6000!");
      std::string err_msg = GetShapeErrMsg(1, ConcatString(roi_max_num), excepted_shape);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
    }
  } else {
    string excepted_shape = ConcatString("The input shape of rois not equal 3 or 2!");
    std::string err_msg = GetShapeErrMsg(1, ConcatString(roi_max_num), excepted_shape);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
  

INFER_FUNC_REG(ROIPooling, ROIPoolingInferShape);
VERIFY_FUNC_REG(ROIPooling, ROIPoolingVerify);

}  // namespace ge
