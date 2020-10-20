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
#include "op_log.h"

namespace ge {

IMPLEMT_INFERFUNC(ROIPooling, ROIPoolingInferShape) {
  auto pooled_h = op.get_attr_pooled_h();
  auto pooled_w = op.get_attr_pooled_w();

  auto xShape = op.get_input_desc_x().GetShape().GetDims();
  auto xDtype = op.get_input_desc_x().GetDataType();
  auto roisShape = op.get_input_desc_rois().GetShape().GetDims();
  auto roi_max_num = roisShape[2];

  int64_t inputN, inputC1, poolH, poolW;
  inputN = xShape[0];
  inputC1 = xShape[1];

  poolH = pooled_h;
  poolW = pooled_w;

  vector<int64_t> yShape({roi_max_num * inputN, inputC1, poolH, poolW});

  auto outdesc = op.GetOutputDesc("y");
  outdesc.SetShape(Shape(yShape));
  outdesc.SetDataType(ge::DataType(xDtype));
  (void)op.update_output_desc_y(outdesc);

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(ROIPooling, ROIPoolingVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ROIPooling, ROIPoolingInferShape);
VERIFY_FUNC_REG(ROIPooling, ROIPoolingVerify);

}  // namespace ge
