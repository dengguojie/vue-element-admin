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
 * \file ssddetectionoutput_ops.cpp
 * \brief
 */
#include "inc/nn_detect_ops.h"

#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#include "graph/debug/ge_log.h"
#include "util/util.h"
#include "util/error_util.h"
#include "op_log.h"

namespace ge {
IMPLEMT_INFERFUNC(SSDDetectionOutput, SSDDetectionOutputInferShape) {
  auto keep_top_k = op.get_attr_keep_top_k();
  if (keep_top_k == -1) {
    keep_top_k = 1024;
  }
  keep_top_k = static_cast<int>((keep_top_k + 127) / 128) * 128;

  auto loc_shape = op.get_input_desc_bbox_delta().GetShape().GetDims();
  if (loc_shape.empty()) {
    std::string err_msg = GetInputInvalidErrMsg("loc_shape");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  auto batch = loc_shape[0];
  auto Boxtype = op.get_input_desc_bbox_delta().GetDataType();

  vector<int64_t> actualNumShape({batch, 8});
  auto outdesc0 = op.GetOutputDesc("out_boxnum");
  outdesc0.SetShape(Shape(actualNumShape));
  outdesc0.SetDataType(ge::DT_INT32);
  (void)op.update_output_desc_out_boxnum(outdesc0);

  vector<int64_t> boxShape({batch, keep_top_k, 8});
  auto outdesc = op.GetOutputDesc("y");
  outdesc.SetShape(Shape(boxShape));
  outdesc.SetDataType(ge::DataType(Boxtype));
  (void)op.update_output_desc_y(outdesc);

  return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(SSDDetectionOutput, SSDDetectionOutputVerify) {
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(SSDDetectionOutput, SSDDetectionOutputInferShape);
VERIFY_FUNC_REG(SSDDetectionOutput, SSDDetectionOutputVerify);
}  // namespace ge
