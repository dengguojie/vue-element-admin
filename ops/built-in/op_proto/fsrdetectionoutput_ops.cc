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
 * \file fsrdetectionoutput_ops.cpp
 * \brief
 */
#include "inc/nn_detect_ops.h"

#include <string.h>

#include "util/util.h"
#include "op_log.h"

namespace ge {
// ----------------FSRDetectionOutput-------------------
IMPLEMT_INFERFUNC(FSRDetectionOutput, FSRDetectionOutputInferShape) {
  auto batch_rois = op.get_attr_batch_rois();
  batch_rois = op.get_input_desc_rois().GetShape().GetDims().at(0);
  auto post_nms_topn = op.get_input_desc_rois().GetShape().GetDims().at(2);
  if (post_nms_topn >= 1024) {
    post_nms_topn = 1024;
  }
  auto num_classes = op.get_attr_num_classes();

  auto priorShape = op.get_input_desc_bbox_delta().GetShape().GetDims();
  auto priorDtype = op.get_input_desc_bbox_delta().GetDataType();
  vector<int64_t> actualNumShape({batch_rois, num_classes, 8});
  auto outdesc0 = op.GetOutputDesc("actual_bbox_num");
  outdesc0.SetShape(Shape(actualNumShape));
  outdesc0.SetDataType(ge::DataType(3));
  (void)op.update_output_desc_actual_bbox_num(outdesc0);
  vector<int64_t> boxShape({batch_rois, num_classes, post_nms_topn, 8});
  auto outdesc = op.GetOutputDesc("box");
  outdesc.SetShape(Shape(boxShape));
  outdesc.SetDataType(ge::DataType(priorDtype));
  (void)op.update_output_desc_box(outdesc);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(FSRDetectionOutput, FSRDetectionOutputVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(FSRDetectionOutput, FSRDetectionOutputInferShape);
VERIFY_FUNC_REG(FSRDetectionOutput, FSRDetectionOutputVerify);
}  // namespace ge
