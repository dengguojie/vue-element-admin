/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2018. All rights reserved.
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
 * \file fsrdetectionoutput_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
// Caffe ParseParams
Status ParseParams_FSRDetectionOutput(const Message* op_origin, ge::Operator& op_dest) {
  // trans op_src to op_dest
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (layer == nullptr) {
    OP_LOGE("FSRDetectionOutput", "Dynamic cast op_src to LayerParameter failed");
    return FAILED;
  }

  const caffe::FSRDetectionOutputParameter& param = layer->fsrdetectionoutput_param();
  if (param.has_num_classes()) {
    op_dest.SetAttr("num_classes", param.num_classes());
  }
  if (param.has_score_threshold()) {
    op_dest.SetAttr("score_threshold", param.score_threshold());
  }
  if (param.has_iou_threshold()) {
    op_dest.SetAttr("iou_threshold", param.iou_threshold());
  }
  if (param.has_batch_rois()) {
    op_dest.SetAttr("batch_rois", static_cast<int>(param.batch_rois()));
  }
  return SUCCESS;
}

// test_reduction is the type name of the operator in the OM model.
// It can be specified randomly and cannot be the same as an existing type name. It is case sensitive.
REGISTER_CUSTOM_OP("FSRDetectionOutput")
    .FrameworkType(CAFFE)                // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("FSRDetectionOutput")  // Reduction indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(ParseParams_FSRDetectionOutput)  // AutoMappingFn indicates automatic mapping the parameters of op.
    .ImplyType(ImplyType::TVM);
}  // namespace domi
