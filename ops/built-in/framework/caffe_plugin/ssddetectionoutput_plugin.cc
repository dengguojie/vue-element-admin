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
 * \file ssddetectionoutput_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
// Caffe ParseParams
Status ParseParams_SSDDetectionOutput(const Message* op_origin, ge::Operator& op_dest)
{
  // trans op_src to op_dest
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (layer == nullptr) {
    OP_LOGE("SSDDetectionOutput", "Dynamic cast op_src to LayerParameter failed");
    return FAILED;
  }
  const caffe::SSDDetectionOutputParameter& param = layer->ssddetectionoutput_param();

  if (param.has_num_classes()) {
    op_dest.SetAttr("num_classes", param.num_classes());
  }
  if (param.has_share_location()) {
    op_dest.SetAttr("share_location", param.share_location());
  }
  if (param.has_background_label_id()) {
    op_dest.SetAttr("background_label_id", param.background_label_id());
  }
  if (param.has_iou_threshold()) {
    op_dest.SetAttr("iou_threshold", param.iou_threshold());
  }
  if (param.has_top_k()) {
    op_dest.SetAttr("top_k", param.top_k());
  }
  if (param.has_eta()) {
    op_dest.SetAttr("eta", param.eta());
  }
  if (param.has_variance_encoded_in_target()) {
    op_dest.SetAttr("variance_encoded_in_target", param.variance_encoded_in_target());
  }
  if (param.has_code_type()) {
    op_dest.SetAttr("code_type", param.code_type());
  }
  if (param.has_keep_top_k()) {
    op_dest.SetAttr("keep_top_k", param.keep_top_k());
  }
  if (param.has_confidence_threshold()) {
    op_dest.SetAttr("confidence_threshold", param.confidence_threshold());
  }

  return SUCCESS;
}
// test_reduction is the type name of the operator in the OM model.
// It can be specified randomly and cannot be the same as an existing type name. It is case sensitive.
REGISTER_CUSTOM_OP("SSDDetectionOutput")
    .FrameworkType(CAFFE)                // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("SSDDetectionOutput")  // Reduction indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(ParseParams_SSDDetectionOutput)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
