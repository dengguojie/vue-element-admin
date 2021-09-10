/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
 * \file yolo_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
// Caffe ParseParams
Status ParseParamsYolo(const Message* op_origin, ge::Operator& op_dest) {
  OP_LOGI("Yolo", "enter into ParseParams Yolo ------begin!!");
  // trans op_src to op_dest
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (nullptr == layer) {
    OP_LOGE("Yolo", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }
  // get layer
  const caffe::YoloParameter& param = layer->yolo_param();
  if (param.has_boxes()) {
    op_dest.SetAttr("boxes", param.boxes());
  }
  if (param.has_coords()) {
    op_dest.SetAttr("coords", param.coords());
  }
  if (param.has_classes()) {
    op_dest.SetAttr("classes", param.classes());
  }
  if (param.has_yolo_version()) {
    op_dest.SetAttr("yolo_version", param.yolo_version());
  }
  if (param.has_softmax()) {
    op_dest.SetAttr("softmax", param.softmax());
  }
  if (param.has_background()) {
    op_dest.SetAttr("background", param.background());
  }
  if (param.has_softmaxtree()) {
    op_dest.SetAttr("softmaxtree", param.softmaxtree());
  }

  OP_LOGI("Yolo", "ParseParamsYolo ------end!!");

  return SUCCESS;
}

REGISTER_CUSTOM_OP("Yolo")
    .FrameworkType(CAFFE)
    .OriginOpType("Yolo")
    .ParseParamsFn(ParseParamsYolo)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
