/* Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

using namespace ge;
namespace domi {
// Caffe ParseParams
Status ParseParamsYolo(const Message* op_origin,
                       ge::Operator& op_dest) {
  OP_LOGI("Yolo",
            "enter into ParseParams Yolo ------begin!!");
  // trans op_src to op_dest
  const caffe::LayerParameter* layer = \
    dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (nullptr == layer) {
    OP_LOGE("Yolo",
            "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }
  // get layer
  const caffe::YoloParameter& param = \
    layer->yolo_param();
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

