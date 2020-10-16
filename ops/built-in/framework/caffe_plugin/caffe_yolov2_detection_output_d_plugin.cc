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
Status ParseParamsYoloV2DetectionOutput(const Message* op_origin,
                                         ge::Operator& op_dest) {
  OP_LOGI("YoloV2DetectionOutput",
          "enter into ParseParamsYoloV2DetectionOutput ------begin!!");  // trans op_src to op_dest
  const caffe::LayerParameter* layer = \
    dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (nullptr == layer) {
    OP_LOGE("YoloV2DetectionOutput",
            "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }
  // get layer
  const caffe::YoloV2DetectionOutputParameter& param = \
    layer->yolov2_detection_output_param();
  if (param.has_boxes()) {
    op_dest.SetAttr("boxes", param.boxes());
  }
  if (param.has_classes()) {
    op_dest.SetAttr("classes", param.classes());
  }
  if (param.has_coords()) {
    op_dest.SetAttr("coords", param.coords());
  }
  if (param.has_relative()) {
    op_dest.SetAttr("relative", param.relative());
  }
  if (param.has_obj_threshold()) {
    op_dest.SetAttr("obj_threshold", param.obj_threshold());
  }
  if (param.has_score_threshold()) {
    op_dest.SetAttr("score_threshold", param.score_threshold());
  }
  if (param.has_iou_threshold()) {
    op_dest.SetAttr("iou_threshold", param.iou_threshold());
  }
  if (param.has_pre_nms_topn()) {
    op_dest.SetAttr("pre_nms_topn", param.pre_nms_topn());
  }
  if (param.has_post_nms_topn()) {
    op_dest.SetAttr("post_nms_topn", param.post_nms_topn());
  }
  //  get biases
  std::vector<float> v_biases;
  for (int32_t i = 0; i < param.biases_size(); i++) {
    v_biases.push_back((param.biases(i)));
    op_dest.SetAttr("biases", v_biases);
  }

  OP_LOGI("YoloV2DetectionOutput",
          "ParseParamsYoloV2DetectionOutput ------end!!");

  return SUCCESS;
}

REGISTER_CUSTOM_OP("YoloV2DetectionOutput")
    .FrameworkType(CAFFE)
    .OriginOpType("YoloV2DetectionOutput")
    .ParseParamsFn(ParseParamsYoloV2DetectionOutput)
    .ImplyType(ImplyType::TVM);
}  // namespace domi

