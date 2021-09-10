/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"

namespace domi {
// Caffe ParseParams
Status ParseParamsYoloV3DetectionOutputV2(const Message* op_origin, ge::Operator& op_dest) {
  OP_LOGI("YoloV3DetectionOutputV2", "enter into ParseParamsYoloV3DetectionOutputV2 ------begin!!");
  // trans op_src to op_dest
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (nullptr == layer) {
    OP_LOGE("YoloV3DetectionOutputV2", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }
  // get layer
  const caffe::YoloV3DetectionOutputV2Parameter& param = layer->yolov3_detection_output_v2_param();
  int n = layer->bottom_size();
  op_dest.SetAttr("N", n);
  op_dest.SetAttr("out_box_dim", 3);
  std::shared_ptr<ge::OpDesc> op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  op_desc->AddDynamicInputDesc("x", n);
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
  if (param.has_resize_origin_img_to_net()) {
    op_dest.SetAttr("resize_origin_img_to_net", param.resize_origin_img_to_net());
  }
  if (param.has_out_box_dim()) {
    op_dest.SetAttr("out_box_dim", param.out_box_dim());
  }
  //  get biases
  std::vector<float> v_biases;
  for (int32_t i = 0; i < param.biases_size(); i++) {
    v_biases.push_back((param.biases(i)));
  }
  // compatible for current use
  for (int32_t i = 0; i < param.biases_low_size(); i++) {
    v_biases.push_back((param.biases_low(i)));
  }
  for (int32_t i = 0; i < param.biases_mid_size(); i++) {
    v_biases.push_back((param.biases_mid(i)));
  }
  for (int32_t i = 0; i < param.biases_high_size(); i++) {
    v_biases.push_back((param.biases_high(i)));
  }
  op_dest.SetAttr("biases", v_biases);

  OP_LOGI("YoloV3DetectionOutputV2", "ParseParamsYoloV3DetectionOutputV2 ------end!!");
  return SUCCESS;
}

REGISTER_CUSTOM_OP("YoloV3DetectionOutputV2D")
    .FrameworkType(CAFFE)
    .OriginOpType("YoloV3DetectionOutputV2")
    .ParseParamsFn(ParseParamsYoloV3DetectionOutputV2)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
