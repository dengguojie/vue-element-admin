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
 * \file yolov3detectionoutputv2_plugin.cpp
 * \brief
 */
#include "../onnx_common.h"

using namespace std;
using namespace ge;

namespace domi {
Status ParseParamsYolov3detectionoutputv2(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE("Yolov3detectionoutputv2", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  int n = node->input_size();
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  op_desc->AddDynamicInputDesc("x", n);

  int N = 10;
  int boxes = 3;
  int coords = 4;
  int classes = 80;
  int post_nms_topn = 512;
  int pre_nms_topn = 512;
  int out_box_dim = 3;
  float obj_threshold = 0.5;
  float score_threshold = 0.5;
  float iou_threshold = 0.45;
  bool relative = true;
  bool resize_origin_img_to_net = false;
  std::vector<float> v_biases= {};

  for (const auto& attr : node->attribute()) {
    if (attr.name() == "boxes" && attr.type() == ge::onnx::AttributeProto::INT) {
      boxes = attr.i();
    } else if (attr.name() == "coords" && attr.type() == ge::onnx::AttributeProto::INT) {
      coords = attr.i();
    } else if (attr.name() == "classes" && attr.type() == ge::onnx::AttributeProto::INT) {
      classes = attr.i();
    } else if (attr.name() == "N" && attr.type() == ge::onnx::AttributeProto::INT) {
      N = attr.i();
    } else if (attr.name() == "post_nms_topn" && attr.type() == ge::onnx::AttributeProto::INT) {
      post_nms_topn = attr.i();
    } else if (attr.name() == "pre_nms_topn" && attr.type() == ge::onnx::AttributeProto::INT) {
      pre_nms_topn = attr.i();
    } else if (attr.name() == "out_box_dim" && attr.type() == ge::onnx::AttributeProto::INT) {
      out_box_dim = attr.i();
    } else if (attr.name() == "obj_threshold" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      obj_threshold = attr.f();
    } else if (attr.name() == "score_threshold" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      score_threshold = attr.f();
    } else if (attr.name() == "iou_threshold" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      iou_threshold = attr.f();
    } else if (attr.name() == "biases" && attr.type() == ge::onnx::AttributeProto::FLOATS) {
      for (auto biases_f : attr.floats()) {
        v_biases.push_back(biases_f);
      }
    }
  }

  if (v_biases.empty()) {
    ONNX_PLUGIN_LOGE("Yolov3detectionoutputv2", "The attr of biases is required.");
    return FAILED;
  }

  op_dest.SetAttr("N", N);
  op_dest.SetAttr("biases", v_biases);
  op_dest.SetAttr("boxes", boxes);
  op_dest.SetAttr("coords", coords);
  op_dest.SetAttr("classes", classes);
  op_dest.SetAttr("relative", relative);
  op_dest.SetAttr("post_nms_topn", post_nms_topn);
  op_dest.SetAttr("pre_nms_topn", pre_nms_topn);
  op_dest.SetAttr("out_box_dim", out_box_dim);
  op_dest.SetAttr("obj_threshold", obj_threshold);
  op_dest.SetAttr("score_threshold", score_threshold);
  op_dest.SetAttr("iou_threshold", iou_threshold);
  op_dest.SetAttr("resize_origin_img_to_net", resize_origin_img_to_net);
  return SUCCESS;
}

// register YoloV3DetectionOutputV2 op info to GE
REGISTER_CUSTOM_OP("YoloV3DetectionOutputV2")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::YoloV3DetectionOutputV2",
                   "ai.onnx::9::YoloV3DetectionOutputV2",
                   "ai.onnx::10::YoloV3DetectionOutputV2",
                   "ai.onnx::11::YoloV3DetectionOutputV2",
                   "ai.onnx::12::YoloV3DetectionOutputV2",
                   "ai.onnx::13::YoloV3DetectionOutputV2"})
    .ParseParamsFn(ParseParamsYolov3detectionoutputv2)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
