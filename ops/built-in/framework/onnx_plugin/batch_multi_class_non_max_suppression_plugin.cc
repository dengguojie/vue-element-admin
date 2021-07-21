/* Copyright (c) Huawei Technologies Co., Ltd. 2012-2020. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;
Status ParseParamsBatchMultiClassNMS(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  float iou_threshold = 0.0;
  float score_threshold = 0.0;
  int max_size_per_class = 0;
  int max_total_size = 0;

  for (auto attr : node->attribute()) {
    if (attr.name() == "iou_threshold" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      iou_threshold = attr.f();
      op_dest.SetAttr("iou_threshold", iou_threshold);
    } else if (attr.name() == "score_threshold" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      score_threshold = attr.f();
      op_dest.SetAttr("score_threshold", score_threshold);
    } else if (attr.name() == "max_size_per_class" && attr.type() == ge::onnx::AttributeProto::INT) {
      max_size_per_class = attr.i();
      op_dest.SetAttr("max_size_per_class", max_size_per_class);
    } else if (attr.name() == "max_total_size" && attr.type() == ge::onnx::AttributeProto::INT) {
      max_total_size = attr.i();
      op_dest.SetAttr("max_total_size", max_total_size);
    }
  }

  return SUCCESS;
}

// register BatchMultiClassNMS op info to GE
REGISTER_CUSTOM_OP("BatchMultiClassNonMaxSuppression")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::BatchMultiClassNMS",
                   "ai.onnx::9::BatchMultiClassNMS",
                   "ai.onnx::10::BatchMultiClassNMS",
                   "ai.onnx::11::BatchMultiClassNMS",
                   "ai.onnx::12::BatchMultiClassNMS",
                   "ai.onnx::13::BatchMultiClassNMS"})
    .ParseParamsFn(ParseParamsBatchMultiClassNMS)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
