/* Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
#include "../onnx_common.h"
#include "array_ops.h"
namespace domi {

Status ParseParamsInt8Quantize(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE("Int8Quantize", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  float scale = 0.0;
  float offset = 0.0;
  bool sqrt_mode = false;
  std::string round_mode = "Round";
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "Y_scale" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      scale = attr.f();
    } else if (attr.name() == "Y_zero_point" && attr.type() == ge::onnx::AttributeProto::INT) {
      offset = static_cast<float>(attr.i());
    }
  }

  op_dest.SetAttr("scale", scale);
  op_dest.SetAttr("offset", offset);
  op_dest.SetAttr("sqrt_mode", sqrt_mode);
  op_dest.SetAttr("round_mode", round_mode);

  // the input format should be NCHW
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (op_desc == nullptr) {
    ONNX_PLUGIN_LOGE("Int8Quantize", "Get op desc failed.");
    return FAILED;
  }
  for (size_t i = 0; i < op_desc->GetInputsSize(); i++) {
    ge::GeTensorDesc tensor = op_desc->GetInputDesc(i);
    tensor.SetOriginFormat(ge::FORMAT_NCHW);
    tensor.SetFormat(ge::FORMAT_NCHW);
    auto ret_x = op_desc->UpdateInputDesc(i, tensor);
    if (ret_x != ge::GRAPH_SUCCESS) {
      ONNX_PLUGIN_LOGE("Int8Quantize", "update input format failed.");
      return FAILED;
    }
  }
  for (size_t i = 0; i < op_desc->GetOutputsSize(); i++) {
    ge::GeTensorDesc tensor = op_desc->GetOutputDesc(i);
    tensor.SetOriginFormat(ge::FORMAT_NCHW);
    tensor.SetFormat(ge::FORMAT_NCHW);
    auto ret_y = op_desc->UpdateOutputDesc(i, tensor);
    if (ret_y != ge::GRAPH_SUCCESS) {
      ONNX_PLUGIN_LOGE("Int8Quantize", "update output format failed.");
      return FAILED;
    }
  }

  return SUCCESS;
}

// register AscendQuant op info to GE
REGISTER_CUSTOM_OP("AscendQuant")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Int8Quantize", "ai.onnx::9::Int8Quantize", "ai.onnx::10::Int8Quantize",
                   "ai.onnx::11::Int8Quantize", "ai.onnx::12::Int8Quantize", "ai.onnx::13::Int8Quantize"})
    .ParseParamsFn(ParseParamsInt8Quantize)
    .ImplyType(ImplyType::TVM);
}  // namespace domi