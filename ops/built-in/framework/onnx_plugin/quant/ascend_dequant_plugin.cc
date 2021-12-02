/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
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

Status ParseParamsAscendDequant(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int dtype = 0;  // DT_FLOAT
  bool sqrt_mode = false;
  bool relu_flag = false;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "dtype" && attr.type() == ge::onnx::AttributeProto::INT) {
      dtype = attr.i();
    } else if (attr.name() == "sqrt_mode" && attr.i() != 0) {
      sqrt_mode = true;
    } else if (attr.name() == "relu_flag" && attr.i() != 0) {
      relu_flag = true;
    }
  }

  op_dest.SetAttr("dtype", dtype);
  op_dest.SetAttr("sqrt_mode", sqrt_mode);
  op_dest.SetAttr("relu_flag", relu_flag);

    // the input format should be NCHW
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (op_desc == nullptr){
    ONNX_PLUGIN_LOGE("Dequant", "Get op desc failed.");
    return FAILED;
  }
  for (size_t i = 0; i < op_desc->GetInputsSize(); i++){
    ge::GeTensorDesc tensor = op_desc->GetInputDesc(i);
    tensor.SetOriginFormat(ge::FORMAT_NCHW);
    tensor.SetFormat(ge::FORMAT_NCHW);
    auto ret_x = op_desc->UpdateInputDesc(i, tensor);
    if (ret_x != ge::GRAPH_SUCCESS){
      ONNX_PLUGIN_LOGE("Dequant", "update input format failed.");
      return FAILED;
    }
  }
  for (size_t i = 0; i < op_desc->GetOutputsSize(); i++){
    ge::GeTensorDesc tensor = op_desc->GetOutputDesc(i);
    tensor.SetOriginFormat(ge::FORMAT_NCHW);
    tensor.SetFormat(ge::FORMAT_NCHW);
    auto ret_y = op_desc->UpdateOutputDesc(i, tensor);
    if (ret_y != ge::GRAPH_SUCCESS){
      ONNX_PLUGIN_LOGE("Dequant", "update output format failed.");
      return FAILED;
    }
  }
  return SUCCESS;
}

// register AscendDequant op info to GE
REGISTER_CUSTOM_OP("AscendDequant")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::AscendDequant",
                   "ai.onnx::9::AscendDequant",
                   "ai.onnx::10::AscendDequant",
                   "ai.onnx::11::AscendDequant",
                   "ai.onnx::12::AscendDequant",
                   "ai.onnx::13::AscendDequant"})
    .ParseParamsFn(ParseParamsAscendDequant)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
