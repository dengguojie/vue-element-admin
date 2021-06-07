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

#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"

namespace domi {
Status ParseParamsResize(const Message *op_src, ge::Operator &op_dst) {
  const ge::onnx::NodeProto *node =
      reinterpret_cast<const ge::onnx::NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE("Resize", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  std::string coordinate_transformation_mode_value;
  float cubic_coeff;
  int exclude_outside_value;
  float extrapolations_value;
  std::string mode_value;
  std::string nearest_mode_value;
  for (auto attr : node->attribute()) {
    if (attr.name() == "coordinate_transformation_mode" &&
        attr.type() == ge::onnx::AttributeProto::STRING) {
      coordinate_transformation_mode_value = attr.s();
      op_dst.SetAttr("coordinate_transformation_mode",
                     coordinate_transformation_mode_value);
    } else if (attr.name() == "cubic_coeff_a" &&
               attr.type() == ge::onnx::AttributeProto::FLOAT) {
      cubic_coeff = attr.f();
      op_dst.SetAttr("cubic_coeff_a", cubic_coeff);
    } else if (attr.name() == "exclude_outside" &&
               attr.type() == ge::onnx::AttributeProto::INT) {
      exclude_outside_value = attr.i();
      op_dst.SetAttr("exclude_outside", exclude_outside_value);
    } else if (attr.name() == "extrapolation_value" &&
               attr.type() == ge::onnx::AttributeProto::FLOAT) {
      extrapolations_value = attr.f();
      op_dst.SetAttr("extrapolation_value", extrapolations_value);
    } else if (attr.name() == "mode" &&
               attr.type() == ge::onnx::AttributeProto::STRING) {
      mode_value = attr.s();
      op_dst.SetAttr("mode", mode_value);
    } else if (attr.name() == "nearest_mode" &&
               attr.type() == ge::onnx::AttributeProto::STRING) {
      nearest_mode_value = attr.s();
      op_dst.SetAttr("nearest_mode", nearest_mode_value);
    }
  }

  // the input format should be NCHW
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dst);
  if (op_desc == nullptr){
    OP_LOGE("Resize", "Get op desc failed.");
    return FAILED;
  }
  for (size_t i = 0; i < op_desc->GetInputsSize(); i++){
    ge::GeTensorDesc tensor = op_desc->GetInputDesc(i);
    tensor.SetOriginFormat(ge::FORMAT_NCHW);
    tensor.SetFormat(ge::FORMAT_NCHW);
    auto ret_x = op_desc->UpdateInputDesc(i, tensor);
    if (ret_x != ge::GRAPH_SUCCESS){
      OP_LOGE("Resize", "update input format failed.");
      return FAILED;
    }
  }
  // the output format should be NCHW
  for (size_t i = 0; i < op_desc->GetOutputsSize(); i++){
    ge::GeTensorDesc tensor = op_desc->GetOutputDesc(i);
    tensor.SetOriginFormat(ge::FORMAT_NCHW);
    tensor.SetFormat(ge::FORMAT_NCHW);
    auto ret_y = op_desc->UpdateOutputDesc(i, tensor);
    if (ret_y != ge::GRAPH_SUCCESS){
      OP_LOGE("Resize", "update output format failed.");
      return FAILED;
    }
  }
  return SUCCESS;
}

REGISTER_CUSTOM_OP("Resize")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::10::Resize",
                 "ai.onnx::11::Resize",
                 "ai.onnx::12::Resize",
                 "ai.onnx::13::Resize"})
  .ParseParamsFn(ParseParamsResize)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
