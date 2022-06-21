/* Copyright (C) Huawei Technologies Co., Ltd 2022-2022. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this
 * file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http:// www.apache.org/licenses/LICENSE-2.0
 */
#include "onnx_common.h"
#include "array_ops.h"

using namespace ge;

namespace domi {
using NodeProto = ge::onnx::NodeProto;
static const int ONNX_1D_ATTR_LEN = 1;

void SetAllIntListValue(const ge::onnx::AttributeProto &attr, std::vector<int32_t> &int_list) {
  for (auto i = 0; i < attr.ints_size(); ++i) {
    int_list.push_back(attr.ints(i));
  }
}

void SetAttrListValue(const ge::onnx::AttributeProto &attr, std::vector<int32_t> &attr_list) {
  if (attr.ints_size() == ONNX_1D_ATTR_LEN) {
    attr_list.push_back(1);
    attr_list.push_back(1);
    attr_list.push_back(attr.ints(0));
    attr_list.push_back(attr.ints(0));
  }else {
    for (auto i = 0; i < attr.ints_size(); ++i) {
      attr_list.push_back(attr.ints(i));
    }
  }
}

Status ParseParamsDeformableOffsets(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE("DeformableOffsets", "Dynamic cast op_src to NodeProto failed");
    return FAILED;
  }

  std::vector<int> strides = {};
  std::vector<int> pads = {};
  std::vector<int> dilations = {1, 1, 1, 1};
  std::vector<int> ksize = {};
  std::string data_format = "NCHW";
  int deformable_groups = 1;
  bool modulated = true;

  for (const auto& attr : node->attribute()) {
    if (attr.name() == "strides" && attr.type() == ge::onnx::AttributeProto::INTS) {
      SetAttrListValue(attr, strides);
    }else if (attr.name() == "pads" && attr.type() == ge::onnx::AttributeProto::INTS) {
      SetAttrListValue(attr, pads);
    }else if (attr.name() == "dilations" && attr.type() == ge::onnx::AttributeProto::INTS) {
      dilations.clear();
      SetAttrListValue(attr, dilations);
    }else if (attr.name() == "ksize" && attr.type() == ge::onnx::AttributeProto::INTS) {
      SetAllIntListValue(attr, ksize);
    }else if (attr.name() == "data_format" && attr.type() == ge::onnx::AttributeProto::STRING) {
      data_format = attr.s();
    }else if (attr.name() == "deformable_groups" && attr.type() == ge::onnx::AttributeProto::INT) {
      deformable_groups = attr.i();
    }else if (attr.name() == "modulated" && attr.type() == ge::onnx::AttributeProto::INT) {
      modulated = attr.i();
    }
  }
  
  op_dest.SetAttr("strides", strides); 
  op_dest.SetAttr("pads", pads);
  op_dest.SetAttr("dilations", dilations);
  op_dest.SetAttr("ksize", ksize);
  op_dest.SetAttr("data_format", data_format);
  op_dest.SetAttr("deformable_groups", deformable_groups);
  op_dest.SetAttr("modulated", modulated);
 
  if (ChangeFormatFromOnnx(op_dest, 0, ge::FORMAT_NCHW, true) != SUCCESS ||
      ChangeFormatFromOnnx(op_dest, 1, ge::FORMAT_NCHW, true) != SUCCESS ||
      ChangeFormatFromOnnx(op_dest, 0, ge::FORMAT_NCHW, false) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "ChangeFormatFromOnnx failed.");
    return FAILED;
  }
  return SUCCESS;
}

// register DeformableOffsets op info to GE
REGISTER_CUSTOM_OP("DeformableOffsets")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::9::DeformableOffsets", "ai.onnx::10::DeformableOffsets", "ai.onnx::11::DeformableOffsets",
                   "ai.onnx::12::DeformableOffsets", "ai.onnx::13::DeformableOffsets",
                   "ai.onnx::14::DeformableOffsets", "ai.onnx::15::DeformableOffsets"})
    .ParseParamsFn(ParseParamsDeformableOffsets)
    .ImplyType(ImplyType::TVM);
}  // namespace domi