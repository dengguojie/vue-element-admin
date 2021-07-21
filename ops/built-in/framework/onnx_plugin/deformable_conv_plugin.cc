/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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

using namespace ge;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
Status ParseParamsDeformableConv2D(const Message* op_src, ge::Operator& op_dest) {
  ONNX_PLUGIN_LOGI("DeformableConv2D", "Start into the ParseParamsDeformableConv2D!");
  const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE("DeformableConv2D", "Dynamic cast op_src to NodeProto failed");
    return FAILED;
  }

  std::vector<int> strides = {};
  std::vector<int> pads = {};
  std::vector<int> dilations = {1, 1, 1, 1};
  std::string data_format = "NHWC";
  int deformable_groups = 1;
  int groups = 0;
  bool modulated = true;
  bool set_strides = false;
  bool set_pads = false;

  for (const auto& attr : node->attribute()) {
    if (attr.name() == "strides" && attr.type() == ge::onnx::AttributeProto::INTS) {
      if (attr.ints_size() == 1) {
        strides.push_back(1);
        strides.push_back(1);
        strides.push_back(attr.ints(0));
        strides.push_back(attr.ints(0));
      } else if (attr.ints_size() == 2) {
        strides.push_back(1);
        strides.push_back(1);
        strides.push_back(attr.ints(0));
        strides.push_back(attr.ints(1));
      } else {
        ONNX_PLUGIN_LOGE("DeformableConv2D", "the strides attr shape is wrong.");
        return FAILED;
      }

      set_strides = true;
    }
    if (attr.name() == "pads" && attr.type() == ge::onnx::AttributeProto::INTS) {
      if (attr.ints_size() == 1) {
        pads.push_back(attr.ints(0));
        pads.push_back(attr.ints(0));
        pads.push_back(attr.ints(0));
        pads.push_back(attr.ints(0));
      } else if (attr.ints_size() == 2) {
        pads.push_back(attr.ints(0));
        pads.push_back(attr.ints(1));
        pads.push_back(attr.ints(0));
        pads.push_back(attr.ints(1));
      } else {
        ONNX_PLUGIN_LOGE("DeformableConv2D", "the pads attr shape is wrong.");
        return FAILED;
      }

      set_pads = true;
    }

    if (attr.name() == "dilations" && attr.type() == ge::onnx::AttributeProto::INTS) {
      dilations.clear();
      if (attr.ints_size() == 1) {
        dilations.push_back(1);
        dilations.push_back(1);
        dilations.push_back(attr.ints(0));
        dilations.push_back(attr.ints(0));
      } else if (attr.ints_size() == 2) {
        dilations.push_back(1);
        dilations.push_back(1);
        dilations.push_back(attr.ints(0));
        dilations.push_back(attr.ints(1));
      } else {
        ONNX_PLUGIN_LOGE("DeformableConv2D", "the dilations attr shape is wrong.");
        return FAILED;
      }
    }
    if (attr.name() == "data_format" && attr.type() == ge::onnx::AttributeProto::STRING) {
      data_format = attr.s();
    }
    if (attr.name() == "deformable_groups" && attr.type() == ge::onnx::AttributeProto::INT) {
      deformable_groups = attr.i();
    }
    if (attr.name() == "groups" && attr.type() == ge::onnx::AttributeProto::INT) {
      groups = attr.i();
    }
    if (attr.name() == "modulated" && attr.type() == ge::onnx::AttributeProto::INT) {
      modulated = attr.i();
    }
  }
  if (set_strides) {
    op_dest.SetAttr("strides", strides);
  } else {
    ONNX_PLUGIN_LOGE("DeformableConv2D", "onnx DeformableConv2D op has no strides attr.");
  }
  if (set_pads) {
    op_dest.SetAttr("pads", pads);
  } else {
    ONNX_PLUGIN_LOGE("DeformableConv2D", "onnx DeformableConv2D op has no pads attr.");
  }

  op_dest.SetAttr("dilations", dilations);
  op_dest.SetAttr("groups", groups);
  op_dest.SetAttr("data_format", data_format);
  op_dest.SetAttr("deformable_groups", deformable_groups);
  op_dest.SetAttr("modulated", modulated);

  ge::TensorDesc input_tensorx = op_dest.GetInputDesc("x");
  input_tensorx.SetOriginFormat(ge::FORMAT_NCHW);
  input_tensorx.SetFormat(ge::FORMAT_NCHW);
  auto ret = op_dest.UpdateInputDesc("x", input_tensorx);
  if (ret != ge::GRAPH_SUCCESS) {
    ONNX_PLUGIN_LOGE("DeformableConv2D", "update x format failed");
    return FAILED;
  }

  ge::TensorDesc input_tensor_offsets = op_dest.GetInputDesc("offsets");
  input_tensor_offsets.SetOriginFormat(ge::FORMAT_NCHW);
  input_tensor_offsets.SetFormat(ge::FORMAT_NCHW);
  ret = op_dest.UpdateInputDesc("offsets", input_tensor_offsets);
  if (ret != ge::GRAPH_SUCCESS) {
    ONNX_PLUGIN_LOGE("DeformableConv2D", "update offsets format failed");
    return FAILED;
  }

  ge::TensorDesc input_tensor_filter = op_dest.GetInputDesc("filter");
  input_tensor_filter.SetOriginFormat(ge::FORMAT_NCHW);
  input_tensor_filter.SetFormat(ge::FORMAT_NCHW);
  ret = op_dest.UpdateInputDesc("filter", input_tensor_filter);
  if (ret != ge::GRAPH_SUCCESS) {
    ONNX_PLUGIN_LOGE("DeformableConv2D", "update filter format failed");
    return FAILED;
  }

  ge::TensorDesc output_tensor = op_dest.GetOutputDesc("y");
  output_tensor.SetOriginFormat(ge::FORMAT_NCHW);
  output_tensor.SetFormat(ge::FORMAT_NCHW);
  auto ret_output = op_dest.UpdateOutputDesc("y", output_tensor);
  if (ret_output != ge::GRAPH_SUCCESS) {
    ONNX_PLUGIN_LOGE("DeformableConv2D", "update output format failed");
    return FAILED;
  }
  return SUCCESS;
}

// register DeformableConv2D op info to GE
REGISTER_CUSTOM_OP("DeformableConv2D")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::9::DeformableConv2D", "ai.onnx::10::DeformableConv2D", "ai.onnx::11::DeformableConv2D",
                   "ai.onnx::12::DeformableConv2D", "ai.onnx::13::DeformableConv2D"})
    .ParseParamsFn(ParseParamsDeformableConv2D)
    .ImplyType(ImplyType::TVM);
}  // namespace domi