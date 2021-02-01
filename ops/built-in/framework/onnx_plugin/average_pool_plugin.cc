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
#include <string>
#include <vector>

#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;
struct AvgPoolAttr {
  std::string auto_pad = "NOTSET";
  int64_t ceil_mode = 0;
  int64_t count_include_pad = 0;
  std::vector<int> kernel_shape = {1, 1, 1, 1};
  std::vector<int> pads = {0, 0, 0, 0};
  std::vector<int> strides = {1, 1, 1, 1};
};

Status UpdateAttrFromOnnx(const NodeProto* node, AvgPoolAttr& node_attr) {
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "auto_pad" && attr.type() == ge::onnx::AttributeProto::STRING) {
      node_attr.auto_pad = attr.s();
    }

    if (attr.name() == "ceil_mode" && attr.type() == ge::onnx::AttributeProto::INT) {
      node_attr.ceil_mode = attr.i();
    }

    if (attr.name() == "count_include_pad" && attr.type() == ge::onnx::AttributeProto::INT) {
      node_attr.count_include_pad = attr.i();
    }

    if (attr.name() == "kernel_shape" && attr.type() == ge::onnx::AttributeProto::INTS) {
      if (attr.ints().size() != 2) {
        OP_LOGE("AveragePool", "Only support kernel_shape.size() = 2");
        return FAILED;
      }
      node_attr.kernel_shape[2] = attr.ints(0);
      node_attr.kernel_shape[3] = attr.ints(1);
    }
    if (attr.name() == "strides" && attr.type() == ge::onnx::AttributeProto::INTS) {
      if (attr.ints().size() != 2) {
        OP_LOGE("AveragePool", "Only support strides.size() = 2");
        return FAILED;
      }
      node_attr.strides[2] = attr.ints(0);
      node_attr.strides[3] = attr.ints(1);
    }
    if (attr.name() == "pads" && attr.type() == ge::onnx::AttributeProto::INTS) {
      if (attr.ints().size() != 4) {
        OP_LOGE("AveragePool", "Only support pads.size() = 4");
        return FAILED;
      }
      node_attr.pads.resize(attr.ints().size());
      node_attr.pads[0] = attr.ints(0);
      node_attr.pads[1] = attr.ints(2);
      node_attr.pads[2] = attr.ints(1);
      node_attr.pads[3] = attr.ints(3);
    }
  }
  return SUCCESS;
}

Status ParseParamsAveragePool(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("AveragePool", "reinterpret_cast op_src to NodeProto failed.");
    return FAILED;
  }

  AvgPoolAttr node_attr;
  if (UpdateAttrFromOnnx(node, node_attr) != SUCCESS) {
    return FAILED;
  }

  std::map<string, string> padding_mode = {
      {"NOTSET", "CALCULATED"}, {"SAME_UPPER", "SAME"}, {"SAME_LOWER ", "SAME"}, {"VALID", "VALID"}};
  // set attr for AvgPoolV2
  op_dest.SetAttr("ksize", node_attr.kernel_shape);
  op_dest.SetAttr("strides", node_attr.strides);
  op_dest.SetAttr("padding_mode", padding_mode[node_attr.auto_pad]);
  op_dest.SetAttr("pads", node_attr.pads);
  op_dest.SetAttr("ceil_mode", (bool)node_attr.ceil_mode);
  op_dest.SetAttr("exclusive", !(bool)node_attr.count_include_pad);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("AvgPoolV2")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::11::AveragePool")
    .ParseParamsFn(ParseParamsAveragePool)
    .ImplyType(ImplyType::TVM);
}  //  namespace domi
