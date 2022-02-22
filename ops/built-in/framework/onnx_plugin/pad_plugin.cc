/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
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
#include "pad_ops.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
Status parse_params_pad_v11(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
  if (nullptr == node) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  std::string mode_value = "constant";
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
      mode_value = attr.s();
    }
  }
  op_dest.SetAttr("paddings_contiguous", false);
  op_dest.SetAttr("mode", mode_value);
  return SUCCESS;
}

Status parse_params_pad_v9(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  // 1.add dynamic input and out
  opDesc->AddDynamicInputDesc("x", 1);
  opDesc->AddDynamicOutputDesc("output", 1);
  // 2.set original_type
  ge::AttrUtils::SetStr(opDesc, "original_type", "ai.onnx::9::Pad");

  // attr max and min default value
  std::vector<int32_t> v_pads;
  bool set_pads_flag = false;
  float value = 0.0;
  std::string mode_value = "constant";
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
      mode_value = attr.s();
    } else if (attr.name() == "pads") {
      set_pads_flag = true;
      unsigned int len = attr.ints_size();
      if (len & 1) {
        ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "the length of pads must be even, such as [x1_begin, x2_begin...x1_end, x2_end,...]");
        return FAILED;
      }
      for (unsigned int i = 0; i < len / 2; i++) {
        v_pads.push_back(static_cast<int32_t>(attr.ints(i)));
        v_pads.push_back(static_cast<int32_t>(attr.ints(i + len / 2)));
      }
    } else if (attr.name() == "value" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      value = attr.f();
    }
  }

  if (!set_pads_flag) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  std::vector<int64_t> dims = {(int64_t)v_pads.size()};
  ge::Tensor tensor1 = Vec2Tensor(v_pads, dims, ge::DT_INT32);
  op_dest.SetAttr("paddings", tensor1);
  std::vector<int64_t> dims_1 = {1};
  ge::Tensor tensor2 = Scalar2Tensor(value, dims_1, ge::DT_FLOAT);
  op_dest.SetAttr("constant_values", tensor2);
  op_dest.SetAttr("mode", mode_value);
  return SUCCESS;
}

static Status ParseOpToGraphPad(const Operator& op, Graph& graph) {
  auto data0 = op::Data("data0").set_attr_index(0);
  ge::Tensor value1;
  if (op.GetAttr("paddings", value1) != SUCCESS) {
    return FAILED;
  }
  ge::Tensor value2;
  if (op.GetAttr("constant_values", value2) != SUCCESS) {
    return FAILED;
  }

  std::string mode = "constant";
  op.GetAttr("mode", mode);
  auto data1 = op::Const("data1").set_attr_value(value1);
  auto data2 = op::Const("data2").set_attr_value(value2);
  auto pad_v3 = op::PadV3().set_input_x(data0)
                           .set_input_paddings(data1)
                           .set_input_constant_values(data2)
                           .set_attr_mode(mode);

  std::vector<Operator> inputs{data0};
  std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
  output_indexs.emplace_back(pad_v3, vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Pad",
                   "ai.onnx::9::Pad",
                   "ai.onnx::10::Pad"})
    .ParseParamsFn(parse_params_pad_v9)
    .ParseOpToGraphFn(ParseOpToGraphPad)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("PadV3")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::11::Pad",
                   "ai.onnx::12::Pad",
                   "ai.onnx::13::Pad"})
    .ParseParamsFn(parse_params_pad_v11)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
