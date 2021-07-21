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

using namespace ge;
using ge::Operator;

namespace domi {
Status parse_params_pad_v11(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
  if (nullptr == node) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  for (const auto& attr : node->attribute()) {
    if (attr.name() == "mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
      std::string mode_value = attr.s();
      if (mode_value == "reflect" || mode_value == "edge") {
        ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Mode attr of Pad only supports constant, current is %s .", mode_value.c_str());
        return FAILED;
      }
    }
  }
  op_dest.SetAttr("paddings_contiguous", false);

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
  opDesc->AddDynamicInputDesc("x", 3);
  opDesc->AddDynamicOutputDesc("output", 1);
  // 2.set original_type
  ge::AttrUtils::SetStr(opDesc, "original_type", "ai.onnx::9::Pad");

  // attr max and min default value
  std::vector<int32_t> v_pads;
  bool set_pads_flag = false;
  float value = 0.0;

  for (const auto& attr : node->attribute()) {
    if (attr.name() == "mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
      std::string mode_value = attr.s();
      if (mode_value == "reflect" || mode_value == "edge") {
        ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Mode attr of Pad only supports constant, current is %s .", mode_value.c_str());
        return FAILED;
      }
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
  int num = v_pads.size();
  ge::TensorDesc tensorDesc1;
  std::vector<int64_t> pads_dims = {num};
  ge::Shape pads_shape(pads_dims);
  tensorDesc1.SetShape(pads_shape);
  tensorDesc1.SetDataType(DT_INT32);

  ge::TensorDesc tensorDesc2;
  std::vector<int64_t> value_dims = {1};
  ge::Shape value_shape(value_dims);
  tensorDesc2.SetShape(value_shape);
  tensorDesc2.SetDataType(DT_FLOAT);

  ge::Tensor tensor1(tensorDesc1, reinterpret_cast<uint8_t*>(v_pads.data()), v_pads.size() * sizeof(DT_INT32));
  op_dest.SetAttr("paddings", tensor1);
  ge::Tensor tensor2(tensorDesc2, reinterpret_cast<uint8_t*>(&value), sizeof(DT_FLOAT));
  op_dest.SetAttr("constant_values", tensor2);

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

  auto data1 = op::Const("data1").set_attr_value(value1);
  auto data2 = op::Const("data2").set_attr_value(value2);
  auto pad_v3 = op::PadV3().set_input_x(data0).set_input_paddings(data1).set_input_constant_values(data2);

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
