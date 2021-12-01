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

using namespace std;
using namespace ge;
using ge::Operator;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
using OpDesc = std::shared_ptr<ge::OpDesc>;

Status ParseParamsUpsample(const Message *op_src, ge::Operator &op_dest) {
  const NodeProto *node = reinterpret_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    return FAILED;
  }
  op_dest.SetAttr("original_type", "ai.onnx::11::Upsample");
  std::string mode_value = "nearest";
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
        mode_value = attr.s();
    } 
  }
  if (mode_value != "nearest" && mode_value != "linear") {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Mode attr of Upsample only supports nearest and linear, current is %s .", mode_value.c_str());
    return FAILED;
  }
  op_dest.SetAttr("mode", mode_value);
  OpDesc op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  op_desc->AddDynamicInputDesc("x", 2);
  op_desc->AddDynamicOutputDesc("y", 1);
  return SUCCESS;
}

Status ParseOpToGraphUpsample(const ge::Operator& op, Graph& graph) {
  auto data0 = op::Data("data0").set_attr_index(0);
  auto data1 = op::Data("data1").set_attr_index(1);

  std::vector<float> pads_vector(8, 0);
  int64_t len = pads_vector.size();
  const vector<int64_t> dims = {len};
  ge:: Tensor const_value = Vec2Tensor(pads_vector, dims, DT_FLOAT, ge::FORMAT_ND);
  auto const_op = op::Const("const_data").set_attr_value(const_value);
  auto ret_x = ChangeFormatFromOnnx(data0, 0, ge::FORMAT_NCHW, false);
  if (ret_x != ge::GRAPH_SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "update upsample_x format failed.");
    return FAILED;
  }
  std::string mode_value1;
  if (op.GetAttr("mode", mode_value1) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "get value of mode from op failed.");
    return FAILED;
  }
  auto resize_op = op::Resize("resize").set_input_x(data0)
                                       .set_input_roi(const_op)
                                       .set_input_scales(data1)
                                       .set_attr_mode(mode_value1);
  
  ChangeFormatFromOnnx(resize_op, 0, ge::FORMAT_NCHW, true);
  ChangeFormatFromOnnx(resize_op, 0, ge::FORMAT_NCHW, false);

  std::vector<ge::Operator> inputs = {data0, const_op, data1};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
  output_indexs.emplace_back(resize_op, std::vector<size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

// register Upsample op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::Upsample",
                 "ai.onnx::9::Upsample",
                 "ai.onnx::11::Upsample",
                 "ai.onnx::12::Upsample",
                 "ai.onnx::13::Upsample"})
  .ParseParamsFn(ParseParamsUpsample)
  .ParseOpToGraphFn(ParseOpToGraphUpsample)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
