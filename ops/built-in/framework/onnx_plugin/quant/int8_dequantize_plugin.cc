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
#include "../onnx_common.h"
#include "array_ops.h"
#include "quantize_ops.h"
#include "nn_pooling_ops.h"

using namespace std;
using namespace ge;
using ge::Operator;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
using OpDesc = std::shared_ptr<ge::OpDesc>;

Status ParseParamsInt8Dequantize(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (nullptr == node) {
    ONNX_PLUGIN_LOGE("ParseParamsInt8Dequantize", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  float scale = 0.f;
  float offset = 0.f;
  for (auto& attr : node->attribute()) {
    if (attr.name().find("scale") != std::string::npos) {
      scale = attr.f();
    }
    if (attr.name().find("zero_point") != std::string::npos) {
      offset = static_cast<float>(attr.i());
    }
  }
  OpDesc op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  op_desc->AddDynamicInputDesc("x", 1);
  op_desc->AddDynamicOutputDesc("y", 1);
  op_dest.SetAttr("original_type", "ai.onnx::11::Int8Dequantize");
  op_dest.SetAttr("scale", scale);
  op_dest.SetAttr("offset", offset);
  return SUCCESS;
}

Status ParseOpToGraphInt8Dequantize(const ge::Operator& op, Graph& graph) {
  float scale = 0.f;
  float offset = 0.f;
  ge::Operator::OpListInt maxpool_ksize = {1, 1, 1, 1};
  ge::Operator::OpListInt maxpool_strides = {1, 1, 1, 1};
  op.GetAttr("scale", scale);
  op.GetAttr("offset", offset);
  auto data1 = op::Data("data1").set_attr_index(0);
  auto ascend_op = op::AscendAntiQuant("Int8DeqAscendAntiquant")
                       .set_input_x(data1)
                       .set_attr_scale(scale)
                       .set_attr_offset(offset)
                       .set_attr_dtype(DT_FLOAT16);
  auto maxpool = op::MaxPool("Int8DeqAscendMaxpool")
                     .set_input_x(ascend_op)
                     .set_attr_ksize(maxpool_ksize)
                     .set_attr_strides(maxpool_strides)
                     .set_attr_padding("VALID")
                     .set_attr_data_format("NCHW");
  std::vector<ge::Operator> inputs = {data1};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;

  output_indexs.emplace_back(maxpool, std::vector<size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Int8Dequantize", "ai.onnx::9::Int8Dequantize", "ai.onnx::10::Int8Dequantize",
                   "ai.onnx::11::Int8Dequantize", "ai.onnx::12::Int8Dequantize", "ai.onnx::13::Int8Dequantize"})
    .ParseParamsFn(ParseParamsInt8Dequantize)
    .ParseOpToGraphFn(ParseOpToGraphInt8Dequantize)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
