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
#include "array_ops.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;
using DescPtr = std::shared_ptr<ge::OpDesc>;
Status ParseParamsSum(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  uint32_t input_num = node->input_size();
  if (input_num < 1) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "input_num must be 1");
    return FAILED;
  }
  DescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  op_desc->AddDynamicInputDesc("x", input_num);
  op_dest.SetAttr("N", input_num);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("AccumulateNV2")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::Sum",
                 "ai.onnx::9::Sum",
                 "ai.onnx::10::Sum",
                 "ai.onnx::11::Sum",
                 "ai.onnx::12::Sum",
                 "ai.onnx::13::Sum"})
  .ParseParamsFn(ParseParamsSum)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
