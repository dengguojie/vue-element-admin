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

using namespace std;
using namespace ge;
using ge::Operator;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
using OpDesc = std::shared_ptr<ge::OpDesc>;

Status ParseParamsInt8GivenIntTensorFill(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (nullptr == node) {
    ONNX_PLUGIN_LOGE("ParseParamsInt8GivenIntTensorFill", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  std::vector<int32_t> vals;
  std::vector<int64_t> dims;
  for (auto &attr : node->attribute()) {
    if (attr.name() == "values") {
      int num = attr.ints_size();
      for (int i = 0; i < num; ++i) {
        vals.push_back(attr.ints(i));
      }
    } else if(attr.name() == "shape") {
      int num = attr.ints_size();
      for (int i = 0; i < num; ++i) {
        dims.push_back(attr.ints(i));
      }
    }
  }
  
  if (vals.empty()) {
    ONNX_PLUGIN_LOGE("ParseParamsInt8GivenIntTensorFill", "Must have attr values");
    return FAILED;
  }

  if (dims.empty()) {
    dims.push_back((int64_t)vals.size());
  }

  ge::Tensor value_tensor = Vec2Tensor(vals, dims, ge::DT_INT32);
  op_dest.SetAttr("value", value_tensor);
  return SUCCESS;
}


REGISTER_CUSTOM_OP("Constant")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Int8GivenIntTensorFill",
                   "ai.onnx::9::Int8GivenIntTensorFill",
                   "ai.onnx::10::Int8GivenIntTensorFill",
                   "ai.onnx::11::Int8GivenIntTensorFill",
                   "ai.onnx::12::Int8GivenIntTensorFill",
                   "ai.onnx::13::Int8GivenIntTensorFill"})
    .ParseParamsFn(ParseParamsInt8GivenIntTensorFill)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
