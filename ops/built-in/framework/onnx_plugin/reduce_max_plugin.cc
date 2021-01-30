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
Status ParseParamsReduceMax(const Message *op_src, ge::Operator &op_dest) {
  const NodeProto *node = reinterpret_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE("ReduceMax", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  std::vector<int> v_axis;
  bool keep_dims = true;
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "axes" &&
        attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        v_axis.push_back(attr.ints(i));
      }
    } else if (attr.name() == "keepdims" &&
               attr.type() == ge::onnx::AttributeProto::INT) {
      if (attr.i() != 1) {
        keep_dims = false;
      }
    }
  }
  if (v_axis.size() == 0){
	  v_axis =	{};
  }
  op_dest.SetAttr("axes", v_axis);
  op_dest.SetAttr("keep_dims", keep_dims);
  return SUCCESS;
}

// register ReduceMax op info to GE
REGISTER_CUSTOM_OP("ReduceMaxD")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::11::ReduceMax")
  .ParseParamsFn(ParseParamsReduceMax)
  .ImplyType(ImplyType::TVM);
}  // namespace domi