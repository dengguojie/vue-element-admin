/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
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
Status ParseParamsReduceMin(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("ReduceMin", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  std::vector<int> v_axis;
  bool set_axes_flag = false;
  bool keep_dims = true;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axes" &&
        attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        v_axis.push_back(attr.ints(i));
      }
      set_axes_flag = true;
    } else if (attr.name() == "keepdims" &&
               attr.type() == ge::onnx::AttributeProto::INT) {
      if (attr.i() != 1) {
        keep_dims = false;
      }
    }
  }
  if (set_axes_flag) {
    op_dest.SetAttr("axes", v_axis);
  } else {
    OP_LOGI("ReduceMin", "onnx ReduceMin op has no axes attr, use default.");
  }
  op_dest.SetAttr("keepDims", keep_dims);

  return SUCCESS;
}

// register ReduceMin op info to GE
REGISTER_CUSTOM_OP("ReduceMinD")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::11::ReduceMin")
  .ParseParamsFn(ParseParamsReduceMin)
  .ImplyType(ImplyType::TVM);
}  // namespace domi