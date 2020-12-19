/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Create: 2020-08-27
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

using domi::ONNX;

namespace domi {
Status ParseParamsSqueeze(const Message *op_src, ge::Operator &op_dest) {
  OP_LOGI("Squeeze",
          "[PLUGIN_CONCAT]---------ParseParams Squeeze start----------");
  const ge::onnx::NodeProto *node =
      reinterpret_cast<const ge::onnx::NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE("Squeeze", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  bool has_axes = false;
  bool has_dims = false;

  std::vector<int64_t> axes;
  std::string attr_name = "axes";
  for (auto it : node->attribute()) {
    if (it.name() != attr_name) {
      continue;
    }

    has_axes = true;
    int dimsSize = it.ints_size();
    if (dimsSize == 0) {
      break;
    }
    has_dims = true;

    for (auto dim : it.ints()) {
      axes.push_back(dim);
      OP_LOGI("Squeeze", "Dim: %ld", dim);
    }
  }

  if (!has_axes || !has_dims) {
    OP_LOGE("Squeeze", "axes is invalid!");
    return PARAM_INVALID;
  }

  op_dest.SetAttr("axis", axes);
  return SUCCESS;
}

// register Add op info to GE
REGISTER_CUSTOM_OP("Squeeze")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::11::Squeeze")
  .ParseParamsFn(ParseParamsSqueeze)
  .ImplyType(ImplyType::GELOCAL);
}  // namespace domi
