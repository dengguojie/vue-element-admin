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
#include "operator.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"

namespace domi {
// Replace ge ParseParams function to process graph TransposeD node attrs
Status ParseParamsTransposeD(const Message *op_src, ge::Operator &op_dest) {
  const ge::onnx::NodeProto *node =
      reinterpret_cast<const ge::onnx::NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE("TransposeD", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  // set default value for attribute perm
  std::vector<int32_t> perm_default = {3, 2, 1, 0};
  op_dest.SetAttr("perm", perm_default);

  // update attribute perm if given
  for (auto attr : node->attribute()) {
    if (attr.name() == "perm") {
      std::vector<int32_t> perm_input;

      for (auto axis : attr.ints()) {
        perm_input.push_back(axis);
      }

      op_dest.SetAttr("perm", perm_input);
    }
  }

  return SUCCESS;
}

// register Transpose op info to GE
REGISTER_CUSTOM_OP("TransposeD")
  .FrameworkType(ONNX)
  .OriginOpType("ai.onnx::11::Transpose")
  .ParseParamsFn(ParseParamsTransposeD)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
