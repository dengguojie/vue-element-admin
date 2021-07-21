/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file affine_grid.cc
 * \brief
 */
#include "onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

Status ParseParamsAffineGrid(const Message *op_src, ge::Operator &op_dest) {
  const NodeProto *node = dynamic_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  bool align_corners = false;
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "align_corners" && attr.i() != 0) {
        align_corners = true;
        break;
    }
  }
  op_dest.SetAttr("align_corners", align_corners);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("AffineGrid")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::AffineGrid",
                 "ai.onnx::9::AffineGrid",
                 "ai.onnx::10::AffineGrid",
                 "ai.onnx::11::AffineGrid",
                 "ai.onnx::12::AffineGrid",
                 "ai.onnx::13::AffineGrid"})
  .ParseParamsFn(ParseParamsAffineGrid)
  .ImplyType(ImplyType::TVM);
} // namespace domi
