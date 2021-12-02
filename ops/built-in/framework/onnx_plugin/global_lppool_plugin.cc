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

#include "onnx_common.h"
#include "array_ops.h"

namespace domi {

Status ParseParamsGlobalLpPool(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node =
      dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  
  float p = 2;
  for (auto attr : node->attribute()) {
    p = static_cast<float>(attr.i());
  }
  op_dest.SetAttr("p", p);
  return SUCCESS;
}

// register Gemm op info to GE
REGISTER_CUSTOM_OP("GlobalLpPool")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::GlobalLpPool",
                   "ai.onnx::9::GlobalLpPool",
                   "ai.onnx::10::GlobalLpPool",
                   "ai.onnx::11::GlobalLpPool",
                   "ai.onnx::12::GlobalLpPool",
                   "ai.onnx::13::GlobalLpPool"})
    .ParseParamsFn(ParseParamsGlobalLpPool)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
