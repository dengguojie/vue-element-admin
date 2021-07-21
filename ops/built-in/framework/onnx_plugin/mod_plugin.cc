/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file gather_plugin.cpp
 * \brief
 */
#include "onnx_common.h"

namespace domi {

Status ParseParamsMod(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (nullptr == node) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int fmod = 0;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "fmod") {
      fmod = attr.i();
      ONNX_PLUGIN_LOGW(op_dest.GetName().c_str(), "Current optype not surpport fmod, please ignore");
    }
  }
  op_dest.SetAttr("fmod", fmod);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("Mod")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::10::Mod",
                   "ai.onnx::11::Mod",
                   "ai.onnx::12::Mod",
                   "ai.onnx::13::Mod"})
    .ParseParamsFn(ParseParamsMod)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
