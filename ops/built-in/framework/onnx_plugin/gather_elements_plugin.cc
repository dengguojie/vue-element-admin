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
#include <string>

#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"

#include "op_log.h"

namespace domi {

Status ParseParamsGatherElements(const Message* op_src, ge::Operator& op_dest) {
  OP_LOGI("GatherElements", "[PLUGIN_GATHER]---------ParseParams Gather start----------");
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (nullptr == node) {
    OP_LOGE("GatherElements", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int dim_value = 0;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      dim_value = attr.i();
    }
  }
  op_dest.SetAttr("dim", dim_value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("GatherElements")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::11::GatherElements",
                   "ai.onnx::9::GatherElements",
                   "ai.onnx::12::GatherElements",
                   "ai.onnx::13::GatherElements"})
    .ParseParamsFn(ParseParamsGatherElements)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
