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
 * \file reshape_plugin.cpp
 * \brief
 */
#include <string>
#include <vector>

#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"

#include "op_log.h"

namespace domi {

Status ParseParamsReshape(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  int32_t allow_zero = 0;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "allowzero" && attr.type() == ge::onnx::AttributeProto::INT) {
      allow_zero = attr.i();
    }
  }
  op_dest.SetAttr("allowzero", allow_zero);
  return SUCCESS;
}

// register Add op info to GE
REGISTER_CUSTOM_OP("Reshape")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::9::Reshape",
                   "ai.onnx::10::Reshape",
                   "ai.onnx::11::Reshape",
                   "ai.onnx::12::Reshape",
                   "ai.onnx::13::Reshape"})
    .ParseParamsFn(ParseParamsReshape)
    .ImplyType(ImplyType::GELOCAL);
}  // namespace domi
