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


#include <string>
#include <vector>

#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"

#include "op_log.h"

namespace domi {

Status ParseParamsLpNormalization(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("LpNormalization", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int p_attr = 2;
  int axis_attr = -1;
  bool keep_dims_attr = false;
  float epsilon_attr = 1e-12;

  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      axis_attr = attr.i();
    } else if (attr.name() == "p" && attr.type() == ge::onnx::AttributeProto::INT) {
      p_attr = static_cast<int>(attr.i());
      if (p_attr != 1 && p_attr != 2) {
        OP_LOGE("LpNormalization", "Attribute P given wrong value.");
        return FAILED;
      }
    }
  }

  op_dest.SetAttr("axes", axis_attr);
  op_dest.SetAttr("keepdim", keep_dims_attr);
  op_dest.SetAttr("p", p_attr);
  op_dest.SetAttr("epsilon", epsilon_attr);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("LpNorm")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::LpNormalization",
                  "ai.onnx::9::LpNormalization",
                  "ai.onnx::10::LpNormalization",
                  "ai.onnx::11::LpNormalization",
                  "ai.onnx::12::LpNormalization",
                  "ai.onnx::13::LpNormalization"})
    .ParseParamsFn(ParseParamsLpNormalization)
    .ImplyType(ImplyType::TVM);
}