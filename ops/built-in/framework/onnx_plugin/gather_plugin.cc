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

static const int DEFAULT_AXIS = 0;

Status ParseParamsGather(const Message* op_src, ge::Operator& op_dest) {
  OP_LOGI("Gather", "[PLUGIN_GATHER]---------ParseParams Gather start----------");
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (nullptr == node) {
    OP_LOGE("Gather", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int axis_val = 0;
  bool set_axis_flag = false;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
      axis_val = attr.i();
      set_axis_flag = true;
      break;
    }
  }
  if (!set_axis_flag) {
    OP_LOGI("Gather", "onnx Gather op has no axis attr.");
    axis_val = DEFAULT_AXIS;
  }
  op_dest.SetAttr("axis", axis_val);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("GatherV2D")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::11::Gather")
    .OriginOpType({"ai.onnx::9::Gather",
                   "ai.onnx::12::Gather",
                   "ai.onnx::13::Gather"})
    .ParseParamsFn(ParseParamsGather)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
