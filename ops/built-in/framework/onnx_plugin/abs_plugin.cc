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
 * \file abs_plugin.cpp
 * \brief
 */
#include <string>
#include <vector>

#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"

#include "op_log.h"

namespace domi {

Status ParseParamsAbs(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("Abs", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  return SUCCESS;
}

// register Abs op info to GE
REGISTER_CUSTOM_OP("Abs")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::11::Abs")
    .OriginOpType("ai.onnx::12::Abs")
    .ParseParamsFn(ParseParamsAbs)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
