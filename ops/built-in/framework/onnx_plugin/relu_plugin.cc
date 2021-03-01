/**
 * Copyright 2018 Huawei Technologies Co., Ltd
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
 * \file relu_plugin.cpp
 * \brief
 */
#include <string>

#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"

#include "op_log.h"

namespace domi {

Status ParseParamsRelu(const Message* op_origin, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_origin);
  if (nullptr == node) {
    OP_LOGE("Relu", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  return SUCCESS;
}

// register ReLU op info to GE
REGISTER_CUSTOM_OP("Relu")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::11::Relu")
	  .OriginOpType({"ai.onnx::9::Relu",
	                 "ai.onnx::12::Relu",
				           "ai.onnx::13::Relu"})
    .ParseParamsFn(ParseParamsRelu)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
