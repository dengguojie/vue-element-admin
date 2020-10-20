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
 * \file shape_plugin.cpp
 * \brief
 */
#include <string>
#include <vector>

#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"

#include "op_log.h"

namespace domi {

Status ParseParamsShape(const Message* op_src, ge::Operator& op_dest) {
  op_dest.SetAttr("dtype", static_cast<uint32_t>(ge::DT_INT64));
  return SUCCESS;
}

// register Add op info to GE
REGISTER_CUSTOM_OP("Shape")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::11::Shape")
    .ParseParamsFn(ParseParamsShape)
    .ImplyType(ImplyType::GELOCAL);
}  // namespace domi
