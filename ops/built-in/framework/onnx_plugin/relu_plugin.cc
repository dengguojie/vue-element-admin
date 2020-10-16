/* Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include <string>

namespace domi
{

Status ParseParamsRelu(const Message* op_origin, ge::Operator& op_dest)
{
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
    .ParseParamsFn(ParseParamsRelu)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
