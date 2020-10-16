/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "register/register.h"
#include "op_log.h"

using namespace ge;
namespace domi {
Status AutoMappingFnArgMin(const google::protobuf::Message* op_src, ge::Operator& op)
{
    Status ret = AutoMappingFn(op_src, op);
    if (ret != SUCCESS) {
        OP_LOGE("ArgMin", "tensorflow plugin parser failed. auto mapping failed.");
        return FAILED;
    }
    ge::DataType dataType;
    if (op.GetAttr("output_type", dataType) != GRAPH_SUCCESS) {
        OP_LOGI("ArgMin", "GetAttr DstT failed");
        return FAILED;
    }
    op.SetAttr("dtype", dataType);
    OP_LOGI("ArgMin", "op[ArgMin] tensorflow plugin parser[AutoMapping] success.");
    return SUCCESS;
}

REGISTER_CUSTOM_OP("ArgMin")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ArgMin")
    .ParseParamsFn(AutoMappingFnArgMin)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
