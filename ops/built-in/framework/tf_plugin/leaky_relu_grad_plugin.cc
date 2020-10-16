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
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"

namespace domi {

Status AutoMappingFnLeakyReluGrad(const google::protobuf::Message* op_src, ge::Operator& op)
{
    Status ret = AutoMappingFn(op_src, op);
    if (ret != SUCCESS) {
        OP_LOGE("LeakyReluGrad", "tensorflow plugin parser failed. auto mapping failed.");
        return FAILED;
    }
    float alpha;
    if (op.GetAttr("alpha", alpha) != ge::GRAPH_SUCCESS) {
        OP_LOGI("LeakyReluGrad", "GetAttr alpha failed");
        return FAILED;
    }
    op.SetAttr("negative_slope", alpha);
    OP_LOGI("LeakyReluGrad", "op[LeakyReluGrad] tensorflow plugin parser success.");
    return SUCCESS;
}

REGISTER_CUSTOM_OP("LeakyReluGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LeakyReluGrad")
    .ParseParamsFn(AutoMappingFnLeakyReluGrad)
    .ImplyType(ImplyType::TVM);
} // namespace domi
