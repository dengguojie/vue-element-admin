/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use
 * this file except in compliance with the License.
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
namespace domi
{
// Status AutoMappingFnHorovodBroadcast(const google::protobuf::Message* op_src, ge::Operator& op)
// {
//     if (op_src == nullptr) {
//         OP_LOGE("HorovodBroadcast", "op[HorovodBroadcast] tensorflow plugin parser[AutoMapping] failed. op_src is nullptr.");
//         return FAILED;
//     }
//     map<string, pair<string,string>>value;
//     value["in"] = pair<string,string>("x","T");
//     value["out"] = pair<string,string>("y","T");
//     if (AutoMappingFnDynamic(op_src, op, value) == SUCCESS) {
//         OP_LOGI("HorovodBroadcast", "op[HorovodBroadcast] tensorflow plugin parser[AutoMapping] success.");
//         return SUCCESS;
//     } else {
//         OP_LOGE("HorovodBroadcast", "op[HorovodBroadcast] tensorflow plugin parser[AutoMapping] failed.");
//         return FAILED;
//     }
// }

// register HorovodBroadcast op to GE
REGISTER_CUSTOM_OP("HorovodBroadcast")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("HorovodBroadcast")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::HCCL);
}  // namespace domi