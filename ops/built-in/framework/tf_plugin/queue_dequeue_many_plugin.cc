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

using namespace ge;
namespace domi {
Status AutoMappingFnDequeueMany(const google::protobuf::Message* op_src,
                                ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["out"] = pair<string, string>("components", "component_types");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

// register QueueDequeueMany op to GE
REGISTER_CUSTOM_OP("QueueDequeueMany")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("QueueDequeueManyV2")
    .ParseParamsFn(AutoMappingFnDequeueMany)
    .ImplyType(ImplyType::AI_CPU);
}  // namespace domi
