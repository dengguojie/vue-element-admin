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
#include "framework/omg/omg_types.h"
#include "operator.h"


using namespace ge;
namespace domi
{

// parallel_concat is the type name of the operator in the OM model.
// It can be specified randomly and cannot be the same as an existing type name.
// It is case sensitive.

Status AutoMappingFnParallelConcat(const google::protobuf::Message* op_src,
                                   ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("values", "N");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("ParallelConcat")
    .FrameworkType(TENSORFLOW)  // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("ParallelConcat")  // parallel_concat indicates the type name of the operator in the tensorflow framework.
    .ParseParamsFn(AutoMappingFnParallelConcat)
    .ImplyType(ImplyType::TVM); // Implementation type. Enumerated type, The options are as follows: TVM, AI_CPU.
}  // namespace domi
