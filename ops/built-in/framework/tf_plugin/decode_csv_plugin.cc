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
 * \file decode_csv_plugin.cpp
 * \brief
 */

#include "register/register.h"
#include "graph/operator.h"

namespace domi {
Status DecodeCSVMappingFn(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("record_defaults", "OUT_TYPE");
  AutoMappingFnDynamic(op_src, op, value);
  value["out"] = pair<string, string>("output", "OUT_TYPE");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("DecodeCSV")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DecodeCSV")
    .ParseParamsFn(DecodeCSVMappingFn)
    .ImplyType(ImplyType::AI_CPU);
}  // namespace domi
