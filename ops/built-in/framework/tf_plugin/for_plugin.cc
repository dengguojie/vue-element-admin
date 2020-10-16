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
#include "graph/ge_attr_value.h"

namespace domi {
Status AutoMappingFnFor(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("input", "T");
  value["out"] = pair<string, string>("output", "T");
  if (AutoMappingFnDynamic(op_src, op, value) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status ParseSubgraphPostFnFor(const std::string &subgraph_name, const ge::Graph &graph) {
  return AutoMappingSubgraphIndex(graph,
                                  [](int data_index) { return (data_index == 0) ? 0 : data_index + 2; },
                                  [](int retval_index) { return retval_index; });
}
REGISTER_CUSTOM_OP("For")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("For")
    .ParseParamsFn(AutoMappingFnFor)
    .ParseSubgraphPostFn(ParseSubgraphPostFnFor)
    .ImplyType(ImplyType::GELOCAL);
}  // namespace domi
