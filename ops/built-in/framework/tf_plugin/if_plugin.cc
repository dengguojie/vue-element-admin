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
Status AutoMappingFnIf(const google::protobuf::Message *op_src, ge::Operator &op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("input", "Tin");
  value["out"] = pair<string, string>("output", "Tout");
  if (AutoMappingFnDynamic(op_src, op, value) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status ParseSubgraphPostFnIf(const std::string &subgraph_name, const ge::Graph &graph) {
  return AutoMappingSubgraphIndex(graph,
                                  [](int data_index) { return data_index + 1; },
                                  [](int retval_index) { return retval_index; });
}

REGISTER_CUSTOM_OP("If")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("If")
    .ParseParamsFn(AutoMappingFnIf)
    .ParseSubgraphPostFn(ParseSubgraphPostFnIf)
    .ImplyType(ImplyType::GELOCAL);

REGISTER_CUSTOM_OP("_If")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("_If")
    .ParseParamsFn(AutoMappingFnIf)
    .ParseSubgraphPostFn(ParseSubgraphPostFnIf)
    .ImplyType(ImplyType::GELOCAL);

REGISTER_CUSTOM_OP("StatelessIf")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StatelessIf")
    .ParseParamsFn(AutoMappingFnIf)
    .ParseSubgraphPostFn(ParseSubgraphPostFnIf)
    .ImplyType(ImplyType::GELOCAL);
}  // namespace domi
