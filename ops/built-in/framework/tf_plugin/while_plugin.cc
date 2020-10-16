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
Status AutoMappingFnWhile(const google::protobuf::Message *op_src, ge::Operator &op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("input", "T");
  value["out"] = pair<string, string>("output", "T");
  if (AutoMappingFnDynamic(op_src, op, value) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status ParseSubgraphPostFnWhileInput(int data_index, int &parent_index) {
  parent_index = data_index;
  return SUCCESS;
}

Status ParseSubgraphPostFnWhileOutput(const std::string &subgraph_name, int retval_index, int &parent_index) {
  if (subgraph_name == "cond") {
    parent_index = -1;
  } else {
    parent_index = retval_index;
  }
  return SUCCESS;
}

Status ParseSubgraphPostFnWhile(const std::string &subgraph_name, const ge::Graph &graph) {
  return AutoMappingSubgraphIndex(graph, ParseSubgraphPostFnWhileInput,
                                  [&](int retval_index, int &parent_index) -> Status {
                                    return ParseSubgraphPostFnWhileOutput(subgraph_name, retval_index, parent_index);
                                  });
}

REGISTER_CUSTOM_OP("While")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("While")
    .ParseParamsFn(AutoMappingFnWhile)
    .ParseSubgraphPostFn(ParseSubgraphPostFnWhile)
    .ImplyType(ImplyType::GELOCAL);

REGISTER_CUSTOM_OP("_While")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("_While")
    .ParseParamsFn(AutoMappingFnWhile)
    .ParseSubgraphPostFn(ParseSubgraphPostFnWhile)
    .ImplyType(ImplyType::GELOCAL);

REGISTER_CUSTOM_OP("StatelessWhile")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StatelessWhile")
    .ParseParamsFn(AutoMappingFnWhile)
    .ParseSubgraphPostFn(ParseSubgraphPostFnWhile)
    .ImplyType(ImplyType::GELOCAL);
}  // namespace domi
