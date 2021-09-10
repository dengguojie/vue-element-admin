/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
 * \file while_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "graph/ge_attr_value.h"

namespace domi {
Status AutoMappingFnWhile(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("input", "T");
  value["out"] = pair<string, string>("output", "T");
  if (AutoMappingFnDynamic(op_src, op, value) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status ParseSubgraphPostFnWhileInput(int data_index, int& parent_index) {
  parent_index = data_index;
  return SUCCESS;
}

Status ParseSubgraphPostFnWhileOutput(const std::string& subgraph_name, int retval_index, int& parent_index) {
  if (subgraph_name == "cond") {
    parent_index = -1;
  } else {
    parent_index = retval_index;
  }
  return SUCCESS;
}

Status ParseSubgraphPostFnWhile(const std::string& subgraph_name, const ge::Graph& graph) {
  return AutoMappingSubgraphIndex(graph, ParseSubgraphPostFnWhileInput,
                                  [&](int retval_index, int& parent_index) -> Status {
                                    return ParseSubgraphPostFnWhileOutput(subgraph_name, retval_index, parent_index);
                                  });
}

REGISTER_CUSTOM_OP("While")
    .FrameworkType(CAFFE)
    .OriginOpType("While")
    .ParseParamsFn(AutoMappingFnWhile)
    .ParseSubgraphPostFn(ParseSubgraphPostFnWhile)
    .ImplyType(ImplyType::GELOCAL);

REGISTER_CUSTOM_OP("_While")
    .FrameworkType(CAFFE)
    .OriginOpType("_While")
    .ParseParamsFn(AutoMappingFnWhile)
    .ParseSubgraphPostFn(ParseSubgraphPostFnWhile)
    .ImplyType(ImplyType::GELOCAL);

REGISTER_CUSTOM_OP("StatelessWhile")
    .FrameworkType(CAFFE)
    .OriginOpType("StatelessWhile")
    .ParseParamsFn(AutoMappingFnWhile)
    .ParseSubgraphPostFn(ParseSubgraphPostFnWhile)
    .ImplyType(ImplyType::GELOCAL);
}  // namespace domi
