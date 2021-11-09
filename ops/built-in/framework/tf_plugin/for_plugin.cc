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
 * \file for_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "graph/ge_attr_value.h"

namespace domi {
const int INDEX_2 = 2;

Status AutoMappingFnFor(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("input", "T");
  value["out"] = pair<string, string>("output", "T");
  if (AutoMappingFnDynamic(op_src, op, value) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status ParseSubgraphPostFnFor(const std::string& subgraph_name, const ge::Graph& graph) {
  return AutoMappingSubgraphIndex(
      graph, [](int data_index) { return (data_index == 0) ? 0 : data_index + INDEX_2; },
      [](int retval_index) { return retval_index; });
}
REGISTER_CUSTOM_OP("For")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("For")
    .ParseParamsFn(AutoMappingFnFor)
    .ParseSubgraphPostFn(ParseSubgraphPostFnFor)
    .ImplyType(ImplyType::GELOCAL);
}  // namespace domi
