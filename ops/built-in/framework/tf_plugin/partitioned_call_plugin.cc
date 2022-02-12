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
 * \file partitioned_call_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "graph/ge_attr_value.h"

namespace domi {
Status AutoMappingFnPartitionedCall(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("args", "Tin");
  value["out"] = pair<string, string>("output", "Tout");
  if (AutoMappingFnDynamic(op_src, op, value) != SUCCESS) {
    return FAILED;
  }

  return SUCCESS;
}

Status ParseSubgraphPostFnPartitionedCall(const std::string& subgraph_name, const ge::Graph& graph) {
  return AutoMappingSubgraphIndex(
      graph, [](int data_index) { return data_index; }, [](int retval_index) { return retval_index; });
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(TENSORFLOW)
    .OriginOpType({"PartitionedCall", "StatefulPartitionedCall"})
    .ParseParamsFn(AutoMappingFnPartitionedCall)
    .ParseSubgraphPostFn(ParseSubgraphPostFnPartitionedCall)
    .ImplyType(ImplyType::GELOCAL);
}  // namespace domi
