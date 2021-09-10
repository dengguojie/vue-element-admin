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
 * \file if_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "graph/ge_attr_value.h"

namespace domi {
Status AutoMappingFnIf(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("input", "Tin");
  value["out"] = pair<string, string>("output", "Tout");
  if (AutoMappingFnDynamic(op_src, op, value) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status ParseSubgraphPostFnIf(const std::string& subgraph_name, const ge::Graph& graph) {
  return AutoMappingSubgraphIndex(
      graph, [](int data_index) { return data_index + 1; }, [](int retval_index) { return retval_index; });
}

REGISTER_CUSTOM_OP("If")
    .FrameworkType(CAFFE)
    .OriginOpType("If")
    .ParseParamsFn(AutoMappingFnIf)
    .ParseSubgraphPostFn(ParseSubgraphPostFnIf)
    .ImplyType(ImplyType::GELOCAL);

REGISTER_CUSTOM_OP("_If")
    .FrameworkType(CAFFE)
    .OriginOpType("_If")
    .ParseParamsFn(AutoMappingFnIf)
    .ParseSubgraphPostFn(ParseSubgraphPostFnIf)
    .ImplyType(ImplyType::GELOCAL);

REGISTER_CUSTOM_OP("StatelessIf")
    .FrameworkType(CAFFE)
    .OriginOpType("StatelessIf")
    .ParseParamsFn(AutoMappingFnIf)
    .ParseSubgraphPostFn(ParseSubgraphPostFnIf)
    .ImplyType(ImplyType::GELOCAL);
}  // namespace domi
