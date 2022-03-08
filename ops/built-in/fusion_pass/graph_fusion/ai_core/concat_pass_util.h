/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the
 License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file concat_pass_util.h
 *
 * @brief util for concat pass
 *
 * @version 1.0
 *
 */

#ifndef TBE_OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONCAT_PASS_UTIL_H_
#define TBE_OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONCAT_PASS_UTIL_H_

#include "graph/utils/attr_utils.h"
#include "graph/utils/node_utils.h"

namespace fe {
struct OriNodeInfo {
  ge::NodePtr pre_base_node;
  int64_t concat_dim;
  int64_t base_inputs_num;
  int64_t nums_input;
  int64_t max_inputs;
  int64_t node_idx;
  int64_t split_rank;
  std::string op_type;
  OriNodeInfo() {
    pre_base_node == nullptr;
    concat_dim = 0;
    base_inputs_num = 1;
    nums_input = 1;
    max_inputs = 1;
    node_idx = 0;
    split_rank = 0;
    op_type = "";
  }
};
int64_t GetMaxInputsNum(const ge::NodePtr& fused_node);
bool CheckNeedChanged(const ge::OpDescPtr& fused_desc, const int64_t max_inputs);
Status RemoveInvalidEdge(ge::NodePtr& fused_node, ge::OpDescPtr& fused_desc, const string op_type);
Status SplitConcatNode(ge::ComputeGraph& graph, vector<ge::NodePtr>& new_nodes, ge::NodePtr& fused_node,
                       const int64_t max_inputs, const string op_type);
}  // namespace fe

#endif  // TBE_OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONCAT_PASS_UTIL_H_
