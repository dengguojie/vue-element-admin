/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file tbe_ops_pass_util.h
 *
 * @brief util for ops pass
 *
 * @version 1.0
 *
 */



#ifndef TBE_OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_TBE_OPS_PASS_UTIL_H_
#define TBE_OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_TBE_OPS_PASS_UTIL_H_

#include "graph/utils/attr_utils.h"
#include "graph/utils/node_utils.h"

#define NOT_CHANGED_WITH_DYNAMIC_NODE(fused_nodes)                                                           \
  for (auto fused_node : fused_nodes) {                                                                      \
    bool unknown_shape = false;                                                                              \
    if (ge::NodeUtils::GetNodeUnknownShapeStatus(*(fused_node.get()), unknown_shape) == ge::GRAPH_SUCCESS && \
        unknown_shape) {                                                                                     \
      OP_LOGI(fused_node->GetOpDesc()->GetType().c_str(), "this node is dynamic node, will not be changed.");\
      return NOT_CHANGED;                                                                                    \
    }                                                                                                        \
  }

bool HasUnKnowDimShape(const ge::NodePtr &node_ptr);
bool HasUnKnowShape(const ge::NodePtr &node_ptr);
void ClearOpInferDepends(const ge::NodePtr& node_ptr);
bool IsUnknownShape(const std::vector<int64_t>& shape);
void RemoveInputDesc(ge::OpDescPtr op_desc, uint32_t index);
bool IsUnknownRankShape(const std::vector<int64_t>& shape);

/**
* Get int type const value from one node
* @param [in] fused_node the node with const input
* @param [in] name name of the const input in fused_node
* @param [out] const_value const int values in const input
* @return true:success, false:failed.
*/
bool GetIntConstValue(const ge::NodePtr& fused_node, const string& const_name,
                      std::vector<int64_t>&  const_value);

#endif  // TBE_OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_TBE_OPS_PASS_UTIL_H_
