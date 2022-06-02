/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
#ifndef OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_AI_CORE_CUBE_BROADCAST_FUSION_CHECK_UTIL_H_
#define OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_AI_CORE_CUBE_BROADCAST_FUSION_CHECK_UTIL_H_

#include <string.h>
#include "anchor_util.h"
#include "error_util.h"
#include "graph/anchor.h"
#include "graph/node.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

inline bool IsBroadcastMatMulSupported(std::vector<ge::NodePtr> &broadcast_nodes, std::vector<ge::NodePtr> &cube_nodes,
                                       const string &fused_op_type) {
  if (broadcast_nodes.empty() || cube_nodes.empty()) {
    return true;
  }
  auto cube_output_desc = GetCurrNodeOutputDesc(cube_nodes[0], 0);
  auto cube_shape = cube_output_desc->GetShape().GetDims();
  for (const auto& broadcast_node : broadcast_nodes) {
    auto broad_output_desc = GetCurrNodeOutputDesc(broadcast_node, 0);
    auto broad_shape = broad_output_desc->GetShape().GetDims();
    // Broadcast scene support batch 1 add with batch n, not support no batch add with batch n
    FUSION_PASS_CHECK(cube_shape != broad_shape, OP_LOGD(fused_op_type.c_str(),
                      "The ub_fusion only support that output shape of node [%s] and node [%s] are equal",
                      broadcast_node->GetType().c_str(), cube_nodes[0]->GetType().c_str()), return false);
  }
  return true;
}
#endif  // OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_AI_CORE_CUBE_BROADCAST_FUSION_CHECK_UTIL_H_
