/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file batch_multi_class_nms_refresh_const_node_fusion_pass.h
 * \brief batch_multi_class_nms_refresh_const_node fusion pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BATCH_MULTI_CLASS_NMS_REFRESH_SUBGRAPH_CONST_NODE_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BATCH_MULTI_CLASS_NMS_REFRESH_SUBGRAPH_CONST_NODE_FUSION_PASS_H_
#include <vector>
#include <string>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class BatchMultiClassNonMaxSuppressionRefreshSubgraphConstNodeFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

 private:
  vector<ge::NodePtr> GetNodesFromMapping(const string& id, Mapping& mapping);
  Status ReplaceNodeWithNewNode(ge::ComputeGraphPtr& subgraph, ge::NodePtr& oldNode, ge::NodePtr& newNode);
  Status UpdateConstNode(ge::ComputeGraphPtr& subgraph, map<int64_t, ge::NodePtr>& constNodeMap, int64_t maxTotalSize);
  Status AddPadDNode(ge::ComputeGraphPtr& subgraph, int32_t index, vector<vector<int64_t>> paddingVector);
  Status UpdateOutput(ge::ComputeGraphPtr& subgraph, int64_t maxTotalSize);
  void GetNMSNodeIndex(ge::ComputeGraph& graph, const ge::NodePtr& checkNode, int64_t& nmsNodeIndex);
  const string FUSED_OP_TYPE = "BatchMultiClassNonMaxSuppressionRefreshSubgraphConstNode";
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BATCH_MULTI_CLASS_NMS_REFRESH_SUBGRAPH_CONST_NODE_FUSION_PASS_H_
