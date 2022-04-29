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
 * \file batch_multi_class_nms_enable_subgraph_fusion_pass.h
 * \brief batch_multi_class_nms_enable_subgraph fusion pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BATCH_MULTI_CLASS_NMS_ENABLE_SUBGRAPH_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BATCH_MULTI_CLASS_NMS_ENABLE_SUBGRAPH_FUSION_PASS_H_
#include <vector>
#include <string>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class BatchMultiClassNonMaxSuppressionEnableSubgraphFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

 private:
  vector<ge::NodePtr> GetNodesFromMapping(const string& id, Mapping& mapping);
  void GetNMSNodeIndex(ge::ComputeGraph& graph, const ge::NodePtr& checkNode, int64_t& nmsNodeIndex);
  Status AddNMSSubgraphAttr(ge::NodePtr& node, int64_t batchsize, int64_t maxTotalSize);
  Status UpdateSubgraphAttribute(ge::NodePtr& startNode, int64_t nmsNodeIndex);
  const string FUSED_OP_TYPE = "BatchMultiClassNonMaxSuppressionEnableSubgraph";
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BATCH_MULTI_CLASS_NMS_ENABLE_SUBGRAPH_FUSION_PASS_H_
