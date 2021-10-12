/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file conv2d_group_fusion_pass.h
 * \brief split fusion pass(conv2d --> conv2d/splited conv2d/depthwise conv2d)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV2D_GROUP_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV2D_GROUP_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class Conv2DGroupFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

private:
  Status SwapNumChn(ge::OpDescPtr opDesc, bool bInput, uint32_t index);
  Status ProcessDepthwiseConv(ge::NodePtr convNode);
  int64_t GetGroups(ge::OpDescPtr &convDesc);
  bool GenerateSplitNode(ge::ComputeGraph &graph, ge::OpDescPtr &convDesc, int64_t &groups, ge::NodePtr &splitNode,
                         ge::GeTensorDesc &splitOutDesc);
  bool GenerateNewConvNodes(ge::ComputeGraph &graph, ge::OpDescPtr &convDesc, const ge::GeTensorDesc &splitOutDesc,
                            vector<ge::NodePtr> &newConvNodes, ge::GeTensorDesc &newConvOutDesc);
  bool GenerateConcatNode(ge::ComputeGraph &graph, ge::OpDescPtr &convDesc, const int64_t &groups,
                          ge::GeTensorDesc &newConvOutDesc, ge::NodePtr &concatNode);
  bool Relink(ge::NodePtr &convNode, ge::NodePtr &splitNode, vector<ge::NodePtr> &newConvNodes,
              ge::NodePtr &concatNode);
  bool IsVariableOrDataNode(const ge::NodePtr &convInputNode);
  Status CloneAndLinkQuants(ge::ComputeGraph &graph, const ge::NodePtr &splitNode, const int64_t &group,
                            vector<ge::NodePtr> &newConvNodes);
  Status SplitDequant(ge::ComputeGraph &graph, const ge::NodePtr &concatNode, const int64_t &group,
                      vector<ge::NodePtr> &newConvNodes);
  Status ProcQuantIfNeed(ge::ComputeGraph &graph, const ge::NodePtr &splitNode, const ge::NodePtr &concatNode,
                         const int64_t &groups, vector<ge::NodePtr> &newConvNodes);
  Status ProcessGroupConv(ge::ComputeGraph &graph, ge::NodePtr &convNode);
  const string FUSED_OP_TYPE = "Conv2D";
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV2D_GROUP_FUSION_PASS_H_
