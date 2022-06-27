/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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
 * \file attention_score_fusion_pass.h
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_ATTENTION_SOCRE_FUSION_PASS_H
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_ATTENTION_SOCRE_FUSION_PASS_H
#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ZAttentionScoreFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusion_nodes) override;
  Status CheckPlatformInfo();
  bool IsTargetPlateform(const std::string plateform);

private:
  ge::NodePtr batchMatmulNode1 = nullptr;
  ge::NodePtr batchMatmulNode2 = nullptr;
  ge::NodePtr batchMatmulNode3 = nullptr;
  ge::NodePtr softmaxNode = nullptr;
  ge::NodePtr confusionTransposeNode = nullptr;
  ge::NodePtr fusedMulAddNode = nullptr;
  bool traning = true;
  Status SetAttrForBsbDesc(std::shared_ptr<ge::OpDesc> bsbDesc);
  Status DeleteFusionNode(ge::ComputeGraph &graph);
  Status AddControlEdgesForBsbNode(ge::NodePtr bsbNode);
  Status AddOutputEdgeForNode(ge::NodePtr oriNode, ge::NodePtr newNode, int unlinkIndex, int newNodeIndex) const;
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_ATTENTION_SOCRE_FUSION_PASS_H
