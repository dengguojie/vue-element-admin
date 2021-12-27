/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
 * \file layernorm_beta_gamma_backprop_v2_fusion_pass.h
 * \brief clip fusion pass(min --> max)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LAYERNORM_BETA_GAMMA_BACKPROP_V2_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LAYERNORM_BETA_GAMMA_BACKPROP_V2_FUSION_PASS_H_

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {

struct PassRemoveEdge {
  ge::InDataAnchorPtr inAnchorPtr;
  ge::OutDataAnchorPtr outAnchorPtr;
};

// Match result.
struct LayerNormMatchResult {
  ge::NodePtr layerNormBetaGammaBackpropPtr;
  std::vector<ge::NodePtr> castNodeVec;
};

class LayerNormBetaGammaBackpropV2FusionPass : public PatternFusionBasePass {
 protected:
  std::vector<FusionPattern*> DefinePatterns() override;
  Status Run(ge::ComputeGraph& graph) override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, std::vector<ge::NodePtr>& newNodes) override;

 private:
  Status MatchPass(const ge::ComputeGraph& graph, std::vector<LayerNormMatchResult>& passMatchResultVec);
  Status GetAllLayerNormBetaGammaBackpropV2Nodes(const ge::ComputeGraph& graph,
                                                 std::vector<ge::NodePtr>& batchNormNodeVec);
  Status MatchLayerNormBetaGammaBackpropV2Node(const ge::NodePtr lnNodePtr, LayerNormMatchResult& matchResult);
  Status FusionGraphWithPass(ge::ComputeGraph& graph, const LayerNormMatchResult& matchResult);
  const string FUSED_OP_TYPE = "LayerNormBetaGammaBackpropV2";
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LAYERNORM_BETA_GAMMA_BACKPROP_V2_FUSION_PASS_H_
