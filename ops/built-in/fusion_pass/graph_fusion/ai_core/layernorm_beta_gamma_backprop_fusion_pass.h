/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file layernorm_beta_gamma_backprop_fusion_pass.h
 * \brief clip fusion pass(min --> max)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LAYERNORM_BETA_GAMMA_BACKPROP_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LAYERNORM_BETA_GAMMA_BACKPROP_FUSION_PASS_H_

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

using std::map;
using std::string;
using std::vector;
using namespace ge;
using namespace std;

namespace fe {

struct PassRemoveEdge {
  InDataAnchorPtr inAnchorPtr;
  OutDataAnchorPtr outAnchorPtr;
};

// Match result.
struct LayerNormMatchResult {
  NodePtr layerNormBetaGammaBackpropPtr;
  vector<NodePtr> castNodeVec;
};

class LayerNormBetaGammaBackpropFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Run(ge::ComputeGraph& graph) override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

 private:
  Status MatchPass(ge::ComputeGraph& graph, vector<LayerNormMatchResult>& passMatchResultVec);
  Status GetAllLayerNormBetaGammaBackpropNodes(ge::ComputeGraph& graph, vector<NodePtr>& batchNormNodeVec);
  Status MatchLayerNormBetaGammaBackpropNode(NodePtr bnNodePtr, LayerNormMatchResult& matchResult);
  Status FusionGraphWithPass(ge::ComputeGraph& graph, LayerNormMatchResult& matchResult);
  const string FUSED_OP_TYPE = "LayerNormBetaGammaBackprop";
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LAYERNORM_BETA_GAMMA_BACKPROP_FUSION_PASS_H_