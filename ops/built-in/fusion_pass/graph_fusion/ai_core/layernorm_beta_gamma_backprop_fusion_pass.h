/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief clip fusion pass(min --> max)
 *
 */
#ifndef FE_FUSEDBATCHNORM_FUSION_PASS_H
#define FE_FUSEDBATCHNORM_FUSION_PASS_H

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
  vector<FusionPattern *> DefinePatterns() override;
  Status Run(ge::ComputeGraph& graph) override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &newNodes) override;

 private:
  Status MatchPass(ge::ComputeGraph& graph, vector<LayerNormMatchResult>& passMatchResultVec);
  Status GetAllLayerNormBetaGammaBackpropNodes(ge::ComputeGraph& graph, vector<NodePtr>& batchNormNodeVec);
  Status MatchLayerNormBetaGammaBackpropNode(NodePtr bnNodePtr, LayerNormMatchResult &matchResult);
  Status FusionGraphWithPass(ge::ComputeGraph& graph, LayerNormMatchResult &matchResult);
  const string FUSED_OP_TYPE = "LayerNormBetaGammaBackprop";
};
}  // namespace fe
#endif  // FE_FUSEDBATCHNORM_FUSION_PASS_H