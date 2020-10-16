/**	
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief LayerNormGrad fusion pass(LayerNormGrad --> LayerNormXBackprop & LayerNormBetaGammaBackprop)
 *	
 */	

#ifndef FE_RNN_FUSION_PASS_H
#define FE_RNN_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class RNNFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &newNodes) override;

private:
  ge::GeTensorDesc ProcessStatic(ge::NodePtr fusedNode, int32_t num_output,
          ge::NodePtr &innerproductNode, ge::ComputeGraph &graph,
          vector<ge::NodePtr> &newNodes, bool &failStatus);
  vector<ge::NodePtr> ProcessRnnCell(ge::NodePtr fusedNode, ge::ComputeGraph &graph,
          ge::GeTensorDesc &outInnerProductTensorDesc, vector<ge::NodePtr> &newNodes,
          bool &failStatus, const bool has_static);
  const string FUSED_OP_TYPE = "SplitVD_BasicRNNCell_ConcatD";
};

}  // namespace fe

#endif  // FE_RNN_FUSION_PASS_H 