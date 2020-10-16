/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief LayerNormGrad fusion pass(LayerNormGrad --> LayerNormXBackprop & LayerNormBetaGammaBackprop)
 *
 */

#ifndef FE_LSTM_FUSION_PASS_H
#define FE_LSTM_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ALSTMFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &newNodes) override;

private:
    vector<ge::NodePtr> ProcessLstmCellV2(ge::NodePtr fusedNode,ge::ComputeGraph &graph,ge::GeTensorDesc &outInnerProductTensorDesc,const ge::GeTensorDesc &wxhTensorDesc,const ge::GeTensorDesc &hTensorDesc,vector<ge::NodePtr> &newNodes,bool &failStatus, int32_t biasIndex, bool has_static);
    ge::GeTensorDesc ProcessStatic(ge::NodePtr fusedNode ,int32_t num_output,ge::NodePtr &innerproductNode,ge::NodePtr &dequantNode,ge::ComputeGraph &graph,vector<ge::NodePtr> &newNodes,bool &failStatus, int32_t xStaticIndex, int32_t wxStaticIndex);
    ge::GeTensorPtr ProcessWxh(ge::NodePtr fusedNode,bool &failStatus ,int32_t &wxIndex, int32_t &whIndex, int32_t c0Index);
    const string FUSED_OP_TYPE = "SplitVD_BasicLSTMCellV2_ConcatD";
};

}  // namespace fe

#endif  // FE_LSTM_FUSION_PASS_H
