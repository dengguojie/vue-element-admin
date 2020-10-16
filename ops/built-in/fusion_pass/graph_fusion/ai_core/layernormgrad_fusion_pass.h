/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @file  layernormgrad_fusion_pass.h
 *
 * @brief LayerNormGrad fusion pass(LayerNormGrad --> LayerNormXBackprop & LayerNormBetaGammaBackprop)
 *
 */

#ifndef FE_LAYERNORMGRAD_FUSION_PASS_H
#define FE_LAYERNORMGRAD_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class LayerNormGradFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
private:
    const string FUSED_OP_TYPE = "LayerNormXBackprop_LayerNormBetaGammaBackprop";
};

}  // namespace fe

#endif  // FE_LAYERNORMGRAD_FUSION_PASS_H