/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief LayerNormGrad fusion pass(LayerNormGrad --> LayerNormXBackprop & LayerNormBetaGammaBackprop)
 *
 */

#ifndef FE_COPY_FUSION_H
#define FE_COPY_FUSION_H

#include"graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class CopyFusionPass: public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
               Mapping &mapping,
               vector<ge::NodePtr> &newNodes) override;
  const string FUSED_OP_TYPE = "Copy";
};
}  // namespace fe
#endif  // FE_COPY_FUSION_H
