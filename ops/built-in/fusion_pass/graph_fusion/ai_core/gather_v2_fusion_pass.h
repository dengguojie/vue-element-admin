/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief split fusion pass(gather_v2 --> gather_v2_d)
 *
 */

#ifndef FE_GATHERV2_FUSION_H
#define FE_GATHERV2_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ConstToAttrGatherV2Pass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
private:
    const string FUSED_OP_TYPE = "GatherV2D";
};

}  // namespace fe

#endif  // FE_GATHERV2_FUSION_H