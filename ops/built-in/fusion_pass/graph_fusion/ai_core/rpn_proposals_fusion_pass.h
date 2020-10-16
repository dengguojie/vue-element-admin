/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @file  layernormgrad_fusion_pass.h
 *
 * @brief RpnProposals fusion pass(RpnProposalsD --> ScoreFilterPreSort & RpnProposalsPostProcessing)
 *
 */

#ifndef FE_RPNPROPOSALS_FUSION_PASS_H
#define FE_RPNPROPOSALS_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class RpnProposalsFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
private:
    const string FUSED_OP_TYPE = "ScoreFilterPreSort_RpnProposalPostProcessing";

};

}  // namespace fe

#endif  // FE_RPNPROPOSALS_FUSION_PASS_H

