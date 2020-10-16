/**
 * @file proposal_fusion_pass.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief 
 *
 */
#ifndef FE_PROPOSAL_FUSION_H
#define FE_PROPOSAL_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe
{
    class ProposalFusionPass: public PatternFusionBasePass
    {
    protected:
        vector<FusionPattern*> DefinePatterns() override;
      Status Fusion(ge::ComputeGraph &graph,
                    Mapping &mapping,
                    vector<ge::NodePtr> &fusionNodes) override;

    private:
        void GenerateShifts(int height, int width, float feat_stride, vector<float>& shifts);
        Status GenerateAnchorsFp16(uint16_t *output1, ge::NodePtr proposalVNode);
        const string FUSED_OP_TYPE = "ProposalD";
    };


}  // namespace fe

#endif  // FE_PROPOSAL_FUSION_H
