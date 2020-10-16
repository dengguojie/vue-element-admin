/**
 * @file range_fusion_pass.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief 
 *
 */
#ifndef FE_RANGED_FUSION_H
#define FE_RANGED_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe
{
    class RangeFusionPass: public PatternFusionBasePass
    {
    protected:
        vector<FusionPattern*> DefinePatterns() override;
        Status Fusion(ge::ComputeGraph &graph,
                      Mapping &mapping,
                      vector<ge::NodePtr> &fusionNodes) override;
    private:
        const string FUSED_OP_TYPE = "RangeD";
    };

}  // namespace fe

#endif  // FE_RANGE_FUSION_H
