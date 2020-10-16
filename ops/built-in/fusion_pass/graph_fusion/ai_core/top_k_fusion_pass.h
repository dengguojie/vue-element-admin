/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief TopK fusion pass(TopKV2 --> TopK)
 *
 */

#ifndef FE_TOPK_FUSION_H
#define FE_TOPK_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe
{
    class TopKFusionPass: public PatternFusionBasePass
    {
    protected:
        vector<FusionPattern*> DefinePatterns() override;
        Status Fusion(ge::ComputeGraph &graph,
                      Mapping &mapping,
                      vector<ge::NodePtr> &fusionNodes) override;
    private:
        const string FUSED_OP_TYPE = "TopKD";
    };

}  // namespace fe

#endif  // FE_TOPK_FUSION_H
