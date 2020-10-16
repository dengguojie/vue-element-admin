/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @file  maxpoolwithargmax_fusion_pass.h
 *
 * @brief MaxPoolWithArgmax fusion pass(MaxPoolWithArgmax --> MaxPoolWithArgmax & Mask2Argmax)
 *
 */

#ifndef FE_MAXPOOLWITHARGMAX_FUSION_PASS_H
#define FE_MAXPOOLWITHARGMAX_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class MaxPoolWithArgmaxFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
private:
    const string FUSED_OP_TYPE = "MaxPoolWithArgmax_Mask2Argmax_TransData";
};

}  // namespace fe

#endif  // FE_MAXPOOLWITHARGMAX_FUSION_PASS_H