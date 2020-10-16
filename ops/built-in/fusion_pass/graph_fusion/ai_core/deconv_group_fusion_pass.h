/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief split fusion pass(deconv --> conv2d_backprop_input_d)
 *
 */

#ifndef FE_DECONV_GROUP_FUSION_H
#define FE_DECONV_GROUP_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class DeconvGroupFusionPass : public PatternFusionBasePass {
protected:
    vector<FusionPattern *> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph &graph, Mapping& mapping, vector<ge::NodePtr> &fusionNodes) override;

private:
    const string FUSED_OP_TYPE = "Deconvolution";
};
}  // namespace fe

#endif  // FE_DECONV_GROUP_FUSION_H
