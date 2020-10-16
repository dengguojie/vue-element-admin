/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @file  biasadd_conv_fusion_pass.h
 *
 * @brief conv-biasadd fusion pass(conv-biasadd --> conv)
 *
 */

#ifndef FE_BIASADD_CONV_FUSION_H
#define FE_BIASADD_CONV_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class BiasaddConvFusionPass: public PatternFusionBasePass {
protected:
    vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
private:
    const string FUSED_OP_TYPE = "Conv2D/Conv3D/DepthwiseConv2D";
};

}  // namespace fe

#endif  // FE_BIASADD_CONV_FUSION_H
