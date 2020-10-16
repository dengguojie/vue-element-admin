/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief split fusion pass(depthwise_conv2d_backprop_filter --> depthwise_conv2d_backprop_filter_d)
 *
 */

#ifndef FE_DEPTHWISE_CONV2D_BACKPROP_FILTER_FUSION_H
#define FE_DEPTHWISE_CONV2D_BACKPROP_FILTER_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class DepthwiseConv2DBackpropFilterFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
private:
    const string FUSED_OP_TYPE = "DepthwiseConv2DBackpropFilterD";
};

}  // namespace fe

#endif  // FE_DEPTHWISE_CONV2D_BACKPROP_FILTER_FUSION_H
