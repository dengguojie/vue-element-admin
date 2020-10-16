/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief split fusion pass(conv2d --> conv2d/splited conv2d/depthwise conv2d)
 *
 */

#ifndef FE_CONV2D_GROUP_FUSION_H
#define FE_CONV2D_GROUP_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class Conv2DGroupFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern* > DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &newNodes) override;

private:
   Status SwapNumChn(ge::OpDescPtr opDesc, bool bInput, uint32_t index);
   Status ProcessDepthwiseConv(ge::NodePtr convNode);
   const string FUSED_OP_TYPE = "Conv2D";
};
}
#endif
