/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief split fusion pass(conv3d_backprop_filter --> conv3d_backprop_filter_d)
 *
 */

#ifndef FE_CONV3DBACKPROPFILTER_FUSION_H
#define FE_CONV3DBACKPROPFILTER_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ConstToAttrConv3dBackpropFilterPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping& mapping, vector<ge::NodePtr> &newNodes) override;
private:
   const string FUSED_OP_TYPE = "Conv3DBackpropFilter";
};

}  // namespace fe

#endif  // FE_CONV3DBACKPROPFILTER_FUSION_H