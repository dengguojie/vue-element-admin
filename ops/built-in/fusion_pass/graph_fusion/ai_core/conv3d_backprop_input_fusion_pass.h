/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief split fusion pass(conv3d_backprop_input --> conv3d_backprop_input_d)
 *
 */

#ifndef FE_CONV3DBACKPROPINPUT_FUSION_H
#define FE_CONV3DBACKPROPINPUT_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ConstToAttrConv3dBackpropInputPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping& mapping, vector<ge::NodePtr> &newNodes) override;
private:
    const string FUSED_OP_TYPE = "Conv3DBackpropInputD";
};

}  // namespace fe

#endif  // FE_CONV3DBACKPROPINPUT_FUSION_H
