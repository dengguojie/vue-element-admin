/**
 * @file nms_with_mask_fusion_pass.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief fusion pass(Add a pad op before NMSWithMask for input of box_scores)
 *
 */

#ifndef FE_NMS_WITH_MASK_FUSION_H
#define FE_NMS_WITH_MASK_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class NMSWithMaskFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
private:
   const string FUSED_OP_TYPE = "PadD_NMSWithMask";
};

}  // namespace fe

#endif  // FE_NMS_WITH_MASK_FUSION_H
