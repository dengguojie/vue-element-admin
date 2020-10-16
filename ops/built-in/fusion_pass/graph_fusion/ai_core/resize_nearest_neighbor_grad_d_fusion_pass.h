/**
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief ResizeNearestNeighborGrad fusion pass
 *
 */

#ifndef FE_RESIZE_NEAREST_NEIGHBOR_GRAD_D_FUSION_H
#define FE_RESIZE_NEAREST_NEIGHBOR_GRAD_D_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ConstToAttrResizeNearestNeighborGradPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
private:
    const string FUSED_OP_TYPE = "ResizeNearestNeighborV2GradD";

};
}  // namespace fe

#endif // FE_RESIZE_NEAREST_NEIGHBOR_GRAD_D_FUSION_H
