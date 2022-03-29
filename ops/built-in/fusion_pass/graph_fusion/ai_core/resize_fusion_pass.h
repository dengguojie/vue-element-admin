/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief resize fusion pass(resize-->
 * resize_nearest_neighbor_v2d/resize_bilinear_v2d)
 *
 */

#ifndef FE_RESIZE_FUSION_H
#define FE_RESIZE_FUSION_H

#include <vector>

#include "graph/tensor.h"
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ResizeFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
};
}  // namespace fe

#endif  // FE_RESIZE_FUSION_H