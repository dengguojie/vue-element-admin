/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @brief padd depthwise_conv2d fusion pass
 *
 */

#ifndef _OPTIMIZER_FUSION_PADD_DEPTHWISECONV2D_FUSION_H_
#define _OPTIMIZER_FUSION_PADD_DEPTHWISECONV2D_FUSION_H_

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class PaddDepthwiseConv2dFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;

private:
    const string FUSED_OP_TYPE = "DepthwiseConv2D";
};
}  // namespace fe
#endif  // _OPTIMIZER_FUSION_PADD_DEPTHWISECONV2D_FUSION_H_
