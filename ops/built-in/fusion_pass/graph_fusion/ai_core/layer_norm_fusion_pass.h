/**
 * @file layer_norm_fusion_pass.h
 *
 * Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
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
 * @brief layer norm fusion pass
 *
 */

#ifndef FE_LAYER_NORM_FUSION_H
#define FE_LAYER_NORM_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class LayerNormFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping& mapping,
                vector<ge::NodePtr> &fusionNodes) override;
private:
    const string FUSED_OP_TYPE = "LayerNorm";
};

}  // namespace fe

#endif  // FE_LAYER_NORM_FUSION_H
