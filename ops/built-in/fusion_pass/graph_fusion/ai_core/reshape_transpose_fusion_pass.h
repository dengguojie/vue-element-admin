/**
 * @file reshape_transpose_fusion_pass.h
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief confusionTranspose fusion pass(Transpose-Reshape --> confusionTranspose)
 *
 */

// .h和.cpp文件提交至ops/built-in/fusion_pass/目录下
#ifndef FE_TILE_FUSION_H
#define FE_TILE_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ReshapeTransposeFusionPass : public PatternFusionBasePass { //ConstToAttrXXXPass类名根据算子自定义，XXX为算子名称
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
private:
    const string FUSED_OP_TYPE = "ConfusionTranspose";
};

}  // namespace fe

#endif  // FE_TILE_FUSION_H
