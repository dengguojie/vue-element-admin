/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @file  confusion_matrix_fusion_pass.cpp
 *
 * @brief confusion_matrix fusion pass(     --> confusion_matrix)
 *
 */

#ifndef FE_CONFUSION_MATRIX_FUSION_H
#define FE_CONFUSION_MATRIX_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe
{
class ConfusionMatrixFusionPass: public PatternFusionBasePass {
protected:
    vector<FusionPattern*> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph &graph,
                  Mapping &mapping,
                  vector<ge::NodePtr> &fusionNodes) override;
private:
    const string FUSED_OP_TYPE = "ConfusionMatrix";
};

}  // namespace fe

#endif  // FE_CONFUSION_MATRIX_FUSION_H
