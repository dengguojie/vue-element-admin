/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief Permute fusion pass(softmax --> TransposeD)
 *
 */

#ifndef FE_SOFTMAX_FUSION_PASS_H
#define FE_SOFTMAX_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class SoftmaxFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector <ge::NodePtr> &newNodes) override;
  const string FUSED_OP_TYPE = "TransposeD_SoftmaxV2";
};

}  // namespace fe

#endif  // FE_SOFTMAX_FUSION_PASS_H