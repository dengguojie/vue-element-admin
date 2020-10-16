/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @file  a_conv2d_mul_fusion_pass.h
 *
 * @brief conv-mul fusion pass(conv2d-mul --> conv)
 *
 */

#ifndef FE_A_CONV_MUL_FUSION_PASS_H
#define FE_A_CONV_MUL_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class Conv2DMulFusionPass: public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &newNodes) override;
private:
    const string FUSED_OP_TYPE= "Conv2D_Mul";
};

}  // namespace fe

#endif  // FE_A_CONV_MUL_FUSION_PASS_H
