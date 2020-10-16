/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @file  square_sum_maximum_rsqrt_mul_fusion_pass.h
 *
 * @brief square_sum_maximum_rsqrt_mul fusion pass(square_sum_maximum_rsqrt_mul
 * --> l2_normalize)
 *
 */

#ifndef FE_A_A_SQUARE_SUM_MAXIMUN_RSQRT_MUL_FUSION_PASS_H
#define FE_A_A_SQUARE_SUM_MAXIMUN_RSQRT_MUL_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include <string>
#include <vector>

namespace fe {
class AASquareSumMaximumRsqrtMulFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;

private:
  Status CheckPeerAllInDataAnchors(const ge::OutDataAnchorPtr &outputAnchor,
                                   const size_t &expectedNum);
  Status IsMatch(ge::NodePtr &squareNode, ge::NodePtr &sumNode,
                 ge::NodePtr &maximumNode, ge::NodePtr &rsqrtNode,
                 ge::NodePtr &mulNode);
  const string FUSED_OP_TYPE = "L2Normalize";
};

} // namespace fe
#endif // FE_A_A_SQUARE_SUM_MAXIMUN_RSQRT_MUL_FUSION_PASS_H
