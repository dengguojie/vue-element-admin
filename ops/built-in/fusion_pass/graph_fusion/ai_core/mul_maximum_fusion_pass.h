/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @file  mul_maximum_fusion_pass.h
 *
 * @brief mul_maximum fusion pass(mul_maximum --> leaky_relu)
 *
 */

#ifndef FE_MUL_MAXIMUN_FUSION_PASS_H
#define FE_MUL_MAXIMUN_FUSION_PASS_H

#include <vector>
#include <string>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {

class MulMaximumFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;

private:
  Status IsMatch(ge::NodePtr &mulNode, ge::NodePtr &maximumNode);
  Status CheckPeerMulInDataAnchors(const ge::OutDataAnchorPtr& outputAnchor,
          const size_t& expectedNum);
  const string FUSED_OP_TYPE = "LeakyRelu";
};

}  // namespace fe
#endif  // FE_MUL_MAXIMUN_FUSION_PASS_H
