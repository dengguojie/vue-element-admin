/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief mul_grad(mul+mul+sum)
 *
 */

#ifndef FE_CONFUSION_MUL_GRAD_PASS_H
#define FE_CONFUSION_MUL_GRAD_PASS_H

#include <vector>
#include <string>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {

class MulGradFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &newNodes) override;

 private:
  vector<ge::NodePtr> GetNodesFromMapping(const string &id, Mapping &mapping);
  Status CheckPeerMul1InDataAnchors(const ge::OutDataAnchorPtr& outputAnchor,
                                      const size_t& expectedNum);
  const string FUSED_OP_TYPE = "ConfusionMulGrad";
};
}  // namespace fe
#endif  // FE_CONFUSION_MUL_GRAD_PASS_H

