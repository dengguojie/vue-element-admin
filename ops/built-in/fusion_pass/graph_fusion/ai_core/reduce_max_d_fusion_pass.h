/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief fusedbatchnormgrad fusion pass(min --> max)
 *
 */

#ifndef FE_REDUCEMAXD_FUSION_PASS_H
#define FE_REDUCEMAXD_FUSION_PASS_H

#include <vector>
#include <string>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {

class ReduceMaxDFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &newNodes) override;

 private:
  vector<ge::NodePtr> GetNodesFromMapping(const string &id, Mapping &mapping);
  const string FUSED_OP_TYPE = "ReduceMaxD";
};
}  // namespace fe
#endif  // FE_REDUCEMAXD_FUSION_PASS_H

