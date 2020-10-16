/**
 * @file prior_box_fusion_pass.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief 
 *
 * @author a00502265
 */
#ifndef FE_PRIOR_BOX_FUSION_H
#define FE_PRIOR_BOX_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
  class PriorBoxPass: public PatternFusionBasePass {
  protected:
    vector<FusionPattern*> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph &graph,
                  Mapping &mapping,
                  vector<ge::NodePtr> &newNodes) override;
  private:
      const string FUSED_OP_TYPE = "PriorBoxD";
  };
}  // namespace fe

#endif  // FE_PRIOR_BOX_FUSION_H
