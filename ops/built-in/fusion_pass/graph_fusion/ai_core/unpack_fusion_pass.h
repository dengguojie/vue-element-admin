/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief Unpack fusion pass(Unpack --> Split + Unpack)
 *
 */

#ifndef FE_SPLIT_FUSION_PASS_H
#define FE_SPLIT_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
  class UnpackFusionPass : public PatternFusionBasePass {
  protected:
    vector<FusionPattern*> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph &graph,
                  Mapping &mapping,
                  vector <ge::NodePtr> &newNodes) override;

  private:
      Status AddUnpackOps(ge::OpDescPtr fused_desc,
                          ge::ComputeGraph &graph,
                          vector<ge::NodePtr> &newNodes,
                          std::vector<ge::GeTensorDesc> output_desc,
                          ge::NodePtr fused_node,
                          ge::NodePtr splitvd_base_node,
                          int64_t num, int64_t axis,
                          int64_t i, int64_t j, int64_t mini_out);
      const string FUSED_OP_TYPE = "SplitVD_Unpack";
  };
}  // namespace fe

#endif  // FE_SPLIT_FUSION_PASS_H