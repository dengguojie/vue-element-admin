/**
 * @file mean_grad_fusion_pass.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief 
 *
 * @author z00522339
 */

#ifndef FE_MEAN_GRAD_FUSION_PASS_H
#define FE_MEAN_GRAD_FUSION_PASS_H


#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

using namespace std;

namespace fe {

class MeanGradFusionPass: public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;

 private:
  Status ParseParaFromConst(ge::NodePtr, int32_t & param, int index);
  Status RemoveConstOpInput(ge::ComputeGraph& graph, ge::NodePtr node);
  const string FUSED_OP_TYPE = "MeanGrad";
};

}  // namespace fe


#endif  // FE_MEAN_GRAD_FUSION_PASS_H
