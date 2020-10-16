/**
 * @file matrix_set_diag_fusion_pass.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief 
 *
 * @author z00475643
 */
#ifndef FE_MATRIX_SET_DIAG_FUSION_H
#define FE_MATRIX_SET_DIAG_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
namespace fe
{
  class MatrixSetDiagFusionPass: public PatternFusionBasePass
  {
  protected:
    vector<FusionPattern*> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph &graph,
                  Mapping &mapping,
                  vector<ge::NodePtr> &fusionNodes) override;
  private:
      const string FUSED_OP_TYPE = "MatrixSetDiagD";
  };

}  // namespace fe

#endif  // FE_MATRIX_SET_DIAG_FUSION_H
