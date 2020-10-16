/**
 * @file matrix_diag_fusion_pass.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief clip fusion pass(min --> max)
 *
 * @author z00475643
 */

#ifndef FE_MATRIX_DIAG_FUSION_H
#define FE_MATRIX_DIAG_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
namespace fe
{
  class MatrixDiagFusionPass: public PatternFusionBasePass
  {
  protected:
    vector<FusionPattern*> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph &graph,
                  Mapping &mapping,
                  vector<ge::NodePtr> &fusionNodes) override;
  private:
    const string FUSED_OP_TYPE = "MatrixDiagD";
  };

}  // namespace fe

#endif  // FE_MATRIX_DIAG_FUSION_H
