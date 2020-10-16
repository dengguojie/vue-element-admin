#ifndef FE_SPP_FUSION_H
#define FE_SPP_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
  class SPPPass: public PatternFusionBasePass {
  protected:
    vector<FusionPattern*> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph &graph,
                  Mapping &mapping,
                  vector<ge::NodePtr> &newNodes) override;

  private:
      Status MakeConcatLayer(ge::OpDescPtr &concatOpDesc,
              vector<ge::OpDescPtr> fatherOp,
              int64_t concatDims);
      Status MakePoolingLayer(ge::OpDescPtr &poolingOpDesc,
              const ge::GeTensorDesc &inputDesc,
              int64_t hyramidLevel, int64_t poolMethod);
      const string FUSED_OP_TYPE = "SppPooling_ConcatD";
  };
}  // namespace fe

#endif  // FE_SPP_FUSION_H
