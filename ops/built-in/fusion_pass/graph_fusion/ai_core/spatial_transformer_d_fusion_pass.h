#ifndef FE_SPATIAL_TRANSFORMER_D_FUSION_H
#define FE_SPATIAL_TRANSFORMER_D_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
  class SpatialTransformerDPass: public PatternFusionBasePass {
  protected:
    vector<FusionPattern*> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph &graph,
                  Mapping &mapping,
                  vector<ge::NodePtr> &newNodes) override;

  private:
      Status StnPreAddConst(ge::NodePtr &thisNode,
                            ge::OpDescPtr &thisOpDesc);
      Status StnHIndexFP16(const int32_t h, const int32_t w, uint16_t *output1);
      Status StnWIndexFP16(const int32_t h, const int32_t w, uint16_t *output1);
      Status MakeStnComputeLayer(ge::OpDescPtr &thisOpDesc,
                                 const ge::OpDescPtr &bottomOpDesc,
                                 const ge::OpDescPtr &formerOpDesc);
      Status MakeStnPreLayer(ge::OpDescPtr &thisOpDesc,
                             const ge::OpDescPtr &formerOpDesc,
                             bool hasInput1);
      const string FUSED_OP_TYPE = "StnPre_StnCompute";
  };
}  // namespace fe

#endif  // FE_SPATIAL_TRANSFORMER_D_FUSION_H
