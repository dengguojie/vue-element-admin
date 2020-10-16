#ifndef FE_MAX_POOL3D_GRAD_GRAD_FUSION_H
#define FE_MAX_POOL3D_GRAD_GRAD_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
namespace fe
{
  class MaxPool3DGradGradFusionPass: public PatternFusionBasePass
  {
  protected:
    vector<FusionPattern*> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph &graph,
                  Mapping &mapping,
                  vector<ge::NodePtr> &fusionNodes) override;

  private:
      int GetDHW(const std::vector<int32_t> & ksize , int32_t & D, int32_t & H, int32_t & W, ge::Format format);
      const string FUSED_OP_TYPE = "MaxPool3DGradGradD";
  };

}  // namespace fe

#endif  // FE_MAX_POOL3D_GRAD_GRAD_FUSION_H