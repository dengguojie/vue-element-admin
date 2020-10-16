#ifndef FE_SPACE_TO_BATCH_ND_FUSION_H
#define FE_SPACE_TO_BATCH_ND_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ConstToAttrSpaceToBatchNdPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
private:
    const string FUSED_OP_TYPE = "SpaceToBatchNDD";
};
}  // namespace fe
#endif  // FE_SPACE_TO_BATCH_ND_FUSION_H
