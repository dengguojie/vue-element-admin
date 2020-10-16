#ifndef FE_FULLYCONNECTION_RESHAPE_FUSION_PASS_H
#define FE_FULLYCONNECTION_RESHAPE_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class FullyConnectionReshapePass: public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
private:
    const string FUSED_OP_TYPE = "FullyConnection";
};
}  // namespace fe

#endif  // FE_FULLYCONNECTION_RESHAPE_FUSION_PASS_H