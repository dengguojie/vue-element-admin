#ifndef FE_TRANSDATA_TRANSDATA_FUSION_PASS_H
#define FE_TRANSDATA_TRANSDATA_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class TransdataTransdataPass: public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
private:
    Status RemoveNode(ge::NodePtr node, ge::ComputeGraph &graph);
    const string FUSED_OP_TYPE = "TransData_FullyConnection";
};
}  // namespace fe

#endif  // FE_TRANSDATA_TRANSDATA_FUSION_PASS_H