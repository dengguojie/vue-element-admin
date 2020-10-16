#ifndef OPS_PASS_THROUGH_SENCOND_FUSION_H
#define OPS_PASS_THROUGH_SENCOND_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class PassThroughSecondFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;

private:
    Status InsertCurNode(ge::ComputeGraph &graph, ge::NodePtr oriNode);
    Status RemoveThisNode(ge::ComputeGraph &graph, ge::NodePtr thisNode);
    Status SetNodeAttrAndOpDesc(ge::OpDescPtr &curOpDesc, const ge::OpDescPtr &oriOpDesc);
    Status RemoveWeightNode(ge::ComputeGraph &graph, ge::NodePtr oriNode);
    Status UnlinkEdge(ge::NodePtr oriNode);
    const string FUSED_OP_TYPE = "SpaceToDepth";
};
}  // namespace fe
#endif  // OPS_PASS_THROUGH_SENCOND_FUSION_H
