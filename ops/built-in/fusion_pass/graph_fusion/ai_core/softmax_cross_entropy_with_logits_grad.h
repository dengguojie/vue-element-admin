#ifndef FE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_GRAD_H
#define FE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_GRAD_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class SoftmaxCrossEntropyWithLogitsGradPass: public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;

private:
    Status RemoveNode(ge::NodePtr node, ge::ComputeGraph &graph);
    const string FUSED_OP_TYPE = "SoftmaxCrossEntropyWithLogits_Mul";
};
}  // namespace fe

#endif  // FE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_GRAD_H