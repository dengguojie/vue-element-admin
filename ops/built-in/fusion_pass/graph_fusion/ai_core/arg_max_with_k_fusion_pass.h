#ifndef FE_ARG_MAX_WITH_K_FUSION_PASS_H
#define FE_ARG_MAX_WITH_K_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
    class ArgMaxWithKFusionPass: public PatternFusionBasePass {
        protected:
        vector<FusionPattern*> DefinePatterns() override;
        Status Fusion(ge::ComputeGraph &graph,
        Mapping &mapping,
        vector<ge::NodePtr> &newNodes) override;
    private:
        const string FUSED_OP_TYPE = "ArgMaxWithKD";
    };
}  // namespace fe

#endif  // FE_ARG_MAX_WITH_K_FUSION_PASS_H