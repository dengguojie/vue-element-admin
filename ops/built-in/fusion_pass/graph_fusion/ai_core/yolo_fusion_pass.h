#ifndef FE_YOLO_FUSION_H
#define FE_YOLO_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
  class YoloPass: public PatternFusionBasePass {
  protected:
    vector<FusionPattern*> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph &graph,
                  Mapping& mapping,
                  vector<ge::NodePtr> &newNodes) override;
  private:
      const string FUSED_OP_TYPE = "Yolo";
  };
}  // namespace fe

#endif  // FE_YOLO_FUSION_H
