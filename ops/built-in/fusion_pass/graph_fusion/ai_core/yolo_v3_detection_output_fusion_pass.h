#ifndef FE_YOLOV3_DETECTION_OUTPUT_FUSION_H
#define FE_YOLOV3_DETECTION_OUTPUT_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
  class YoloV3DetectionOutputPass: public PatternFusionBasePass {
  protected:
    vector<FusionPattern*> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph &graph,
                  Mapping &mapping,
                  vector<ge::NodePtr> &fusionNodes) override;
  private:
      const string FUSED_OP_TYPE = "YoloV3DetectionOutputD";
  };
}  // namespace fe

#endif  // FE_YOLOV3_DETECTION_OUTPUT_FUSION_H