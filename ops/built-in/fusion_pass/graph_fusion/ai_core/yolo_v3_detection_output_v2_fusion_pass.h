#ifndef FE_YOLOV3_DETECTION_OUTPUT_V2_FUSION_H
#define FE_YOLOV3_DETECTION_OUTPUT_V2_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
  class YoloV3DetectionOutputV2Pass: public PatternFusionBasePass {
  protected:
    vector<FusionPattern*> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph &graph,
                  Mapping &mapping,
                  vector<ge::NodePtr> &fusionNodes) override;
  private:
      const string FUSED_OP_TYPE = "YoloV3DetectionOutputV2";
  };
}  // namespace fe

#endif  // FE_YOLOV3_DETECTION_OUTPUT_V2_FUSION_H
