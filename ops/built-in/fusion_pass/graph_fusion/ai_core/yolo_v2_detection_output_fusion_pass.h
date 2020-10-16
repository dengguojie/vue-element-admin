#ifndef FE_YOLOV2_DETECTION_OUTPUT_FUSION_H
#define FE_YOLOV2_DETECTION_OUTPUT_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
  class YoloV2DetectionOutputPass: public PatternFusionBasePass {
  protected:
    vector<FusionPattern*> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph &graph,
                  Mapping &mapping,
                  vector<ge::NodePtr> &fusionNodes) override;
  private:
      const string FUSED_OP_TYPE = "YoloV2DetectionOutputD";
  };
}  // namespace fe

#endif  // FE_YOLOV2_DETECTION_OUTPUT_FUSION_H