/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief batch_multi_class_nms fusion pass
 *
 */

#ifndef BATCH_MULTI_CLASS_NMS_FUSION_PASS_H
#define BATCH_MULTI_CLASS_NMS_FUSION_PASS_H

#include <vector>
#include <string>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {

class BatchMultiClassNonMaxSuppressionFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &newNodes) override;

private:
  vector<ge::NodePtr> GetNodesFromMapping(const string &id, Mapping &mapping);
  bool CheckTransposeBeforeSlice(ge::NodePtr checkNode);
  const string FUSED_OP_TYPE = "BatchMultiClassNonMaxSuppression";
};
} // namespace fe
#endif // BATCH_MULTI_CLASS_NMS_FUSION_PASS_H
