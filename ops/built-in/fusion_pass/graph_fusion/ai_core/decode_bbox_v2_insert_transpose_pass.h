/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief decode_bbox_v2 fusion pass
 *
 */

#ifndef DECODE_BBOX_V2_INSERT_TRANSPOSE_PASS_H
#define DECODE_BBOX_V2_INSERT_TRANSPOSE_PASS_H

#include <vector>
#include <string>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {

class DecodeBboxV2InsertTransposePass : public PatternFusionBasePass {
protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &newNodes) override;

private:
  vector<ge::NodePtr> GetNodesFromMapping(const string &id, Mapping &mapping);
  const string FUSED_OP_TYPE = "DecodeBboxV2";
  };
}  // namespace fe
#endif  // DECODE_BBOX_V2_INSERT_TRANSPOSE_PASS_H

