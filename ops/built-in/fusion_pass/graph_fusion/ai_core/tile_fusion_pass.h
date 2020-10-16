/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @file  tile_fusion_pass.h
 *
 * @brief tile fusion pass(tile --> tile_d)
 *
 * author z00512353
 */

#ifndef FE_TILE_FUSION_H
#define FE_TILE_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ConstToAttrTilePass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping& mapping,
                vector<ge::NodePtr> &newNodes) override;
private:
    const string FUSED_OP_TYPE = "TileD";
};

}  // namespace fe

#endif  // FE_TILE_FUSION_H