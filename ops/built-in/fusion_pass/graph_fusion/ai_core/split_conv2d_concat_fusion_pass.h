/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief convert split+conv2d+concat to group conv2d
 *
 */

#ifndef FE_SPLIT_CONV2D_CONCAT_FUSION_H
#define FE_SPLIT_CONV2D_CONCAT_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {

class SplitConv2dConcatPass : public PatternFusionBasePass {
protected:
    vector<FusionPattern *> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                  vector<ge::NodePtr> &newNodes) override;

private:
    bool AddConcatDesc(ge::NodePtr &splitNode, ge::NodePtr &ccatNode,
                       std::vector<ge::OpDescPtr> &constDesc);
    bool LinkGroupConv2d( ge::NodePtr &groupConv,  ge::NodePtr &splitNode,  ge::NodePtr &ccatNode,
                         std::vector< ge::NodePtr> &constCcat);
    bool AnalyzeMidLayer(ge::Node::Vistor<ge::NodePtr> &sptOutput,
                         ge::OpDescPtr &convGpDesc);
    bool VerifySptCcatAxis(ge::OpDescPtr &convDesc, ge::NodePtr &splitNode,
                           ge::NodePtr &ccatNode);
    bool LinkNewConcat(ge::ComputeGraph &graph, ge::NodePtr &splitNode,
            std::vector<ge::NodePtr> &constCcat, std::vector<ge::NodePtr> &constDim);
    bool UpdateConv2dDesc(ge::OpDescPtr &convDesc, ge::NodePtr &splitNode,
                          ge::NodePtr &ccatNode);
    const string FUSED_OP_TYPE = "Split_Conv2D_ConcatV2";
};

} // namespace fe
#endif
