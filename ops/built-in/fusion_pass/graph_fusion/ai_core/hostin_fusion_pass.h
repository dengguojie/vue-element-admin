/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief bnhost fusion pass
 *
 */

#ifndef INHOST_FUSION_PASS_H
#define INHOST_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
    class HostINFusionPass : public PatternFusionBasePass {

    protected:
        vector<FusionPattern*> DefinePatterns() override;

        Status Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                      vector<ge::NodePtr>& newNodes) override;

    private:

        /**
         * Do SwapCo fusion for PSROIPooling
         * @param graph: original graph info
         * @param convNodePtr: instance norm node info
         * @param newNodes: new nodes after fusion
         * @return SUCCESS/FAILED
         */
        Status INFuison(ge::ComputeGraph& graph,
                        ge::NodePtr& inNodePtr, vector<ge::NodePtr>& newNodes);

        /**
         * Check parameters of bn right or not
         * @param inNodePtr: bn node
         * @return SUCCESS/FAILED
         */
        Status CheckParameter(ge::NodePtr& inNodePtr);

        /**
         * Set output_dim and group_size attr value
         * @param newNodePtr: new node
         * @return SUCCESS/FAILED
         */
        Status SetAttrValueForNewNode(const ge::OpDescPtr& psroiOpDescPtr,
                                      ge::OpDescPtr& newOpDescPtr);

        const string FUSED_OP_TYPE = "INInferV2D";
    };
} // namespace fe

#endif // INHOST_FUSION_PASS_H
