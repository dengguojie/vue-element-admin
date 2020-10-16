/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief bnhost fusion pass
 *
 */

#ifndef SOFTMAX_TRANSPOSE_FUSION_PASS_H
#define SOFTMAX_TRANSPOSE_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
    class softmaxTransFusionPass : public PatternFusionBasePass {

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
                                      ge::OpDescPtr& newOpDescPtr,
                                      int64_t shapeLens);

        /**
         * Get new input attrs for OpDescPtr
         * @param currentOpDescPtr: current op desc
         * @param shapeLens: the attr shape lens
         * @param transfer: 1 for trans axis -1 and -2
         * @return SUCCESS/FAILED
         */

        Status SetAttrValue(
            const ge::OpDescPtr& OpDescPtr, int64_t shapeLens, int32_t transfer);
        const string FUSED_OP_TYPE = "TransposeD_SoftmaxV2_TransposeD";
    };
} // namespace fe

#endif // SOFTMAX_TRANSPOSE_FUSION_PASS_H
