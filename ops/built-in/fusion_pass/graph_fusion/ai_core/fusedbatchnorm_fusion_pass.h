/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file fusedbatchnorm_fusion_pass.h
 * \brief fusedbatchnorm fusion pass(BatchNorm --> BNTrainingReduce & BNTrainingUpdate)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_FUSEDBATCHNORM_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_FUSEDBATCHNORM_FUSION_PASS_H_

#include <vector>
#include <string>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
    struct PassRemoveEdge {
        ge::InDataAnchorPtr inAnchorPtr;
        ge::OutDataAnchorPtr outAnchorPtr;
};

// Match result.
    struct PassMatchResult {
        ge::NodePtr batchNormPtr;
        std::vector<ge::OutDataAnchorPtr> bnInAnchorVec;
        std::vector<ge::InDataAnchorPtr> bnOutAnchorVec;
        std::vector<ge::NodePtr> variableNodeVec;
        std::vector<ge::NodePtr> assignSubNodeVec;
        std::vector<ge::NodePtr> subNodeVec;
        std::vector<ge::NodePtr> mulNodeVec;
        std::vector<ge::NodePtr> dataNodeVec;
        std::vector<ge::OutDataAnchorPtr> dataNodeOutAnchorVec;
        std::vector<ge::NodePtr> constNodeVec;
        ge::NodePtr outNodePtr;
        std::vector<ge::NodePtr> switchNodeVec;
        std::vector<ge::NodePtr> fwkNodeVec;
        std::vector<ge::NodePtr> castInVec;
        std::vector<ge::NodePtr> castOutVec;
        std::vector<PassRemoveEdge> removeEdgeVec;
        std::vector<ge::OutDataAnchorPtr> bnOtherOutAnchorVec;
};

class FusedBatchnormFusionPass : public PatternFusionBasePass {
    public:
    FusedBatchnormFusionPass() {
        FUSED_OP_TYPE = "BNTrainingReduce_BNTrainingUpdate";
        PASS_OP_TYPE_BATCHNORM = "BatchNorm";
        PASS_OP_TYPE_SUB = "Sub";
        PASS_OP_TYPE_BNREDUCE = "BNTrainingReduce";
        PASS_OP_TYPE_BNUPDATE = "BNTrainingUpdate";
        STREAM_LABEL = "_stream_label";
    }

    ~FusedBatchnormFusionPass() override {
    }
    protected:
    std::vector<FusionPattern*> DefinePatterns() override;
    Status Run(ge::ComputeGraph& graph) override;
    Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, std::vector<ge::NodePtr>& fusionNodes) override;

    Status MatchPass(ge::ComputeGraph& graph, std::vector<PassMatchResult>& passMatchResultVec);
    Status GetAllBatchNormNodes(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& batchNormNodeVec);
    Status MatchBatchNormNode(ge::NodePtr bnNodePtr, PassMatchResult& matchResult);
    Status MatchSubNode(ge::NodePtr subNodePtr, PassMatchResult& matchResult);
    Status FusionGraphWithPass(ge::ComputeGraph& graph, PassMatchResult& matchResult);

    ge::NodePtr FindInputNode(ge::NodePtr nodePtr, std::string opType, PassMatchResult& matchResult, bool isRemoveEdge);
    ge::NodePtr FindOutputNode(ge::NodePtr nodePtr, std::string opType, PassMatchResult& matchResult, bool isRemoveEdge);
    ge::NodePtr FindOutputNodeByName(ge::NodePtr nodePtr, std::string opName, PassMatchResult& matchResult, bool isRemoveEdge);
    ge::NodePtr FindInputNodeByIndex(ge::NodePtr nodePtr, unsigned int index, PassMatchResult& matchResult, bool isRemoveEdge);
    Status SetOutputTensorDescAttr(uint16_t originOutputIndex, uint16_t fuseOutputIndex, ge::NodePtr originNode,
                                   ge::NodePtr fuseNode);

    std::string FUSED_OP_TYPE;
    std::string PASS_OP_TYPE_BATCHNORM;
    std::string PASS_OP_TYPE_SUB;

    std::string PASS_OP_TYPE_BNREDUCE;
    std::string PASS_OP_TYPE_BNUPDATE;
    std::string STREAM_LABEL;
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_FUSEDBATCHNORM_FUSION_PASS_H_
