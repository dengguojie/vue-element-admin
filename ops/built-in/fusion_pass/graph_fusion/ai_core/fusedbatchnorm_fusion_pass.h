/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

using std::map;
using std::string;
using std::vector;
using namespace ge;
using namespace std;

namespace fe {
struct PassRemoveEdge {
  InDataAnchorPtr inAnchorPtr;
  OutDataAnchorPtr outAnchorPtr;
};

// Match result.
struct PassMatchResult {
  NodePtr batchNormPtr;
  vector<OutDataAnchorPtr> bnInAnchorVec;
  vector<InDataAnchorPtr> bnOutAnchorVec;
  vector<NodePtr> variableNodeVec;
  vector<NodePtr> assignSubNodeVec;
  vector<NodePtr> subNodeVec;
  vector<NodePtr> mulNodeVec;
  vector<NodePtr> dataNodeVec;
  vector<OutDataAnchorPtr> dataNodeOutAnchorVec;
  vector<NodePtr> constNodeVec;
  NodePtr outNodePtr;
  vector<NodePtr> switchNodeVec;
  vector<NodePtr> fwkNodeVec;
  vector<NodePtr> castInVec;
  vector<NodePtr> castOutVec;
  vector<PassRemoveEdge> removeEdgeVec;
  vector<OutDataAnchorPtr> bnOtherOutAnchorVec;
};

class FusedBatchnormFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Run(ge::ComputeGraph& graph) override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) override;

 private:
  Status MatchPass(ge::ComputeGraph& graph, vector<PassMatchResult>& passMatchResultVec);
  Status GetAllBatchNormNodes(ge::ComputeGraph& graph, vector<NodePtr>& batchNormNodeVec);
  Status MatchBatchNormNode(NodePtr bnNodePtr, PassMatchResult& matchResult);
  Status MatchSubNode(NodePtr subNodePtr, PassMatchResult& matchResult);
  Status FusionGraphWithPass(ge::ComputeGraph& graph, PassMatchResult& matchResult);

  NodePtr FindInputNode(NodePtr nodePtr, string opType, PassMatchResult& matchResult, bool isRemoveEdge);
  NodePtr FindOutputNode(NodePtr nodePtr, string opType, PassMatchResult& matchResult, bool isRemoveEdge);
  NodePtr FindOutputNodeByName(NodePtr nodePtr, string opName, PassMatchResult& matchResult, bool isRemoveEdge);
  NodePtr FindInputNodeByIndex(NodePtr nodePtr, unsigned int index, PassMatchResult& matchResult, bool isRemoveEdge);
  Status SetOutputTensorDescAttr(uint16_t originOutputIndex, uint16_t fuseOutputIndex, ge::NodePtr originNode,
                                 ge::NodePtr fuseNode);
  const string FUSED_OP_TYPE = "BNTrainingReduce_BNTrainingUpdate";
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_FUSEDBATCHNORM_FUSION_PASS_H_
