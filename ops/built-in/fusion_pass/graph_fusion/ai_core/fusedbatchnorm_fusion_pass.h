/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @file  fusedbatchnorm_fusion_pass.h
 *
 * @brief fusedbatchnorm fusion pass(BatchNorm --> BNTrainingReduce & BNTrainingUpdate)
 *
 */
#ifndef FE_FUSEDBATCHNORM_FUSION_PASS_H
#define FE_FUSEDBATCHNORM_FUSION_PASS_H

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
  vector<FusionPattern *> DefinePatterns() override;
  Status Run(ge::ComputeGraph &graph) override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;

 private:
  Status MatchPass(ge::ComputeGraph &graph,
                   vector<PassMatchResult> &passMatchResultVec);
  Status GetAllBatchNormNodes(ge::ComputeGraph &graph,
                              vector<NodePtr> &batchNormNodeVec);
  Status MatchBatchNormNode(NodePtr bnNodePtr, PassMatchResult &matchResult);
  Status MatchSubNode(NodePtr subNodePtr, PassMatchResult &matchResult);
  Status FusionGraphWithPass(ge::ComputeGraph &graph,
                             PassMatchResult &matchResult);

  NodePtr FindInputNode(NodePtr nodePtr, string opType,
                        PassMatchResult &matchResult, bool isRemoveEdge);
  NodePtr FindOutputNode(NodePtr nodePtr, string opType,
                         PassMatchResult &matchResult, bool isRemoveEdge);
  NodePtr FindOutputNodeByName(NodePtr nodePtr, string opName,
                               PassMatchResult &matchResult,
                               bool isRemoveEdge);
  NodePtr FindInputNodeByIndex(NodePtr nodePtr, unsigned int index,
                               PassMatchResult &matchResult, bool isRemoveEdge);
  Status SetOutputTensorDescAttr(uint16_t originOutputIndex,
                                 uint16_t fuseOutputIndex,
                                 ge::NodePtr originNode,
                                 ge::NodePtr fuseNode);
  const string FUSED_OP_TYPE = "BNTrainingReduce_BNTrainingUpdate";
};
}  // namespace fe
#endif  // FE_FUSEDBATCHNORM_FUSION_PASS_H
