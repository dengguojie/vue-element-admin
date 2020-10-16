/**
 * @file a_remove_useless_node_fusion_pass.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @brief remove useless node framework code
 *
 * @version 1.0
 *
 */

#ifndef FE_REMOVE_NODE_H
#define FE_REMOVE_NODE_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include <map>
#include <vector>
#include "fusion_removenode_registry.h"

namespace fe {

using ge::NodePtr;
using ge::InDataAnchorPtr;
using ge::OutDataAnchorPtr;
using ge::InControlAnchorPtr;
using ge::OutControlAnchorPtr;

struct RemoveEdgePair {
  InDataAnchorPtr inAnchorPtr;
  OutDataAnchorPtr outAnchorPtr;
};

struct InAnchorRelatedInfo {
  vector<NodePtr> removeNode;
  vector<RemoveEdgePair> removeEdgeVec;
};

class RemoveNodeFusionPass : public PatternFusionBasePass {
 protected:
  Status Run(ge::ComputeGraph &graph) override;
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping& mapping,
                vector<NodePtr> &newNodes) override;

 private:
  /**
   * remove single node
   * @param node node ptr
   * @param linkIndexPairVec can not be reference for this value may be modified
   * @param graph graph object
   * @return SUCCESS, NOT_CHANGED, FAILED
   */
  Status RemoveNode(NodePtr node,
                    vector<LinkIndexPair> linkIndexPairVec,
                    ge::ComputeGraph &graph);

  /**
   * get linked input and output anchor mapping
   * @param node node ptr
   * @param inDataAnchorMap input index and inDataAnchor mapping
   * @param outDataAnchorMap output index nad outDataAnchor mapping
   * @param inputOutDataAnchorMap
   *                          input index and outDataAnchor of other end mapping
   * @param outputInDataAnchorMap
   *                          output index and inDataAnchor of other end mapping
   */
  void GetLinkedDataAnchor(NodePtr node,
          map<uint32_t, InDataAnchorPtr>& inDataAnchorMap,
          map<uint32_t, OutDataAnchorPtr>& outDataAnchorMap,
          map<uint32_t, OutDataAnchorPtr>& inputOutDataAnchorMap,
          map<uint32_t, vector<InDataAnchorPtr>>& outputInDataAnchorMap);

  /**
   * verify the link info
   * @param inDataAnchorMap input index and inDataAnchor mapping
   * @param outDataAnchorMap output index nad outDataAnchor mapping
   * @param linkIndexPairVec link info from op register info
   * @return true/false
   */
  bool VerifyLinkPair(map<uint32_t, InDataAnchorPtr>& inDataAnchorMap,
                      map<uint32_t, OutDataAnchorPtr>& outDataAnchorMap,
                      vector<LinkIndexPair>& linkIndexPairVec);

  /**
   * If there are some unlinked inDataAnchor,
   * verify whether these inDataAnchors can be removed
   * then remove them
   * @param inDataAnchorMap input index and inDataAnchor mapping
   * @param linkIndexPairVec the link info
   * @param graph graph object reference
   * @return SUCCESS, NOT_CHANGED, FAILED
   */
  Status HandleUnlinkedInAnchor(map<uint32_t, InDataAnchorPtr>& inDataAnchorMap,
                                vector<LinkIndexPair>& linkIndexPairVec,
                                ge::ComputeGraph &graph);
  /**
   * verify whether this inDataAnchor can be removed
   * @param inDataAnchorPtr inDataAnchor pointer
   * @param relatedInfo collect information for remove operation
   * @return true/false
   */
  bool VerifyRemoveInAnchor(InDataAnchorPtr inDataAnchorPtr,
                            InAnchorRelatedInfo &relatedInfo);

  /**
   * Re-link control anchor of the removing node
   * @param node node pointer
   * @return SUCCESS, FAILED
   */
  Status ReLinkControlAnchor(NodePtr node);

  /**
   * Add control edge
   * @param outCtrlAnchorPtr output control anchor pointer
   * @param inCtrlAnchorPtr input control anchor pointer
   * @return SUCCESS, FAILED
   */
  Status AddCtrlEdge(OutControlAnchorPtr outCtrlAnchorPtr,
                     InControlAnchorPtr inCtrlAnchorPtr);

  /**
   * Re-link data anchors of the removing node
   * @param linkIndexPairVec link info from op register info
   * @param inDataAnchorMap input index and inDataAnchor mapping
   * @param outDataAnchorMap output index nad outDataAnchor mapping
   * @param inputOutDataAnchorMap
   *                          input index and outDataAnchor of other end mapping
   * @param outputInDataAnchorMap
   *                          output index and inDataAnchor of other end mapping
   * @return SUCCESS, FAILED
   */
  Status ReLinkDataAnchor(vector<LinkIndexPair>& linkIndexPairVec,
                map<uint32_t, InDataAnchorPtr>& inDataAnchorMap,
                map<uint32_t, OutDataAnchorPtr>& outDataAnchorMap,
                map<uint32_t, OutDataAnchorPtr>& inputOutDataAnchorMap,
                map<uint32_t, vector<InDataAnchorPtr>>& outputInDataAnchorMap);
};
/*
 * Check whether the second input of GatherNd, which named "indice", is a
 * "zero-tensor".
 * The "zero-tensor" means the last dim in tensor is 0 and all other dims
 * are 1.
 * eg. "zero-tensor": [0]; [1,0]; [1,1,0]; [1,1,1,0]
 * normal tensor: [3,2]; [0,1,2]; [0,0,3,2]; [1,1,0,0]
 * If "indice" gets an "zero-tensor", return true. Otherwise, return false.
 * @param node node pointer
 * @return true, false
 */
bool GatherNdPreCheck(NodePtr node);
}  // namespace fe

#endif  // FE_REMOVE_NODE_H