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
 * \file conv2dbackprop_group_fusion_pass.cpp
 * \brief groups Conv2D & Conv2DBackpropInputD & Conv2DBackpropFilterD fusion
 */
#include "conv2dbackprop_group_fusion_pass.h"
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "op_log.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

namespace fe {
static const string PATTERN_CONV2DBACKPROPINPUTD = "Conv2DBackpropInputD";
static const char* CONV2DBACKPROPINPUTD = "Conv2DBackpropInputD";
static const string BATCHNORMGRAD = "BatchNormGrad";
static const string TRANSDATA = "TransData";
static const string VARIABLE = "Variable";
static const string CONV2D = "Conv2D";
static const string CONV2DBACKPROPFILTERD = "Conv2DBackpropFilterD";
static const string RELU = "Relu";
static const string RELUGRAD = "ReluGrad";
static const string ATTR_GROUPS = "groups";

/*
    fusion pattern
            nodeA   weight                nodeBgrad     nodeA
                \     / \                    / \         /
                 \   /   \                  /   \       /
                Conv2D---Conv2DBackpropInputD---Conv2DBackpropFilterD
                /           /                        |
               /           /                         |
            nodeB       nodeAgrad                   nodeC
*/
vector<FusionPattern*> Conv2dbackpropFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("Conv2dbackpropFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_CONV2DBACKPROPINPUTD, {CONV2DBACKPROPINPUTD}).SetOutput(PATTERN_CONV2DBACKPROPINPUTD);
  patterns.push_back(pattern);
  return patterns;
}

Status Conv2dbackpropFusionPass::IsMatch(ge::NodePtr& Conv2DBackpropInputNode, ge::NodePtr& Conv2DBackpropFilterNode,
                                         ge::NodePtr& Conv2DNode) {
  FUSION_PASS_CHECK(
      Conv2DBackpropInputNode->GetInAllNodes().size() != 2,
      OP_LOGI(FUSED_OP_TYPE.c_str(),
              "Node[%s]'s output not equal to two, actually is [%zu], "
              "match not success.",
              Conv2DBackpropInputNode->GetName().c_str(), Conv2DBackpropInputNode->GetInAllNodes().size()),
      return NOT_CHANGED);
  FUSION_PASS_CHECK(
      Conv2DBackpropInputNode->GetAllOutDataAnchors().size() != 1,
      OP_LOGI(FUSED_OP_TYPE.c_str(),
              "Node[%s]'s output not equal to one, actually is [%zu], "
              "match not success.",
              Conv2DBackpropInputNode->GetName().c_str(), Conv2DBackpropInputNode->GetAllOutDataAnchors().size()),
      return NOT_CHANGED);

  ge::NodePtr VariableNode = Conv2DBackpropInputNode->GetInAllNodes().at(0);
  FUSION_PASS_CHECK(VariableNode->GetType() != TRANSDATA && VariableNode->GetType() != VARIABLE,
                    OP_LOGI(FUSED_OP_TYPE.c_str(),
                            "Node[%s]'s type is not TransData or Variable, actually is "
                            "[%s], match not success.",
                            VariableNode->GetName().c_str(), VariableNode->GetType().c_str()),
                    return NOT_CHANGED);

  ge::NodePtr ConvBackFilterInNode = Conv2DBackpropInputNode->GetInAllNodes().at(1);
  for (auto BatchNormGradOutAnchor : ConvBackFilterInNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    if (BatchNormGradOutAnchor->GetOwnerNode()->GetType() == CONV2DBACKPROPFILTERD) {
      Conv2DBackpropFilterNode = BatchNormGradOutAnchor->GetOwnerNode();
      break;
    }
  }
  FUSION_PASS_CHECK(Conv2DBackpropFilterNode == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Conv2DBackpropFilterDNode is null, fusion failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(Conv2DBackpropFilterNode->GetOpDesc() == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Conv2DBackpropFilterDNode's OpDesc is null"), return NOT_CHANGED);
  FUSION_PASS_CHECK(
      Conv2DBackpropFilterNode->GetInAllNodes().size() != 2,
      OP_LOGW(FUSED_OP_TYPE.c_str(),
              "Node[%s]'s output not equal to two, actually is [%zu], "
              "match not success.",
              Conv2DBackpropFilterNode->GetName().c_str(), Conv2DBackpropFilterNode->GetInAllNodes().size()),
      return NOT_CHANGED);
  FUSION_PASS_CHECK(
      Conv2DBackpropFilterNode->GetAllOutDataAnchors().size() != 1,
      OP_LOGW(FUSED_OP_TYPE.c_str(),
              "Node[%s]'s output not equal to one, actually is [%zu], "
              "match not success.",
              Conv2DBackpropFilterNode->GetName().c_str(), Conv2DBackpropFilterNode->GetAllOutDataAnchors().size()),
      return NOT_CHANGED);

  ge::NodePtr ConvInNode = Conv2DBackpropInputNode->GetInAllNodes().at(0);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Node[%s]'s type is [%s].", ConvInNode->GetName().c_str(),
          ConvInNode->GetType().c_str());
  for (auto ReluOutAnchor : ConvInNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    if (ReluOutAnchor->GetOwnerNode()->GetType() == CONV2D) {
      Conv2DNode = ReluOutAnchor->GetOwnerNode();
      break;
    }
  }
  FUSION_PASS_CHECK(Conv2DNode == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Conv2DNode is null, fusion failed."),
                    return FAILED);
  FUSION_PASS_CHECK(Conv2DNode->GetOpDesc() == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Conv2DNode's OpDesc is null"),
                    return FAILED);
  FUSION_PASS_CHECK(Conv2DNode->GetInAllNodes().size() < 2,
                    OP_LOGW(FUSED_OP_TYPE.c_str(),
                            "Node[%s]'s output less than two, actually is [%zu], match "
                            "not success.",
                            Conv2DNode->GetName().c_str(), Conv2DNode->GetInAllNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(Conv2DNode->GetAllOutDataAnchors().size() != 1,
                    OP_LOGW(FUSED_OP_TYPE.c_str(),
                            "Node[%s]'s output not equal to one, actually is [%zu], "
                            "match not success.",
                            Conv2DNode->GetName().c_str(), Conv2DNode->GetAllOutDataAnchors().size()),
                    return NOT_CHANGED);
  return SUCCESS;
}

Status Conv2dbackpropFusionPass::GetGroups(ge::OpDescPtr srcDesc, int64_t& groups) {
  bool hasGroup = ge::AttrUtils::GetInt(srcDesc, ATTR_GROUPS, groups);
  groups = hasGroup ? groups : 1;
  FUSION_PASS_CHECK(!(groups > 0 && groups < numeric_limits<int>::max()),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Node[%s]'s attr group[%ld] is out of range, fusion failed.",
                            srcDesc->GetName().c_str(), groups),
                    return FAILED);
  return SUCCESS;
}

Status Conv2dbackpropFusionPass::CheckValidation(ge::OpDescPtr Conv2DDesc, ge::OpDescPtr Conv2DBackpropInputDesc,
                                                 ge::OpDescPtr Conv2DBackpropFilterDesc, int64_t& Conv2DGroups) {
  int64_t Conv2DBackpropInputGroups = -1, Conv2DBackpropFilterGroups = -1;
  // get attr group from conv2d, convbackpropInput and convbackpropFilter
  FUSION_PASS_CHECK(GetGroups(Conv2DDesc, Conv2DGroups) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Node[%s]'s group[%ld] is out of range, fusion failed.",
                            Conv2DDesc->GetName().c_str(), Conv2DGroups),
                    return FAILED);
  FUSION_PASS_CHECK(GetGroups(Conv2DBackpropInputDesc, Conv2DBackpropInputGroups) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Node[%s]'s group[%ld] is out of range, fusion failed.",
                            Conv2DBackpropInputDesc->GetName().c_str(), Conv2DBackpropInputGroups),
                    return FAILED);
  FUSION_PASS_CHECK(GetGroups(Conv2DBackpropFilterDesc, Conv2DBackpropFilterGroups) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Node[%s]'s group[%ld] is out of range, fusion failed.",
                            Conv2DBackpropFilterDesc->GetName().c_str(), Conv2DBackpropFilterGroups),
                    return FAILED);
  // group should be equal in conv2d, convbackpropInput and convbackpropFilter
  FUSION_PASS_CHECK(
      !(Conv2DGroups == Conv2DBackpropInputGroups && Conv2DGroups == Conv2DBackpropFilterGroups),
      OP_LOGE(FUSED_OP_TYPE.c_str(),
              "Group is not equal in node[%s]:[%ld], node[%s]:[%ld] and "
              "node[%s]:[%ld], fusion failed.",
              Conv2DDesc->GetName().c_str(), Conv2DGroups, Conv2DBackpropInputDesc->GetName().c_str(),
              Conv2DBackpropInputGroups, Conv2DBackpropFilterDesc->GetName().c_str(), Conv2DBackpropFilterGroups),
      return FAILED);
  return SUCCESS;
}

Status Conv2dbackpropFusionPass::ParseConvNodeChannelIdx(ge::GeTensorDesc& ConvTensordesc, size_t& ConvChannelIdx) {
  // Get node's channel dim idex, only support NCHW and NHWC
  ge::Format Conv2DTensorFormat = ConvTensordesc.GetFormat();
  if (Conv2DTensorFormat == ge::FORMAT_NCHW) {
    ConvChannelIdx = 1;
  } else if (Conv2DTensorFormat == ge::FORMAT_NHWC) {
    ConvChannelIdx = 3;
  } else {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Just support format NCHW & NHWC, but actually is %s",
            ge::TypeUtils::FormatToSerialString(Conv2DTensorFormat).c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status Conv2dbackpropFusionPass::ParseConvNodeNumberIdx(ge::GeTensorDesc& ConvTensordesc, size_t& ConvNumberIdx) {
  // Get node's number dim idex, only support NCHW, HWCN and NHWC
  ge::Format Conv2DTensorFormat = ConvTensordesc.GetFormat();
  if (Conv2DTensorFormat == ge::FORMAT_NCHW || Conv2DTensorFormat == ge::FORMAT_NHWC) {
    ConvNumberIdx = 0;
  } else if (Conv2DTensorFormat == ge::FORMAT_HWCN) {
    ConvNumberIdx = 3;
  } else {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Just support format NCHW & NHWC, but actually is %s",
            ge::TypeUtils::FormatToSerialString(Conv2DTensorFormat).c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status Conv2dbackpropFusionPass::ParseConvNodeChannel(ge::GeTensorDesc& ConvTensordesc, int64_t& ConvChannel) {
  // get node's channel dim number according to channel dim idx
  size_t dimIdx = -1;
  FUSION_PASS_CHECK(ParseConvNodeChannelIdx(ConvTensordesc, dimIdx) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Get node's channel index failed."), return FAILED);
  ConvChannel = ConvTensordesc.GetShape().GetDim(dimIdx);
  return SUCCESS;
}

bool Conv2dbackpropFusionPass::GenerateSplitNode(ge::ComputeGraph& graph, ge::OpDescPtr srcDesc, int64_t groups,
                                                 ge::NodePtr& splitNode, ge::GeTensorDesc& splitOutDesc, size_t dimIdx,
                                                 uint32_t anchorIdx) {
  ge::OpDescPtr splitDesc;
  // get split's input from srcNode's input
  ge::GeTensorDesc inputDesc = srcDesc->GetInputDesc(anchorIdx);
  ge::GeShape inputShape = inputDesc.GetShape();
  int64_t newInputChn = inputShape.GetDim(dimIdx);
  ge::GeShape splitOutShape = inputShape;
  FUSION_PASS_CHECK(newInputChn % groups != 0,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Node[%s]'s dim size[%ld] divide groups[%ld] != 0",
                            srcDesc->GetName().c_str(), newInputChn, groups),
                    return false);
  splitOutShape.SetDim(dimIdx, newInputChn / groups);
  // create split node
  FUSION_PASS_MAKE_SHARED((splitDesc = std::make_shared<ge::OpDesc>(srcDesc->GetName() + "_split", "SplitD")),
                          return false);
  // set attr for split node
  ge::AttrUtils::SetInt(splitDesc, "split_dim", dimIdx);
  ge::AttrUtils::SetInt(splitDesc, "num_split", groups);
  // get split's output from input, except C axis should be C/groups
  splitOutDesc = inputDesc;
  splitOutDesc.Update(splitOutShape, inputDesc.GetFormat(), inputDesc.GetDataType());
  splitOutDesc.SetOriginShape(splitOutShape);
  (void)splitDesc->AddInputDesc(inputDesc);
  for (int i = 0; i < groups; i++) {
    (void)splitDesc->AddOutputDesc(splitOutDesc);
  }
  splitNode = graph.AddNode(splitDesc);
  return true;
}

bool Conv2dbackpropFusionPass::GenerateNewConvNodes(ge::ComputeGraph& graph, ge::OpDescPtr srcDesc,
                                                    const ge::GeTensorDesc& splitOutDesc,
                                                    vector<ge::NodePtr>& newConvNodes, ge::GeTensorDesc& newConvOutDesc,
                                                    int64_t groups, size_t dimIdx, uint32_t anchorIdx) {
  // set new convNodes' 1st input from split
  ge::GeTensorDesc inDesc = splitOutDesc;
  inDesc.SetOriginDataType(srcDesc->GetInputDesc(anchorIdx).GetOriginDataType());
  inDesc.SetDataType(srcDesc->GetInputDesc(anchorIdx).GetDataType());
  // set new convNodes' output from srcNode, but channel should divide groups
  ge::GeTensorDesc outputDesc = srcDesc->GetOutputDesc(0);
  ge::GeShape newConvOutShape = outputDesc.GetShape();
  int64_t newConvOutChn = newConvOutShape.GetDim(dimIdx);
  FUSION_PASS_CHECK(newConvOutChn % groups != 0,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Node[%s]'s dim size[%ld] divide groups[%ld] != 0",
                            srcDesc->GetName().c_str(), newConvOutShape.GetDim(dimIdx), groups),
                    return false);
  newConvOutShape.SetDim(dimIdx, newConvOutChn / groups);
  newConvOutDesc = outputDesc;
  newConvOutDesc.Update(newConvOutShape, srcDesc->GetOutputDesc(0).GetFormat(),
                        srcDesc->GetOutputDesc(0).GetDataType());
  newConvOutDesc.SetOriginShape(newConvOutShape);
  newConvOutDesc.SetOriginDataType(srcDesc->GetOutputDesc(0).GetOriginDataType());
  // create new conv nodes from srcNode
  for (int64_t i = 0; i < groups; i++) {
    ostringstream newConvName;
    newConvName << srcDesc->GetName() << "_" << i;
    ge::OpDescPtr newConvDesc = ge::AttrUtils::CopyOpDesc(srcDesc);
    FUSION_PASS_CHECK(newConvDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.",
                                                      srcDesc->GetName().c_str()),
                      return PARAM_INVALID);
    newConvDesc->SetName(newConvName.str());
    newConvDesc->UpdateInputDesc(anchorIdx, inDesc);
    for (unsigned int j = 1; j < srcDesc->GetAllInputsDesc().size(); j++) {
      newConvDesc->UpdateInputDesc(j, inDesc);
    }
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != newConvDesc->UpdateOutputDesc(0, newConvOutDesc),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Update node:%s's 1st output failed.", srcDesc->GetName().c_str()),
                      return false);
    ge::AttrUtils::SetInt(newConvDesc, "groups", 1);
    ge::NodePtr newConvNode = graph.AddNode(newConvDesc);
    newConvNodes.push_back(newConvNode);
  }
  return true;
}

bool Conv2dbackpropFusionPass::GenerateConcatNode(ge::ComputeGraph& graph, ge::OpDescPtr srcDesc, int64_t groups,
                                                  ge::GeTensorDesc& newConvOutDesc, ge::NodePtr& concatNode,
                                                  size_t dimIdx) {
  // create concat node, which input equal to new convNodes's output
  ge::OpDescPtr concatDesc;
  FUSION_PASS_MAKE_SHARED((concatDesc = std::make_shared<ge::OpDesc>(srcDesc->GetName() + "_concat", "ConcatD")),
                          return false);
  for (int i = 0; i < groups; i++) {
    (void)concatDesc->AddInputDesc(newConvOutDesc);
  }
  (void)concatDesc->AddOutputDesc(srcDesc->GetOutputDesc(0));
  ge::AttrUtils::SetInt(concatDesc, "concat_dim", dimIdx);  // c axis concat
  concatNode = graph.AddNode(concatDesc);
  return true;
}

bool Conv2dbackpropFusionPass::GenerateNewNodes(ge::ComputeGraph& graph, ge::OpDescPtr srcDesc, ge::NodePtr& splitNode,
                                                vector<ge::NodePtr>& newConvNodes, ge::NodePtr& concatNode,
                                                int64_t Conv2DGroups, size_t dimChannelIdx, uint32_t anchorIdx) {
  // create new nodes : groups conv and convBackprop, 2 split, 3 concat
  ge::GeTensorDesc splitOutDesc;
  ge::GeTensorDesc newConvOutDesc;
  if (srcDesc->GetType() != "CONV2DBACKPROPFILTERD") {
    FUSION_PASS_CHECK(
        !GenerateSplitNode(graph, srcDesc, Conv2DGroups, splitNode, splitOutDesc, dimChannelIdx, anchorIdx),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "generate split node before node[%s] failed.", srcDesc->GetName().c_str()),
        return false);
  }
  splitOutDesc = splitNode->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(
      !GenerateNewConvNodes(graph, srcDesc, splitOutDesc, newConvNodes, newConvOutDesc, Conv2DGroups, dimChannelIdx,
                            anchorIdx),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "generate new conv nodes for node[%s] failed.", srcDesc->GetName().c_str()),
      return false);
  FUSION_PASS_CHECK(
      !GenerateConcatNode(graph, srcDesc, Conv2DGroups, newConvOutDesc, concatNode, dimChannelIdx),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "generate concat node after node[%s] failed.", srcDesc->GetName().c_str()),
      return false);
  return true;
}

bool Conv2dbackpropFusionPass::Relink(ge::NodePtr srcNode, ge::NodePtr splitNode, vector<ge::NodePtr>& newConvNodes,
                                      ge::NodePtr concatNode, int64_t Conv2DGroups, uint32_t anchorIdx) {
  ge::Node::Vistor<ge::NodePtr> convOutNodes = srcNode->GetOutAllNodes();
  vector<int> outAnchorIndexes;
  for (size_t i = 0; i < convOutNodes.size(); i++) {
    outAnchorIndexes.push_back(srcNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(i)->GetIdx());
  }
  for (auto outAnchor : srcNode->GetAllOutDataAnchors()) {
    if (outAnchor != nullptr) {
      outAnchor->UnlinkAll();
    }
  }
  if (splitNode->GetInDataAnchor(0) == nullptr) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Node's inanchor is empty.");
  }
  if (splitNode->GetInDataAnchor(0)->GetPeerOutAnchor() == nullptr) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Node's peeroutanchor is empty.");
  }
  // link split to new conv nodes
  for (int i = 0; i < Conv2DGroups; i++) {
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != ge::GraphUtils::AddEdge(splitNode->GetOutDataAnchor(i),
                                                                   newConvNodes[i]->GetInDataAnchor(anchorIdx)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add slice to conv edge fail"), return false);
    // link new conv nodes to concat
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS !=
                          ge::GraphUtils::AddEdge(newConvNodes[i]->GetOutDataAnchor(0), concatNode->GetInDataAnchor(i)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add conv to concat edge fail"), return false);
  }
  // link concat node to old conv's output
  for (size_t i = 0; i < convOutNodes.size(); i++) {
    FUSION_PASS_CHECK(
        ge::GRAPH_SUCCESS != ge::GraphUtils::AddEdge(concatNode->GetOutAnchor(0),
                                                     convOutNodes.at(i)->GetInDataAnchor(outAnchorIndexes[i])),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add concat to output edge fail"), return false);
  }
  return true;
}
// unlink old nodes's edges, and delete old nodes
bool Conv2dbackpropFusionPass::RemoveOldNode(ge::ComputeGraph& graph, ge::NodePtr srcNode) {
  for (auto inAnchor : srcNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  FUSION_PASS_CHECK(SUCCESS != graph.RemoveNode(srcNode),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "RemoveNode %s failed.", srcNode->GetName().c_str()), return false);
  return true;
}
// set conv and convBackprop nodes to depthwiseConv and depthwiseConvBackprop
Status Conv2dbackpropFusionPass::ProcessDepthwiseConv(ge::OpDescPtr& Conv2DDesc, ge::OpDescPtr& Conv2DBackpropInputDesc,
                                                      ge::OpDescPtr& Conv2DBackpropFilterDesc, int64_t Conv2DGroups) {
  Conv2DBackpropInputDesc->SetType("DepthwiseConv2DBackpropInputD");
  Conv2DBackpropFilterDesc->SetType("DepthwiseConv2DBackpropFilterD");
  Conv2DDesc->SetType("DepthwiseConv2D");
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(Conv2DDesc, ATTR_GROUPS, 1),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "change group failed"), return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(Conv2DBackpropInputDesc, ATTR_GROUPS, 1),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "change group failed"), return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(Conv2DBackpropFilterDesc, ATTR_GROUPS, 1),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "change group failed"), return FAILED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Success to do Depthwise fusion.");
  return SUCCESS;
}

Status Conv2dbackpropFusionPass::ProcessGroupConvFusion(ge::ComputeGraph& graph, ge::NodePtr Conv2DNode,
                                                        ge::NodePtr Conv2DBackpropInputNode,
                                                        ge::NodePtr Conv2DBackpropFilterNode, int64_t Conv2DGroups) {
  ge::OpDescPtr Conv2DDesc = Conv2DNode->GetOpDesc();
  ge::OpDescPtr Conv2DBackpropInputDesc = Conv2DBackpropInputNode->GetOpDesc();
  ge::OpDescPtr Conv2DBackpropFilterDesc = Conv2DBackpropFilterNode->GetOpDesc();
  ge::NodePtr splitNode;
  ge::NodePtr splitNode1;
  vector<ge::NodePtr> newConvNodes;
  vector<ge::NodePtr> newConvNodes1;
  vector<ge::NodePtr> newConvNodes2;
  ge::NodePtr concatNode;
  ge::NodePtr concatNode1;
  ge::NodePtr concatNode2;
  size_t dimChannelIdx = -1, dimNumberIdx = -1;
  ge::GeTensorDesc convInputDesc0 = Conv2DDesc->GetInputDesc(0);
  ge::GeTensorDesc convInputDesc1 = Conv2DDesc->GetInputDesc(1);
  FUSION_PASS_CHECK(
      ParseConvNodeChannelIdx(convInputDesc0, dimChannelIdx) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Get node[%s]'s channel index failed.", Conv2DNode->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      ParseConvNodeNumberIdx(convInputDesc1, dimNumberIdx) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Get node[%s]'s number index failed.", Conv2DNode->GetName().c_str()),
      return FAILED);
  if (PatternFusionUtil::IsUnknownShape(Conv2DDesc->GetInputDesc(0).GetShape().GetDim(dimChannelIdx)) ||
      PatternFusionUtil::IsUnknownShape(Conv2DDesc->GetOutputDesc(0).GetShape().GetDim(dimChannelIdx)) ||
      PatternFusionUtil::IsUnknownShape(Conv2DBackpropInputDesc->GetInputDesc(1).GetShape().GetDim(dimChannelIdx)) ||
      PatternFusionUtil::IsUnknownShape(Conv2DBackpropInputDesc->GetOutputDesc(0).GetShape().GetDim(dimChannelIdx)) ||
      PatternFusionUtil::IsUnknownShape(Conv2DBackpropFilterDesc->GetOutputDesc(0).GetShape().GetDim(dimChannelIdx))
      ) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Conv2dbackpropFusionPass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }
  FUSION_PASS_CHECK(
      !GenerateNewNodes(graph, Conv2DDesc, splitNode, newConvNodes, concatNode, Conv2DGroups, dimChannelIdx, 0),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Generate new nodes for node[%s] failed.", Conv2DDesc->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(!GenerateNewNodes(graph, Conv2DBackpropInputDesc, splitNode1, newConvNodes1, concatNode1,
                                      Conv2DGroups, dimChannelIdx, 1),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Generate new nodes for node[%s] failed.",
                            Conv2DBackpropInputDesc->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(!GenerateNewNodes(graph, Conv2DBackpropFilterDesc, splitNode1, newConvNodes2, concatNode2,
                                      Conv2DGroups, dimChannelIdx, 0),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Generate new nodes for node[%s] failed.",
                            Conv2DBackpropFilterDesc->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(!Relink(Conv2DNode, splitNode, newConvNodes, concatNode, Conv2DGroups, 0),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "relink fail"), return FAILED);
  FUSION_PASS_CHECK(!Relink(Conv2DBackpropInputNode, splitNode1, newConvNodes1, concatNode1, Conv2DGroups, 1),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "relink fail"), return FAILED);
  FUSION_PASS_CHECK(!Relink(Conv2DBackpropFilterNode, splitNode1, newConvNodes2, concatNode2, Conv2DGroups, 1),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "relink fail"), return FAILED);
  // link conv's input node to split node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != ge::GraphUtils::AddEdge(Conv2DNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                                 splitNode->GetInDataAnchor(0)),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add data to slice edge fail"), return false);
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != ge::GraphUtils::AddEdge(Conv2DBackpropInputNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                                   splitNode1->GetInDataAnchor(0)),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "add data to slice edge fail"), return false);
  for (int i = 0; i < Conv2DGroups; i++) {
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS !=
                          ge::GraphUtils::AddEdge(splitNode->GetOutDataAnchor(i), newConvNodes2[i]->GetInDataAnchor(0)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add slice to conv edge fail"), return false);
  }
  FUSION_PASS_CHECK(
      PatternFusionUtil::InsertSliceDNodes(graph, Conv2DNode, 1, newConvNodes, Conv2DGroups, dimNumberIdx) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Insert SliceD node between node[%s] and weight node falied.",
              Conv2DNode->GetName().c_str()),
      return FAILED);

  if (Conv2DNode->GetOpDesc()->MutableInputDesc(2) != nullptr) {
    FUSION_PASS_CHECK(
      PatternFusionUtil::InsertSliceDNodes(graph, Conv2DNode, 2, newConvNodes, Conv2DGroups, 0) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Insert SliceD node between node[%s] and bias node falied.",
                Conv2DNode->GetName().c_str()),
        return FAILED);
  }
  FUSION_PASS_CHECK(PatternFusionUtil::InsertSliceDNodes(graph, Conv2DBackpropInputNode, 0, newConvNodes1, Conv2DGroups,
                                                         dimNumberIdx) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Insert SliceD node between node[%s] and weight node falied.",
                            Conv2DNode->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(!RemoveOldNode(graph, Conv2DNode),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "RemoveNode %s failed.", Conv2DNode->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(!RemoveOldNode(graph, Conv2DBackpropInputNode),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "RemoveNode %s failed.", Conv2DBackpropInputNode->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(
      !RemoveOldNode(graph, Conv2DBackpropFilterNode),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "RemoveNode %s failed.", Conv2DBackpropFilterNode->GetName().c_str()),
      return FAILED);
  return SUCCESS;
}

Status Conv2dbackpropFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  ge::NodePtr Conv2DBackpropInputNode = GetNodeFromMapping(PATTERN_CONV2DBACKPROPINPUTD, mapping);
  FUSION_PASS_CHECK(Conv2DBackpropInputNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Conv2DBackpropInputDNode is null, fusion failed."),
                    return PARAM_INVALID);

  ge::NodePtr Conv2DNode = nullptr;
  ge::NodePtr Conv2DBackpropFilterNode = nullptr;
  // check whether there is conv node and convBackprop nodes in the graph
  if (IsMatch(Conv2DBackpropInputNode, Conv2DBackpropFilterNode, Conv2DNode) != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "Conv2D + Conv2DBackpropFilterD + Conv2DBackpropInputD fusion "
            "pattern don't matched");
    return NOT_CHANGED;
  }

  ge::OpDescPtr Conv2DBackpropInputDesc = Conv2DBackpropInputNode->GetOpDesc();
  FUSION_PASS_CHECK(Conv2DBackpropInputDesc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Conv2DBackpropInputDNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(Conv2DBackpropFilterNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Conv2DBackpropFilterDNode is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr Conv2DBackpropFilterDesc = Conv2DBackpropFilterNode->GetOpDesc();
  FUSION_PASS_CHECK(Conv2DBackpropFilterDesc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Conv2DBackpropFilterDNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(Conv2DNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Conv2DNode is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr Conv2DDesc = Conv2DNode->GetOpDesc();
  FUSION_PASS_CHECK(Conv2DDesc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Conv2DNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  // check the attr groups should be the same in conv node and convBackprop nodes
  int64_t Conv2DGroups = -1;
  FUSION_PASS_CHECK(
      CheckValidation(Conv2DDesc, Conv2DBackpropInputDesc, Conv2DBackpropFilterDesc, Conv2DGroups) != SUCCESS,
      OP_LOGW(FUSED_OP_TYPE.c_str(), "Check node[%s], node[%s] and node[%s]'s validation failed.",
              Conv2DDesc->GetName().c_str(), Conv2DBackpropInputDesc->GetName().c_str(),
              Conv2DBackpropFilterDesc->GetName().c_str()),
      return NOT_CHANGED);
  // if attr group equals to 1, no need to do fusion
  if (Conv2DGroups == 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "Group is one in node[%s], node[%s] and node[%s], no need to do "
            "fusion.",
            Conv2DDesc->GetName().c_str(), Conv2DBackpropInputDesc->GetName().c_str(),
            Conv2DBackpropFilterDesc->GetName().c_str());
    return NOT_CHANGED;
  }

  ge::GeTensorDesc Conv2DInanchor0 = Conv2DDesc->GetInputDesc(0);
  ge::GeTensorDesc Conv2DOutanchor0 = Conv2DDesc->GetOutputDesc(0);
  int64_t Conv2DInputChannel = -1;
  int64_t Conv2DOutputChannel = -1;
  if (ParseConvNodeChannel(Conv2DInanchor0, Conv2DInputChannel) != SUCCESS ||
      ParseConvNodeChannel(Conv2DOutanchor0, Conv2DOutputChannel) != SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Parse node[%s]'s channel info not success.", Conv2DDesc->GetName().c_str());
    return NOT_CHANGED;
  }
  if (PatternFusionUtil::IsUnknownShape(Conv2DInputChannel) ||
      PatternFusionUtil::IsUnknownShape(Conv2DOutputChannel)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Conv2dbackpropFusionPass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }
  // if conv's cin = cout, replace conv and convBackprop nodes to depthwise
  if (Conv2DGroups == Conv2DOutputChannel && Conv2DGroups == Conv2DInputChannel) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]'s InputChannel[%ld], OutputChannel[%ld] and group is [%ld]",
            Conv2DDesc->GetName().c_str(), Conv2DInputChannel, Conv2DOutputChannel, Conv2DGroups);
    ProcessDepthwiseConv(Conv2DDesc, Conv2DBackpropInputDesc, Conv2DBackpropFilterDesc, Conv2DGroups);
  } else if (Conv2DInputChannel % Conv2DGroups == 0 && Conv2DOutputChannel % Conv2DGroups == 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]'s InputChannel[%ld], OutputChannel[%ld] and group is [%ld]",
            Conv2DDesc->GetName().c_str(), Conv2DInputChannel, Conv2DOutputChannel, Conv2DGroups);
    // split conv and convBackprop nodes into groups
    if (ProcessGroupConvFusion(graph, Conv2DNode, Conv2DBackpropInputNode, Conv2DBackpropFilterNode, Conv2DGroups)
                                                                                                  == NOT_CHANGED) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "Conv2dbackpropFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
  } else {
    OP_LOGW(FUSED_OP_TYPE.c_str(),
            "InputChannel[%ld] or OutputChannel[%ld] divide group[%ld] not "
            "equal to zero",
            Conv2DInputChannel, Conv2DOutputChannel, Conv2DGroups);
    return NOT_CHANGED;
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv2D + Conv2DBackpropFilterD + Conv2DBackpropInputD fusion success!");
  return SUCCESS;
}
REGISTER_PASS("AConv2dbackpropFusionPass", BUILT_IN_GRAPH_PASS, Conv2dbackpropFusionPass);
}  // namespace fe
