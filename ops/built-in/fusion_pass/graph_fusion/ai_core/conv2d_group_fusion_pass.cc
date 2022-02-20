/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file conv2d_group_fusion_pass.cpp
 * \brief conv2d group fusion pass(conv2d --> conv2d/splited conv2d/depthwise conv2d)
 */
#include "conv2d_group_fusion_pass.h"
#include <vector>
#include <sstream>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "common/util/error_manager/error_manager.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;

namespace fe {
const string PATTERN_CONV2D_ID = "conv2d_group_id";
const string CONV2D_TYPE = "Conv2D";
const string ATTR_GROUPS = "groups";
const int MAX_DIM_NUM = 4;

enum { DIM_N = 0, DIM_C = 1, DIM_H = 2, DIM_W = 3 };

vector<FusionPattern*> Conv2DGroupFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("Conv2DGroupFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
  return patterns);
  pattern->AddOpDesc(PATTERN_CONV2D_ID, {CONV2D_TYPE}).SetOutput(PATTERN_CONV2D_ID);
  patterns.push_back(pattern);
  return patterns;
}

Status Conv2DGroupFusionPass::SwapNumChn(OpDescPtr opDesc, bool bInput, uint32_t index) {
  ge::GeTensorDesc tensorDesc;
  if (bInput) {
    tensorDesc = opDesc->GetInputDesc(index);
  } else {
    tensorDesc = opDesc->GetOutputDesc(index);
  }
  FUSION_PASS_CHECK(
          tensorDesc.GetShape().GetDimNum() != MAX_DIM_NUM,
          OP_LOGE(FUSED_OP_TYPE.c_str(), "dim count not illegal, need:4 real:%lu", tensorDesc.GetShape().GetDimNum()),
  return PARAM_INVALID);
  // Refresh the variable format and shape
  int64_t n = tensorDesc.GetShape().GetDim(DIM_C);
  int64_t c = tensorDesc.GetShape().GetDim(DIM_N);
  int64_t h = tensorDesc.GetShape().GetDim(DIM_H);
  int64_t w = tensorDesc.GetShape().GetDim(DIM_W);
  tensorDesc.SetShape(ge::GeShape({n, c, h, w}));
  tensorDesc.SetOriginShape(ge::GeShape({n, c, h, w}));
  graphStatus retRes;
  if (bInput) {
    retRes = opDesc->UpdateInputDesc(index, tensorDesc);
  } else {
    retRes = opDesc->UpdateOutputDesc(index, tensorDesc);
  }
  FUSION_PASS_CHECK(retRes != ge::GRAPH_SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Update matmul variable failed"),
  return PARAM_INVALID);
  return SUCCESS;
}

Status Conv2DGroupFusionPass::ProcessDepthwiseConv(NodePtr convNode) {
  FUSION_PASS_CHECK(
          convNode->GetInAllNodes().size() < 2,
          OP_LOGE(FUSED_OP_TYPE.c_str(),
                  "The number of input of the node[name=%s, type=%s] is less than 2, there is no weight input.",
                  convNode->GetName().c_str(), convNode->GetType().c_str()),
  return FAILED);
  OpDescPtr filterDesc = convNode->GetInAllNodes().at(1)->GetOpDesc();
  FUSION_PASS_CHECK(filterDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Filter GetOpDesc fail"),
  return PARAM_INVALID);

  OpDescPtr convDesc = convNode->GetOpDesc();
  if (PatternFusionUtil::IsUnknownShape(convDesc->GetInputDesc(1).GetShape().GetDim(1))) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Conv2DGroupFusionPass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }
  if (convDesc->GetInputDesc(1).GetShape().GetDim(1) != 1) {
    std::map<string, string> error_key_map;
    error_key_map["pass_name"] = "GroupConv2DFusionPass";
    error_key_map["errmsg"] = "Conv2D node:[" + convDesc->GetName() +
                              "], filter cin channel must be 1 in depthwise convolution";
    ErrorManager::GetInstance().ReportErrMessage("E20008", error_key_map);
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Filter channel must be 1 in depthwise conv");
    return PARAM_INVALID;
  }
  FUSION_PASS_CHECK(SwapNumChn(filterDesc, false, 0) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Conv parent const node out 0 change nc failed"), return FAILED);

  FUSION_PASS_CHECK(SwapNumChn(convDesc, true, 1) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Conv node input 1 change nc failed"), return FAILED);

  // change op type to depthwise
  OP_LOGI(FUSED_OP_TYPE.c_str(), "change the conv type");
  convDesc->SetType("DepthwiseConv2D");
  // because conv2d no data_format and padding setting but depthwise has
  FUSION_PASS_CHECK(!ge::AttrUtils::SetStr(convDesc, "data_format", "NCHW"),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "set data_format NCHW fail"), return FAILED);
  convDesc->DelAttr("groups");
  return SUCCESS;
}

int64_t Conv2DGroupFusionPass::GetGroups(ge::OpDescPtr &convDesc) {
  int64_t groups = 1;
  bool hasGroup = ge::AttrUtils::GetInt(convDesc, "groups", groups);
  return hasGroup ? groups : 1;
}

bool Conv2DGroupFusionPass::GenerateSplitNode(ge::ComputeGraph &graph, ge::OpDescPtr &convDesc, int64_t &groups,
                                              ge::NodePtr &splitNode, ge::GeTensorDesc &splitOutDesc) {
  OpDescPtr sliceDesc;
  string convOpName = convDesc->GetName();
  GeTensorDesc inputDesc = convDesc->GetInputDesc(0);
  inputDesc.SetOriginDataType(DT_FLOAT16);
  inputDesc.SetDataType(DT_FLOAT16);
  GeShape inputShape = inputDesc.GetShape();
  size_t inChannelIdx = -1;
  FUSION_PASS_CHECK(SUCCESS != PatternFusionUtil::ParseChannelIdx(inputDesc, inChannelIdx),
                    OP_LOGE(FUSED_OP_TYPE.c_str(),
                            "The original format of the conv node[name=%s, type=%s]'s input0 is %s, which is unsupportable.",
                            convDesc->GetName().c_str(), convDesc->GetType().c_str(),
                            ge::TypeUtils::FormatToSerialString(inputDesc.GetFormat()).c_str()),
  return FAILED);
  int newInputChn = inputShape.GetDim(inChannelIdx);
  GeShape splitOutShape = inputShape;
  splitOutShape.SetDim(inChannelIdx, newInputChn / groups);

  FUSION_PASS_MAKE_SHARED((sliceDesc = std::make_shared<ge::OpDesc>(convOpName+"_split", "SplitD")), return FAILED);
  AttrUtils::SetInt(sliceDesc, "split_dim", inChannelIdx);
  AttrUtils::SetInt(sliceDesc, "num_split", groups);
  splitOutDesc = inputDesc;
  splitOutDesc.Update(splitOutShape, inputDesc.GetOriginFormat(), DT_FLOAT16);
  splitOutDesc.SetOriginShape(splitOutShape);
  sliceDesc->AddInputDesc(inputDesc);
  for (int i = 0; i < groups; i++) {
    sliceDesc->AddOutputDesc(splitOutDesc);
  }
  splitNode = graph.AddNode(sliceDesc);
  return true;
}

bool Conv2DGroupFusionPass::GenerateNewConvNodes(ge::ComputeGraph &graph, ge::OpDescPtr &convDesc,
                                                 const ge::GeTensorDesc &splitOutDesc, vector<ge::NodePtr> &newConvNodes,
                                                 ge::GeTensorDesc &newConvOutDesc) {
  int64_t groups = GetGroups(convDesc);

  GeTensorDesc inDesc = splitOutDesc;
  inDesc.SetOriginDataType(convDesc->GetInputDesc(0).GetOriginDataType());
  inDesc.SetDataType(convDesc->GetInputDesc(0).GetDataType());

  string convOpName = convDesc->GetName();
  GeTensorDesc outputDesc = convDesc->GetOutputDesc(0);
  GeShape newConvOutShape = outputDesc.GetShape();
  size_t outChannelIdx = -1;
  FUSION_PASS_CHECK(SUCCESS != PatternFusionUtil::ParseChannelIdx(outputDesc, outChannelIdx),
                    OP_LOGE(FUSED_OP_TYPE.c_str(),
                            "The original format of the conv node[name=%s, type=%s]'s input0 is %s, which is unsupportable.",
                            convDesc->GetName().c_str(), convDesc->GetType().c_str(),
                            ge::TypeUtils::FormatToSerialString(outputDesc.GetFormat()).c_str()),
  return FAILED);
  int newConvOutChn = newConvOutShape.GetDim(outChannelIdx) / groups;
  newConvOutShape.SetDim(outChannelIdx, newConvOutChn);
  newConvOutDesc = outputDesc;
  newConvOutDesc.Update(newConvOutShape, outputDesc.GetOriginFormat(), convDesc->GetOutputDesc(0).GetDataType());
  newConvOutDesc.SetOriginShape(newConvOutShape);
  newConvOutDesc.SetOriginDataType(convDesc->GetOutputDesc(0).GetOriginDataType());
  for (int64_t i = 0; i < groups; i++) {
    ostringstream newConvName;
    newConvName << convOpName << "_" << i;
    OpDescPtr newConvDesc = AttrUtils::CopyOpDesc(convDesc);
    FUSION_PASS_CHECK(newConvDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.",
                                                      convDesc->GetName().c_str()),
    return PARAM_INVALID);
    newConvDesc->SetName(newConvName.str());
    (void)newConvDesc->UpdateInputDesc(0, inDesc);
    for (unsigned int j = 1; j < convDesc->GetAllInputsDesc().size(); j++) {
      newConvDesc->UpdateInputDesc(j, inDesc);
    }
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != newConvDesc->UpdateOutputDesc(0, newConvOutDesc),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Update node:%s's 1st outputfailed.", convDesc->GetName().c_str()),
    return false);
    AttrUtils::SetInt(newConvDesc, "groups", 1);
    NodePtr newConvNode = graph.AddNode(newConvDesc);
    newConvNodes.push_back(newConvNode);
  }
  return true;
}

bool Conv2DGroupFusionPass::GenerateConcatNode(ge::ComputeGraph &graph, ge::OpDescPtr &convDesc, const int64_t &groups,
                                               ge::GeTensorDesc &newConvOutDesc, ge::NodePtr &concatNode) {
  string convOpName = convDesc->GetName();
  OpDescPtr concatDesc;
  newConvOutDesc.SetOriginDataType(DT_FLOAT16);
  newConvOutDesc.SetDataType(DT_FLOAT16);
  FUSION_PASS_MAKE_SHARED((concatDesc = std::make_shared<ge::OpDesc>(convOpName+"_concat", "ConcatD")), return false);
  for (int i = 0; i < groups; i++) {
    concatDesc->AddInputDesc(newConvOutDesc);
  }
  concatDesc->AddOutputDesc(convDesc->GetOutputDesc(0));
  GeTensorDesc inputDesc = convDesc->GetInputDesc(0);
  size_t inChannelIdx = -1;
  FUSION_PASS_CHECK(SUCCESS != PatternFusionUtil::ParseChannelIdx(inputDesc, inChannelIdx),
                    OP_LOGE(FUSED_OP_TYPE.c_str(),
                            "The original format of the conv node[name=%s, type=%s]'s input0 is %s, which is unsupportable.",
                            convDesc->GetName().c_str(), convDesc->GetType().c_str(),
                            ge::TypeUtils::FormatToSerialString(inputDesc.GetFormat()).c_str()),
  return FAILED);
  AttrUtils::SetInt(concatDesc, "concat_dim", inChannelIdx); // c axis concat
  concatNode = graph.AddNode(concatDesc);
  return true;
}

bool Conv2DGroupFusionPass::Relink(ge::NodePtr &convNode, ge::NodePtr &splitNode, vector<ge::NodePtr> &newConvNodes,
                                   ge::NodePtr &concatNode) {
  auto op_desc = convNode->GetOpDesc();
  int64_t groups = GetGroups(op_desc);

  Node::Vistor<NodePtr> inNodes = convNode->GetInAllNodes();
  FUSION_PASS_CHECK(inNodes.size() < 2,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "conv input nodes num(%lu) < 2", inNodes.size()), return false);

  Node::Vistor<NodePtr> outNodes = convNode->GetOutAllNodes();

  int inAnchorIndex = convNode->GetInDataAnchor(0)->GetPeerOutAnchor()->GetIdx();
  vector<int> outAnchorIndexes;
  for (size_t i = 0; i < outNodes.size(); i++) {
    outAnchorIndexes.push_back(convNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(i)->GetIdx());
  }
  for (auto outAnchor : convNode->GetAllOutDataAnchors()) {
    if (outAnchor != nullptr) {
      outAnchor->UnlinkAll();
    }
  }

  graphStatus status = GraphUtils::AddEdge(inNodes.at(0)->GetOutAnchor(inAnchorIndex), splitNode->GetInDataAnchor(0));
  FUSION_PASS_CHECK(status != GRAPH_SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add data to slice edge fail"), return false);
  for (int i = 0; i < groups; i++) {
    status = GraphUtils::AddEdge(splitNode->GetOutDataAnchor(i), newConvNodes[i]->GetInDataAnchor(0));
    FUSION_PASS_CHECK(status != GRAPH_SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add slice to conv edge fail"), return false);
    status = GraphUtils::AddEdge(newConvNodes[i]->GetOutDataAnchor(0), concatNode->GetInDataAnchor(i));
    FUSION_PASS_CHECK(status != GRAPH_SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add conv to concat edge fail"), return false);
  }

  for (size_t i = 0; i < outNodes.size(); i++) {
    status = GraphUtils::AddEdge(concatNode->GetOutAnchor(0),
                                 outNodes.at(i)->GetInDataAnchor(outAnchorIndexes[i]));
    FUSION_PASS_CHECK(status != GRAPH_SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add concat to output edge fail"), return false);
  }
  return true;
}

bool Conv2DGroupFusionPass::IsVariableOrDataNode(const ge::NodePtr &convInputNode) {
  if (convInputNode->GetType() == "Variable") {
    return true;
  }
  else if (convInputNode->GetType() == "Data") {
    string data_real_type;
    if (ge::NodeUtils::GetConstOpType(convInputNode, data_real_type)) {
      return false;
    }
    return true;
  }
  return false;
}

Status Conv2DGroupFusionPass::CloneAndLinkQuants(ge::ComputeGraph &graph, const ge::NodePtr &splitNode, const int64_t &group,
                                                 vector<ge::NodePtr> &newConvNodes) {

  /*
      1. unlink slice node output anchors
      2. copy quant nodes
      3. link to slice node's out anchors
      4. link to conv nodes
      5. remove origin quant node
  */

  OpDescPtr opDescNew     = nullptr;
  NodePtr quantNode       = nullptr;
  NodePtr newQuantNode    = nullptr;

  quantNode = splitNode->GetInAllNodes().at(0);
  if (quantNode == nullptr || quantNode->GetType() != "AscendQuant")
    return SUCCESS;
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start proc quant node: %s", quantNode->GetName().c_str());

  for (const auto& outAnchor : splitNode->GetAllOutDataAnchors()) {
    if (outAnchor != nullptr) {
      outAnchor->UnlinkAll();
    }
  }

  for (int i = 0; i < group; i++) {
    opDescNew = ge::AttrUtils::CloneOpDesc(quantNode->GetOpDesc());
    opDescNew->SetName(opDescNew->GetName() + to_string(i));
    FUSION_PASS_CHECK(SUCCESS != opDescNew->UpdateInputDesc(0, splitNode->GetOpDesc()->GetOutputDesc(0)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "node %s UpdateInputDesc failed.",
                              opDescNew->GetName().c_str()), return FAILED);
    FUSION_PASS_CHECK(SUCCESS != opDescNew->UpdateOutputDesc(0, newConvNodes[i]->GetOpDesc()->GetInputDesc(0)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "node %s UpdateInputDesc failed.",
                              opDescNew->GetName().c_str()), return FAILED);
    if (opDescNew == nullptr)
      return FAILED;
    newQuantNode = graph.AddNode(opDescNew);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(splitNode->GetOutDataAnchor(i),
                                                         newQuantNode->GetInDataAnchor(0)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              splitNode->GetName().c_str(), newQuantNode->GetName().c_str()),
    return FAILED);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(newQuantNode->GetOutDataAnchor(0),
                                                         newConvNodes[i]->GetInDataAnchor(0)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              newQuantNode->GetName().c_str(), newConvNodes[i]->GetName().c_str()),
    return FAILED);
  }

  FUSION_PASS_CHECK(SUCCESS != graph.RemoveNode(quantNode),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "RemoveNode %s failed.", quantNode->GetName().c_str()),
  return FAILED);

  return SUCCESS;
}

Status Conv2DGroupFusionPass::SplitDequant(ge::ComputeGraph &graph, const ge::NodePtr &concatNode, const int64_t &group,
                                           vector<ge::NodePtr> &newConvNodes) {

  /*
    1. unlink concat node input anchors
    2. copy dequant nodes
    3. link to concat node's in anchors
    4. link to conv nodes
    5. splite dequant scale
    6. remove origin dequant node
  */

  // next node must be dquant, or we do nothing
  NodePtr dequantNode = concatNode->GetOutAllNodes().at(0);
  if (dequantNode == nullptr || dequantNode->GetType() != "AscendDequant"
      || concatNode->GetOutAllNodes().size() != 1)
    return SUCCESS;
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start proc dequant node: %s", dequantNode->GetName().c_str());

  OP_LOGI(FUSED_OP_TYPE.c_str(), "GetWeights failed, node name %s, weight size is [%d]",
          dequantNode->GetName().c_str(), OpDescUtils::GetWeights(dequantNode).size());
  vector<NodePtr> newDequantNodes;

  for (int i = 0; i < group; i++) {
    OpDescPtr opDescNew = nullptr;
    opDescNew = ge::AttrUtils::CloneOpDesc(dequantNode->GetOpDesc());
    opDescNew->SetName(opDescNew->GetName() + to_string(i));
    GeTensorDesc inTensor = newConvNodes[0]->GetOpDesc()->GetOutputDesc(0);
    GeTensorDesc outTensor = concatNode->GetOpDesc()->GetInputDesc(0);
    outTensor.SetOriginDataType(DT_FLOAT16);
    outTensor.SetDataType(DT_FLOAT16);
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != opDescNew->UpdateInputDesc(0, inTensor),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "UpdateInputDesc node %s failed", opDescNew->GetName().c_str()),
    return FAILED);
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != opDescNew->UpdateOutputDesc(0, outTensor),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "UpdateOutputDesc node %s failed", opDescNew->GetName().c_str()),
    return FAILED);
    FUSION_PASS_CHECK(SUCCESS != concatNode->GetOpDesc()->UpdateInputDesc(i, outTensor),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "node %s UpdateInputDesc failed.",
                              concatNode->GetName().c_str()), return FAILED);
    NodePtr newDequantNode = graph.AddNode(opDescNew);
    newDequantNodes.push_back(newDequantNode);
  }
  FUSION_PASS_CHECK(PatternFusionUtil::InsertSliceDNodes(graph, dequantNode, 1,
                                                         newDequantNodes, group, 0) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Insert SliceD node between node[%s] and weight node falied.",
                            dequantNode->GetName().c_str()),
  return FAILED);
  for (int i = 0; i < group; i++) {
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(newConvNodes[i]->GetOutDataAnchor(0),
                                                         newDequantNodes[i]->GetInDataAnchor(0)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              newConvNodes[i]->GetName().c_str(), newDequantNodes[i]->GetName().c_str()),
    return FAILED);
    if (concatNode->GetInDataAnchor(i) != nullptr) {
      concatNode->GetInDataAnchor(i)->UnlinkAll();
    }
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(newDequantNodes[i]->GetOutDataAnchor(0),
                                                         concatNode->GetInDataAnchor(i)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              newDequantNodes[i]->GetName().c_str(), concatNode->GetName().c_str()),
    return FAILED);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "add edge from [%s] to [%s]",
            newDequantNodes[i]->GetName().c_str(), concatNode->GetName().c_str());
  }

  FUSION_PASS_CHECK(SUCCESS != graph.RemoveNode(dequantNode),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "RemoveNode %s failed.", dequantNode->GetName().c_str()),
  return FAILED);

  return SUCCESS;

}

Status Conv2DGroupFusionPass::ProcQuantIfNeed(ge::ComputeGraph &graph, const ge::NodePtr &splitNode, const ge::NodePtr &concatNode,
                                              const int64_t &groups, vector<ge::NodePtr> &newConvNodes) {
  Status ret;
  ret = CloneAndLinkQuants(graph, splitNode, groups, newConvNodes);
  if (SUCCESS != ret) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "CloneAndLinkQuants failed!");
    return FAILED;
  }

  ret = SplitDequant(graph, concatNode, groups, newConvNodes);
  if (SUCCESS != ret) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "SplitDequant failed!");
    return FAILED;
  }

  return SUCCESS;
}

Status Conv2DGroupFusionPass::ProcessGroupConv(ge::ComputeGraph &graph, ge::NodePtr &convNode) {
  NodePtr splitNode = nullptr;
  GeTensorDesc splitOutDesc;
  OpDescPtr convDesc = convNode->GetOpDesc();
  int64_t groups = GetGroups(convDesc);
  FUSION_PASS_CHECK(!GenerateSplitNode(graph, convDesc, groups, splitNode, splitOutDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "generate split node fail"), return FAILED);

  vector<NodePtr> newConvNodes;
  GeTensorDesc newConvOutDesc;
  FUSION_PASS_CHECK(!GenerateNewConvNodes(graph, convDesc, splitOutDesc, newConvNodes, newConvOutDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "generate new conv nodes fail"), return FAILED);

  NodePtr concatNode;
  FUSION_PASS_CHECK(!GenerateConcatNode(graph, convDesc, groups, newConvOutDesc, concatNode),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "generate concat fail"), return FAILED);

  FUSION_PASS_CHECK(!Relink(convNode, splitNode, newConvNodes, concatNode),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "relink fail"), return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "conv[%s]'s weight size is [%zu]",
          convNode->GetName().c_str(), OpDescUtils::GetWeights(convNode).size());
  // find sliceD of conv's 2nd or 3rd, 4th input
  size_t inNChannelIdx = -1;
  for (unsigned int i = 1; i < convNode->GetInDataNodes().size(); i++) {
    GeTensorDesc inputDesc = convDesc->GetInputDesc(i);
    FUSION_PASS_CHECK(SUCCESS != PatternFusionUtil::ParseNChannelIdx(inputDesc, inNChannelIdx),
                      OP_LOGE(FUSED_OP_TYPE.c_str(),
                              "The original format of the conv node[name=%s, type=%s]'s input0 is %s, which is unsupportable.",
                              convDesc->GetName().c_str(), convDesc->GetType().c_str(),
                              ge::TypeUtils::FormatToSerialString(inputDesc.GetFormat()).c_str()),
    return FAILED);
    if (inputDesc.GetShape().GetDimNum() == 1) {
      inNChannelIdx = 0;
    }
    FUSION_PASS_CHECK(PatternFusionUtil::InsertSliceDNodes(graph, convNode, i,
                                                           newConvNodes, groups, inNChannelIdx) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Insert SliceD node between node[%s] and weight node falied.",
                              convNode->GetName().c_str()),
    return FAILED);
  }

  FUSION_PASS_CHECK(SUCCESS != ProcQuantIfNeed(graph, splitNode, concatNode, groups, newConvNodes),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "CloneAndLinkQuants fail, conv name %s",
                            convNode->GetName().c_str()), return FAILED);

  for (auto inAnchor : convNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  FUSION_PASS_CHECK(SUCCESS != graph.RemoveNode(convNode),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "RemoveNode %s failed.", convNode->GetName().c_str()),
  return FAILED);
  return SUCCESS;
}

Status Conv2DGroupFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter Conv2DGroupPass::Fusion.");
  NodePtr convNode = GetNodeFromMapping(PATTERN_CONV2D_ID, mapping);
  OpDescPtr convDesc = convNode->GetOpDesc();

  // 1.if the deconv node doesn't have the attribute groups or the value is 1, just return not changed.
  int64_t groups = 1;
  bool hasGroup = ge::AttrUtils::GetInt(convDesc, "groups", groups);
  if (!hasGroup || groups == 1) {
    OP_LOGD(FUSED_OP_TYPE.c_str(),
            "The conv node[name=%s, type=%s] doesn't have the attribute groups, or the value is 1.",
            convDesc->GetName().c_str(), convDesc->GetType().c_str());
    return NOT_CHANGED;
  }

  GeTensorDesc inputDesc = convDesc->GetInputDesc(0);
  size_t inChannelIdx = -1;
  FUSION_PASS_CHECK(
          SUCCESS != PatternFusionUtil::ParseChannelIdx(inputDesc, inChannelIdx),
          OP_LOGW(FUSED_OP_TYPE.c_str(),
                  "The original format of the conv node[name=%s, type=%s]'s input0 is %s, which is unsupportable.",
                  convDesc->GetName().c_str(), convDesc->GetType().c_str(),
                  ge::TypeUtils::FormatToSerialString(inputDesc.GetFormat()).c_str()),
  return NOT_CHANGED);
  int64_t inChn = inputDesc.GetOriginShape().GetDim(inChannelIdx);

  GeTensorDesc outputDesc = convDesc->GetOutputDesc(0);
  size_t outChannelIdx = -1;
  FUSION_PASS_CHECK(
          SUCCESS != PatternFusionUtil::ParseChannelIdx(outputDesc, outChannelIdx),
          OP_LOGW(FUSED_OP_TYPE.c_str(),
                  "The original format of the conv node[name=%s, type=%s]'s output0 is %s, which is unsupportable.",
                  convDesc->GetName().c_str(), convDesc->GetType().c_str(),
                  ge::TypeUtils::FormatToSerialString(outputDesc.GetFormat()).c_str()),
  return NOT_CHANGED);
  int64_t outChn = outputDesc.GetOriginShape().GetDim(outChannelIdx);
  if (PatternFusionUtil::IsUnknownShape(outChn) ||
      PatternFusionUtil::IsUnknownShape(inChn)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Conv2DGroupFusionPass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }
  if (groups == inChn && groups == outChn) {
    return ProcessDepthwiseConv(convNode);
  } else if (inChn % groups == 0 && outChn % groups == 0) {
    if (IsVariableOrDataNode(convNode->GetInAllNodes().at(1))) {
      return ProcessGroupConv(graph, convNode);
    } else {
      return PatternFusionUtil::ProcessGroupPadding(graph, convNode, groups);
    }
  } else {
    OP_LOGW(FUSED_OP_TYPE.c_str(),
            "The number of input channel(%lld) or output channel(%lld) of "
            "the conv node[name=%s, type=%s] is not divisible by groups(%lld)",
            inChn, outChn, convDesc->GetName().c_str(), convDesc->GetType().c_str(), groups);
    return NOT_CHANGED;
  }
}
REGISTER_PASS("GroupConv2DFusionPass", BUILT_IN_GRAPH_PASS, Conv2DGroupFusionPass);
}  // namespace fe
