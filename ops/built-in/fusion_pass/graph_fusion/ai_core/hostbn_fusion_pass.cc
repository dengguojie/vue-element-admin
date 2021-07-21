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
 * \file hostbn_fusion_pass.cpp
 * \brief conv-biasadd fusion pass(conv-biasadd --> conv)
 */
#include "hostbn_fusion_pass.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;
namespace fe {
static const string PATTERN_BNInference = "BNInference";
static const string BNINFERENCE = "BNInference";
static const string MOMENTUM = "momentum";
static const string EPSILON = "epsilon";
static const string USE_GLOBAL_STATS = "use_global_stats";
static const string MODE = "mode";
static const int32_t INT_NUM_FOUR = 4;

vector<FusionPattern*> HostBNFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("HostBNFusionPass");
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter HostBNFusionPass::DefinePatterns.");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);

  pattern->AddOpDesc(PATTERN_BNInference, {BNINFERENCE}).SetOutput(PATTERN_BNInference);
  patterns.push_back(pattern);

  return patterns;
}

Status HostBNFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter GoBNhost");
  ge::NodePtr bnNode = GetNodeFromMapping(PATTERN_BNInference, mapping);
  FUSION_PASS_CHECK(bnNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node bninfenced is null, fusion failed."),
                    return PARAM_INVALID);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "check BNhost");
  FUSION_PASS_CHECK(CheckParameter(bnNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Check BNInferenceD param failed."), return PARAM_INVALID);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "fusion BNhost");
  return BNFuison(graph, bnNode, newNodes);
}

Status HostBNFusionPass::CheckParameter(ge::NodePtr& bnNodePtr) {
  // get psroipooling node inputs.
  Node::Vistor<NodePtr> inNodes = bnNodePtr->GetInDataNodes();
  FUSION_PASS_CHECK((inNodes.size() != 4 && inNodes.size() != 6),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "BNInference input nodes num(%lu) != 4/6", inNodes.size()),
                    return PARAM_INVALID);
  return SUCCESS;
}

Status HostBNFusionPass::SetAttrValueForNewNode(const ge::OpDescPtr& preOpDescPtr, ge::OpDescPtr& newOpDescPtr) {
  // get and update output_dim
  ge::GeAttrValue epsValue;
  FUSION_PASS_CHECK(preOpDescPtr->GetAttr(EPSILON, epsValue) == ge::GRAPH_FAILED,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Get attr %s from node %s error", EPSILON.c_str(),
                            preOpDescPtr->GetName().c_str()),
                    return SUCCESS);

  FUSION_PASS_CHECK(
      newOpDescPtr->SetAttr(EPSILON, epsValue) == ge::GRAPH_FAILED,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Set attr %s to node %s error", EPSILON.c_str(), newOpDescPtr->GetName().c_str()),
      return FAILED);

  return SUCCESS;
}

Status HostBNFusionPass::BNFuison(ge::ComputeGraph& graph, ge::NodePtr& bnNodePtr, vector<ge::NodePtr>& newNodes) {
  // check conv op desc is null or not

  ge::OpDescPtr bnOpDescPtr = bnNodePtr->GetOpDesc();
  FUSION_PASS_CHECK(
      bnOpDescPtr == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.", bnOpDescPtr->GetName().c_str()),
      return PARAM_INVALID);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "NODE %s 1", bnOpDescPtr->GetName().c_str());
  // get conv node inputs.
  Node::Vistor<NodePtr> inputNodes = bnNodePtr->GetInDataNodes();
  if (inputNodes.size() < INT_NUM_FOUR) {
     return PARAM_INVALID;
  }
  ge::NodePtr dataNodePtr = inputNodes.at(0);
  ge::NodePtr meanNodePtr = inputNodes.at(1);
  ge::NodePtr varNodePtr = inputNodes.at(2);
  ge::NodePtr momentumNodePtr = inputNodes.at(3);
  ge::NodePtr scaleNodePtr = nullptr;
  ge::NodePtr offsetNodePtr = nullptr;
  if (inputNodes.size() == 6) {
    scaleNodePtr = inputNodes.at(4);
    offsetNodePtr = inputNodes.at(5);
  }

  // create bnhost opdesc
  ge::OpDescPtr bnhostOpDescPtr = nullptr;
  FUSION_PASS_MAKE_SHARED(bnhostOpDescPtr = std::make_shared<ge::OpDesc>(bnOpDescPtr->GetName(), "BnHost"),
                          return FAILED);
  bnhostOpDescPtr->SetType("BnHost");
  bnhostOpDescPtr->SetName(bnhostOpDescPtr->GetName() + "_BnHost");
  // update output_dim and group_size

  FUSION_PASS_CHECK(SetAttrValueForNewNode(bnOpDescPtr, bnhostOpDescPtr) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Update output_dim and group_size failed."), return NOT_CHANGED);

  // create bninference_d opdesc
  ge::OpDescPtr bninferOpDescPtr = nullptr;
  FUSION_PASS_MAKE_SHARED(bninferOpDescPtr = std::make_shared<ge::OpDesc>(bnOpDescPtr->GetName(), "BNInferenceD"),
                          return FAILED);
  bninferOpDescPtr->SetType("BNInferenceD");
  bninferOpDescPtr->SetName(bninferOpDescPtr->GetName() + "_BNInferenceD");
  FUSION_PASS_CHECK(SetAttrValue(bnOpDescPtr, bninferOpDescPtr) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Update output_dim and group_size failed."), return NOT_CHANGED);

  // get bnhost input
  ge::GeTensorDesc meanInputTensorDesc;
  FUSION_PASS_CHECK(GetSwapInputTensorDesc(bnhostOpDescPtr, meanNodePtr->GetOpDesc(), meanInputTensorDesc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Create bnhost input mean opDesc failed, fusion failed."),
                    return NOT_CHANGED);
  ge::GeTensorDesc varInputTensorDesc;
  FUSION_PASS_CHECK(GetSwapInputTensorDesc(bnhostOpDescPtr, varNodePtr->GetOpDesc(), varInputTensorDesc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Create bnhost input var opDesc failed, fusion failed."),
                    return NOT_CHANGED);
  ge::GeTensorDesc momentumInputTensorDesc;
  FUSION_PASS_CHECK(
      GetSwapInputTensorDesc(bnhostOpDescPtr, momentumNodePtr->GetOpDesc(), momentumInputTensorDesc) != SUCCESS,
      OP_LOGW(FUSED_OP_TYPE.c_str(), "Create bnhost input momentum opDesc failed, fusion failed."), return NOT_CHANGED);

  ge::GeTensorDesc scaleInputTensorDesc;
  ge::GeTensorDesc offsetInputTensorDesc;
  if (inputNodes.size() == 6) {
    FUSION_PASS_CHECK(
        GetSwapInputTensorDesc(bnhostOpDescPtr, scaleNodePtr->GetOpDesc(), scaleInputTensorDesc) != SUCCESS,
        OP_LOGW(FUSED_OP_TYPE.c_str(), "Create bnhost input scale opDesc failed, fusion failed."), return NOT_CHANGED);

    FUSION_PASS_CHECK(
        GetSwapInputTensorDesc(bnhostOpDescPtr, offsetNodePtr->GetOpDesc(), offsetInputTensorDesc) != SUCCESS,
        OP_LOGW(FUSED_OP_TYPE.c_str(), "Create bnhost input offset opDesc failed, fusion failed."), return NOT_CHANGED);
  }

  // get bnhost output

  ge::GeTensorDesc meanOutputTensorDesc;
  FUSION_PASS_CHECK(
      GetMeanOutputTensorDesc(bnhostOpDescPtr, bnOpDescPtr, meanInputTensorDesc, meanOutputTensorDesc) != SUCCESS,
      OP_LOGW(FUSED_OP_TYPE.c_str(), "Create output mean opDesc failed, fusion failed."), return NOT_CHANGED);

  ge::GeTensorDesc varOutputTensorDesc;
  FUSION_PASS_CHECK(
      GetVarOutputTensorDesc(bnhostOpDescPtr, bnOpDescPtr, varInputTensorDesc, varOutputTensorDesc) != SUCCESS,
      OP_LOGW(FUSED_OP_TYPE.c_str(), "Create output var opDesc failed, fusion failed."), return NOT_CHANGED);

  // update output origin shape of pad
  bnhostOpDescPtr->AddInputDesc("mean", meanInputTensorDesc);
  bnhostOpDescPtr->AddInputDesc("variance", varInputTensorDesc);
  bnhostOpDescPtr->AddInputDesc("momentum", momentumInputTensorDesc);
  if (inputNodes.size() == 6) {
    bnhostOpDescPtr->AddInputDesc("scale", scaleInputTensorDesc);
    bnhostOpDescPtr->AddInputDesc("offset", offsetInputTensorDesc);
  }
  bnhostOpDescPtr->AddOutputDesc("alpha", meanOutputTensorDesc);
  bnhostOpDescPtr->AddOutputDesc("beta", varOutputTensorDesc);

  // get bninferenced input
  ge::GeTensorDesc bninfermeanInputTensorDesc;
  FUSION_PASS_CHECK(
      GetSwapInputTensorDesc(bninferOpDescPtr, meanNodePtr->GetOpDesc(), bninfermeanInputTensorDesc) != SUCCESS,
      OP_LOGW(FUSED_OP_TYPE.c_str(), "Create bnhost input mean opDesc failed, fusion failed."), return NOT_CHANGED);
  ge::GeTensorDesc bninfervarInputTensorDesc;
  FUSION_PASS_CHECK(
      GetSwapInputTensorDesc(bninferOpDescPtr, varNodePtr->GetOpDesc(), bninfervarInputTensorDesc) != SUCCESS,
      OP_LOGW(FUSED_OP_TYPE.c_str(), "Create bnhost input var opDesc failed, fusion failed."), return NOT_CHANGED);
  // get bninferenced output
  ge::GeTensorDesc bninferdataInputTensorDesc;
  FUSION_PASS_CHECK(GetInputDataTensorDesc(dataNodePtr, bnNodePtr, bninferdataInputTensorDesc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Create bnhost input var opDesc failed, fusion failed."),
                    return NOT_CHANGED);
  ge::GeTensorDesc bninferOutputTensorDesc;
  FUSION_PASS_CHECK(GetInferOutputTensorDesc(bninferOpDescPtr, bnOpDescPtr, bninferdataInputTensorDesc,
                                             bninferOutputTensorDesc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Create output var opDesc failed, fusion failed."), return NOT_CHANGED);

  bninferOpDescPtr->AddInputDesc("x", bninferdataInputTensorDesc);
  bninferOpDescPtr->AddInputDesc("mean", bninfermeanInputTensorDesc);
  bninferOpDescPtr->AddInputDesc("variance", bninfervarInputTensorDesc);
  bninferOpDescPtr->AddOutputDesc("y", bninferOutputTensorDesc);

  // add SwapCo node to graph
  ge::NodePtr bnhostNodePtr = graph.AddNode(bnhostOpDescPtr);
  ge::NodePtr bninferNodePtr = graph.AddNode(bninferOpDescPtr);
  FUSION_PASS_CHECK(bnhostNodePtr == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode: bnhostNodePtr is null, fusion failed."), return FAILED);
  FUSION_PASS_CHECK(bninferNodePtr == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode: bninferNodePtr is null, fusion failed."),
                    return FAILED);
  newNodes.push_back(bnhostNodePtr);
  newNodes.push_back(bninferNodePtr);

  // delete edge of prenode and psroinode
  int32_t index = bnNodePtr->GetInDataAnchor(0)->GetPeerOutAnchor()->GetIdx();
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::RemoveEdge(dataNodePtr->GetOutAnchor(index), bnNodePtr->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove input edge from fused data node:%s.", bnNodePtr->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::RemoveEdge(meanNodePtr->GetOutAnchor(0), bnNodePtr->GetInDataAnchor(1)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove input edge from fused mean node:%s.", bnNodePtr->GetName().c_str()),
      return FAILED);
  // delete edge of prenode and psroinode
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::RemoveEdge(varNodePtr->GetOutAnchor(0), bnNodePtr->GetInDataAnchor(2)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove input edge from fused var node:%s.", bnNodePtr->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::RemoveEdge(momentumNodePtr->GetOutAnchor(0), bnNodePtr->GetInDataAnchor(3)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove input edge from fused mu node:%s.", bnNodePtr->GetName().c_str()),
      return FAILED);
  if (inputNodes.size() == 6) {
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::RemoveEdge(scaleNodePtr->GetOutAnchor(0), bnNodePtr->GetInDataAnchor(4)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove input edge from fused scale node:%s.", bnNodePtr->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::RemoveEdge(offsetNodePtr->GetOutAnchor(0), bnNodePtr->GetInDataAnchor(5)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove input edge from fused offset node:%s.", bnNodePtr->GetName().c_str()),
        return FAILED);
  }
  // add the original edge of prenode to swapci
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(meanNodePtr->GetOutAnchor(0), bnhostNodePtr->GetInDataAnchor(0)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s to fusion  mean node:%s failed.",
                            meanNodePtr->GetName().c_str(), bnhostNodePtr->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(varNodePtr->GetOutAnchor(0), bnhostNodePtr->GetInDataAnchor(1)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s to fusion var node:%s failed.",
                            varNodePtr->GetName().c_str(), bnhostNodePtr->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(momentumNodePtr->GetOutAnchor(0), bnhostNodePtr->GetInDataAnchor(2)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s to fusion momentum node:%s failed.",
              momentumNodePtr->GetName().c_str(), bnhostNodePtr->GetName().c_str()),
      return FAILED);
  if (inputNodes.size() == 6) {
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(scaleNodePtr->GetOutAnchor(0), bnhostNodePtr->GetInDataAnchor(3)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s to fusion scale node:%s failed.",
                momentumNodePtr->GetName().c_str(), bnhostNodePtr->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(offsetNodePtr->GetOutAnchor(0), bnhostNodePtr->GetInDataAnchor(4)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s to fusion offset node:%s failed.",
                momentumNodePtr->GetName().c_str(), bnhostNodePtr->GetName().c_str()),
        return FAILED);
  }

  // add the input of bninferenced edge
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(dataNodePtr->GetOutAnchor(index), bninferNodePtr->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from data node:%s to bninfer node:%s failed.",
              dataNodePtr->GetName().c_str(), bninferNodePtr->GetName().c_str()),
      return FAILED);

  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(bnhostNodePtr->GetOutAnchor(0), bninferNodePtr->GetInDataAnchor(1)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from mean node:%s to bninfer node:%s failed.",
              bnhostNodePtr->GetName().c_str(), bninferNodePtr->GetName().c_str()),
      return FAILED);

  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(bnhostNodePtr->GetOutAnchor(1), bninferNodePtr->GetInDataAnchor(2)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from var node:%s to bninfer node:%s failed.",
              bnhostNodePtr->GetName().c_str(), bninferNodePtr->GetName().c_str()),
      return FAILED);

  // add the output of bninferenced edge
  // copy output edge
  size_t outanchorsize = bnNodePtr->GetAllOutDataAnchors().size();
  for (size_t outindex = 0; outindex < outanchorsize; outindex++) {
    for (auto inDataAnchor : bnNodePtr->GetOutDataAnchor(outindex)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(bnNodePtr->GetOutDataAnchor(outindex), inDataAnchor) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove bnhost out data edge failed."), return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(bninferNodePtr->GetOutDataAnchor(outindex), inDataAnchor) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add bnhost out data edge failed."), return FAILED);
    }
  }
  if (bnNodePtr->GetOutControlAnchor() != nullptr) {
    for (auto inControlAnchor : bnNodePtr->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(bnNodePtr->GetOutControlAnchor(), inControlAnchor) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove bnhost out control edge failed."), return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(bninferNodePtr->GetOutControlAnchor(), inControlAnchor) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add bnhost out control edge failed."), return FAILED);
    }
  }
  // remove Normalize from graph
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(bnNodePtr),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove bnNodePtr node[%s] failed", bnNodePtr->GetName().c_str()),
                    return FAILED);
  return SUCCESS;
}

Status HostBNFusionPass::GetSwapInputTensorDesc(const ge::OpDescPtr& currentOpDescPtr,
                                                const ge::OpDescPtr& preOpDescPtr, ge::GeTensorDesc& inputTensorDesc) {
  inputTensorDesc = preOpDescPtr->GetOutputDesc(0);
  return SUCCESS;
}

Status HostBNFusionPass::GetInputDataTensorDesc(const ge::NodePtr& dataNodePtr, const ge::NodePtr& preNodePtr,
                                                ge::GeTensorDesc& inputTensorDesc) {
  int32_t node_index = preNodePtr->GetInDataAnchor(0)->GetPeerOutAnchor()->GetIdx();
  inputTensorDesc = dataNodePtr->GetOpDesc()->GetOutputDesc(node_index);
  return SUCCESS;
}

Status HostBNFusionPass::GetMeanOutputTensorDesc(const ge::OpDescPtr& hostOpDescPtr,
                                                 const ge::OpDescPtr& currentOpDescPtr,
                                                 const ge::GeTensorDesc& inputTensorDesc,
                                                 ge::GeTensorDesc& outputTensorDesc) {
  outputTensorDesc = currentOpDescPtr->GetInputDesc(1);
  outputTensorDesc.SetShape(inputTensorDesc.GetShape());
  outputTensorDesc.SetOriginShape(inputTensorDesc.GetShape());
  outputTensorDesc.SetFormat(inputTensorDesc.GetFormat());

  return SUCCESS;
}

Status HostBNFusionPass::GetVarOutputTensorDesc(const ge::OpDescPtr& hostOpDescPtr,
                                                const ge::OpDescPtr& currentOpDescPtr,
                                                const ge::GeTensorDesc& inputTensorDesc,
                                                ge::GeTensorDesc& outputTensorDesc) {
  outputTensorDesc = currentOpDescPtr->GetInputDesc(2);

  outputTensorDesc.SetShape(inputTensorDesc.GetShape());
  outputTensorDesc.SetOriginShape(inputTensorDesc.GetShape());
  outputTensorDesc.SetFormat(inputTensorDesc.GetFormat());

  return SUCCESS;
}
Status HostBNFusionPass::GetMuOutputTensorDesc(const ge::OpDescPtr& hostOpDescPtr,
                                               const ge::OpDescPtr& currentOpDescPtr,
                                               const ge::GeTensorDesc& inputTensorDesc,
                                               ge::GeTensorDesc& outputTensorDesc) {
  outputTensorDesc = currentOpDescPtr->GetInputDesc(3);
  outputTensorDesc.SetShape(inputTensorDesc.GetShape());
  outputTensorDesc.SetOriginShape(inputTensorDesc.GetShape());
  outputTensorDesc.SetFormat(inputTensorDesc.GetFormat());

  return SUCCESS;
}
Status HostBNFusionPass::SetAttrValue(const ge::OpDescPtr& preOpDescPtr, ge::OpDescPtr& newOpDescPtr) {
  // get and update output_dim
  ge::GeAttrValue epsValue;
  FUSION_PASS_CHECK(preOpDescPtr->GetAttr(EPSILON, epsValue) == ge::GRAPH_FAILED,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get attr %s from node %s error", EPSILON.c_str(),
                            preOpDescPtr->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(
      newOpDescPtr->SetAttr(EPSILON, epsValue) == ge::GRAPH_FAILED,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Set attr %s to node %s error", EPSILON.c_str(), newOpDescPtr->GetName().c_str()),
      return FAILED);
  ge::GeAttrValue use_global_stats;
  FUSION_PASS_CHECK(preOpDescPtr->GetAttr(USE_GLOBAL_STATS, use_global_stats) == ge::GRAPH_FAILED,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get attr %s from node %s error", USE_GLOBAL_STATS.c_str(),
                            preOpDescPtr->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(newOpDescPtr->SetAttr(USE_GLOBAL_STATS, use_global_stats) == ge::GRAPH_FAILED,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Set attr %s to node %s error", USE_GLOBAL_STATS.c_str(),
                            newOpDescPtr->GetName().c_str()),
                    return FAILED);
  ge::GeAttrValue mode;
  FUSION_PASS_CHECK(
      preOpDescPtr->GetAttr(MODE, mode) == ge::GRAPH_FAILED,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get attr %s from node %s error", MODE.c_str(), preOpDescPtr->GetName().c_str()),
      return FAILED);

  FUSION_PASS_CHECK(
      newOpDescPtr->SetAttr(MODE, mode) == ge::GRAPH_FAILED,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Set attr %s to node %s error", MODE.c_str(), newOpDescPtr->GetName().c_str()),
      return FAILED);
  string DTLIST = "_output_dt_list";
  string DTINDEX = "_output_dt_index";
  ge::GeAttrValue outputDtList;
  if (preOpDescPtr->GetAttr(DTLIST, outputDtList) == ge::GRAPH_SUCCESS) {
    FUSION_PASS_CHECK(
        newOpDescPtr->SetAttr(DTLIST, outputDtList) == ge::GRAPH_FAILED,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Set attr %s to node %s error", DTLIST.c_str(), newOpDescPtr->GetName().c_str()),
        return FAILED);
  }
  ge::GeAttrValue outputDtIndex;
  if (preOpDescPtr->GetAttr(DTINDEX, outputDtIndex) == ge::GRAPH_SUCCESS) {
    FUSION_PASS_CHECK(newOpDescPtr->SetAttr(DTINDEX, outputDtIndex) == ge::GRAPH_FAILED,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Set attr %s to node %s error", DTINDEX.c_str(),
                              newOpDescPtr->GetName().c_str()),
                      return FAILED);
  }
  return SUCCESS;
}
Status HostBNFusionPass::GetInferOutputTensorDesc(const ge::OpDescPtr& hostOpDescPtr,
                                                  const ge::OpDescPtr& currentOpDescPtr,
                                                  const ge::GeTensorDesc& inputTensorDesc,
                                                  ge::GeTensorDesc& outputTensorDesc) {
  outputTensorDesc = currentOpDescPtr->GetOutputDesc(0);
  outputTensorDesc.SetShape(outputTensorDesc.GetShape());
  outputTensorDesc.SetOriginShape(outputTensorDesc.GetShape());
  outputTensorDesc.SetFormat(outputTensorDesc.GetFormat());

  return SUCCESS;
}
REGISTER_PASS("HostBNFusionPass", BUILT_IN_GRAPH_PASS, HostBNFusionPass);
}  // namespace fe
