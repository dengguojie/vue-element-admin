/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#include "mul_add_add_fusion_pass.h"

#include <math.h>
#include <iostream>
#include <map>
#include <algorithm>
#include "external/graph/operator_factory.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"

using namespace ge;
namespace fe {
static const std::string PATTERN_INPUT = "Input0";
static const std::string PATTERN_MUL = "Mul";
static const std::string PATTERN_ADD_1 = "Add_1";
static const std::string PATTERN_ADD_2 = "Add_2";
static const std::string PATTERN_TRANSDATA = "TransData";
static const std::string CONSTANT = "Const";
static const std::string CONSTANTOP = "Constant";
static const std::string DATAOP = "Data";
static const char* MUL = "Mul";
static const char* ADD = "Add";
static const char* TRANSDATA = "TransData";
static const int64_t ALIGN_UNIT_16 = 16;
static const size_t MIN_DIM_LEN = 2;
static const int64_t DIM_NUM_TWO = 2;
bool alignFlag = false;

vector<FusionPattern*> MulAddAddFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("MulAddAddFusion");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_INPUT)
      .AddOpDesc(PATTERN_MUL, {MUL})
      .AddOpDesc(PATTERN_ADD_1, {ADD})
      .AddOpDesc(PATTERN_TRANSDATA, {TRANSDATA})
      .AddOpDesc(PATTERN_ADD_2, {ADD})
      .SetInputs(PATTERN_ADD_1, {PATTERN_MUL})
      .SetInputs(PATTERN_TRANSDATA, {PATTERN_ADD_1})
      .SetInputs(PATTERN_ADD_2, {PATTERN_INPUT, PATTERN_TRANSDATA})
      .SetOutput(PATTERN_ADD_2);
  patterns.push_back(pattern);
  return patterns;
}

Status MulAddAddFusionPass::RemoveFusedNode(ge::ComputeGraph& graph, ge::NodePtr& fusedNode) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "RemovefusedNode begin");

  for (auto inAnchor : fusedNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  if (fusedNode->GetInControlAnchor() != nullptr) {
    fusedNode->GetInControlAnchor()->UnlinkAll();
  }
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(fusedNode),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove Node:%s", fusedNode->GetName().c_str()),
      return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "RemovefusedNode end");
  return SUCCESS;
}

Status MulAddAddFusionPass::CheckFusedNode(vector<ge::NodePtr>& fusedNodes, ge::NodePtr& transdataDstNode) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckFusedNode begin");

  for (size_t index = 0; index < fusedNodes.size(); index++) {
    FUSION_PASS_CHECK(fusedNodes[index]->GetOpDesc() == nullptr,
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "[%s] get desc failed.", fusedNodes[index]->GetName().c_str()),
                      return NOT_CHANGED);
  }

  const std::vector<ge::Format> kFormatList = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ};
  std::vector<ge::Format> nodeFormatList;

  nodeFormatList.push_back(fusedNodes[0]->GetOpDesc()->GetInputDesc(1).GetFormat());
  nodeFormatList.push_back(fusedNodes[0]->GetOpDesc()->GetInputDesc(0).GetFormat());
  nodeFormatList.push_back(fusedNodes[1]->GetOpDesc()->GetInputDesc(1).GetFormat());
  nodeFormatList.push_back(fusedNodes[3]->GetOpDesc()->GetInputDesc(0).GetFormat());

  FUSION_PASS_CHECK(nodeFormatList != kFormatList,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "input format should be ND/ND/ND/NZ!"), return NOT_CHANGED);

  FUSION_PASS_CHECK(HasUnKnowShape(fusedNodes[0]) || HasUnKnowShape(fusedNodes[1]) || HasUnKnowShape(fusedNodes[3]),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "MulAddAddFusion do not support dynamic shape!"),
                    return NOT_CHANGED);

  vector<int64_t> muldimInfo = fusedNodes[0]->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
  FUSION_PASS_CHECK(muldimInfo.size() < MIN_DIM_LEN,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "The input dimension of mul is less than 2!"), return NOT_CHANGED);

  alignFlag = muldimInfo[muldimInfo.size() - DIM_NUM_TWO] % ALIGN_UNIT_16 == 0;

  FUSION_PASS_CHECK(
      muldimInfo[muldimInfo.size() - 1] != 1,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "The last dimensional value of mulNode input shape is not 1, dims[-1] is [%ld]",
              muldimInfo[muldimInfo.size() - 1]),
      return NOT_CHANGED);

  ge::NodePtr constNode = fusedNodes[1]->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();
  std::string nodeType = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(constNode);
  if (nodeType != CONSTANT && nodeType != CONSTANTOP && nodeType != DATAOP) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The type of add input is not constant.");
    return NOT_CHANGED;
  }
  vector<ge::GeTensorPtr> weightsConst = ge::OpDescUtils::MutableWeights(constNode);
  FUSION_PASS_CHECK(weightsConst.empty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "weightsConst is null ptr!"),
                    return NOT_CHANGED);
  vector<int64_t> constDim = weightsConst[0]->GetTensorDesc().GetShape().GetDims();
  FUSION_PASS_CHECK(constDim.size() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "The const input of add only support one-dimensional data"),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(
      constDim[0] % ALIGN_UNIT_16 == 0 || constDim[0] == 1,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "The constant value can divided by 16, dims[0] is [%ld]", constDim[0]),
      return NOT_CHANGED);

  FUSION_PASS_CHECK(GetTransdataNode(fusedNodes[0], transdataDstNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get transdata node failed."),
                    return NOT_CHANGED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckFusedNode end");
  return SUCCESS;
}

Status MulAddAddFusionPass::CheckFusedControlAnchor(ge::NodePtr& fusedNode, ge::NodePtr& mulAddAddNode) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckFusedControlAnchor begin");

  if (fusedNode->GetOutControlAnchor() != nullptr) {
    if (!fusedNode->GetOutControlAnchor()->GetPeerInControlAnchors().empty()) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "The PeerInControlAnchors of fused node[%s] output control anchor is not empty",
              fusedNode->GetName().c_str());
      for (InControlAnchorPtr inCtrlAnchorPtr : fusedNode->GetOutControlAnchor()->GetPeerInControlAnchors()) {
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(mulAddAddNode->GetOutControlAnchor(), inCtrlAnchorPtr),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Fail to add output control edge for mulAddAddNode"),
            return FAILED);
      }
    }
    fusedNode->GetOutControlAnchor()->UnlinkAll();
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckFusedControlAnchor end");
  return SUCCESS;
}

Status MulAddAddFusionPass::GetTransdataNode(ge::NodePtr& srcNode, ge::NodePtr& mulDstNode) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "GetTransdataNode begin");

  ge::NodePtr parentNode = srcNode->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();
  for (ge::InDataAnchorPtr& inAnchorPtr : parentNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    ge::NodePtr dstNode = inAnchorPtr->GetOwnerNode();
    if (dstNode->GetOpDesc()->GetType() == "TransData") {
      ge::Format srcFormat = dstNode->GetOpDesc()->GetInputDesc(0).GetFormat();
      ge::Format dstFormat = dstNode->GetOpDesc()->GetOutputDesc(0).GetFormat();
      if (srcFormat == ge::FORMAT_ND && dstFormat == ge::FORMAT_FRACTAL_NZ) {
        mulDstNode = dstNode;
        OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckFusedNode end and successful");
        return SUCCESS;
      }
    }
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "GetTransdataNode end but failed, because TransData only support ND to FRACTAL_NZ");
  return FAILED;
}

Status MulAddAddFusionPass::AddMulPadDNode(ge::ComputeGraph& graph, ge::NodePtr& mulNode, ge::NodePtr& mulPadDNode) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into AddMulPadDNode");

  ge::OpDescPtr mulPadDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((mulPadDesc = std::make_shared<ge::OpDesc>(mulNode->GetName() + '_' + "PadD", "PadD")),
                          return INTERNAL_ERROR);
  vector<vector<int64_t>> mulPaddingsValue = {{0, 0}, {0, 0}};
  vector<int64_t> mulConstDimInfo = mulNode->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
  mulPaddingsValue[0][1] = ALIGN_UNIT_16 - (mulConstDimInfo[mulConstDimInfo.size() - DIM_NUM_TWO] % ALIGN_UNIT_16);
  ge::GeTensorDesc mulPadDInputputDesc = mulNode->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(mulPadDesc->AddInputDesc("x", mulPadDInputputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add mulPadDesc input failed."),
                    return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetListListInt(mulPadDesc, "paddings", mulPaddingsValue),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "set mulPadDesc attr paddings failed."),
                    return FAILED);

  ge::GeTensorDesc mulPadDOutputDesc = mulNode->GetOpDesc()->GetInputDesc(0);
  vector<int64_t> mulPadDOutputShape = mulPadDOutputDesc.GetShape().GetDims();
  mulPadDOutputShape[0] = mulPadDOutputShape[0] + mulPaddingsValue[0][1];

  ge::GeShape mulPadDShape(mulPadDOutputShape);
  mulPadDOutputDesc.SetShape(mulPadDShape);
  mulPadDOutputDesc.SetOriginShape(mulPadDShape);
  FUSION_PASS_CHECK(mulPadDesc->AddOutputDesc("y", mulPadDOutputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add mulPadDesc output failed."),
                    return FAILED);

  mulPadDNode = graph.AddNode(mulPadDesc);
  FUSION_PASS_CHECK(mulPadDNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulPadDNode is null, fusion failed."),
                    return PARAM_INVALID);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to AddMulPadDNode");
  return SUCCESS;
}

Status MulAddAddFusionPass::AddPadDNode(ge::ComputeGraph& graph, ge::NodePtr& addNode, ge::NodePtr& padDNode) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into AddPadDNode");

  ge::OpDescPtr padDDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((padDDesc = std::make_shared<ge::OpDesc>(addNode->GetName() + '_' + "PadD", "PadD")),
                          return INTERNAL_ERROR);
  vector<vector<int64_t>> paddingsValue = {{0, 0}};
  vector<int64_t> constDimInfo = addNode->GetOpDesc()->GetInputDesc(1).GetShape().GetDims();
  paddingsValue[0][1] = ALIGN_UNIT_16 - (constDimInfo[0] % ALIGN_UNIT_16);
  ge::GeTensorDesc padDInputputDesc = addNode->GetOpDesc()->GetInputDesc(1);
  FUSION_PASS_CHECK(padDDesc->AddInputDesc("x", padDInputputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add padDNode input failed."), return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetListListInt(padDDesc, "paddings", paddingsValue),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "set padDNode attr paddings failed."),
                    return FAILED);

  ge::GeTensorDesc padDOutputDesc = addNode->GetOpDesc()->GetInputDesc(1);
  vector<int64_t> padDOutputShape = padDInputputDesc.GetShape().GetDims();
  padDOutputShape[0] = padDOutputShape[0] + paddingsValue[0][1];

  ge::GeShape padDShape(padDOutputShape);
  padDOutputDesc.SetShape(padDShape);
  padDOutputDesc.SetOriginShape(padDShape);
  FUSION_PASS_CHECK(padDDesc->AddOutputDesc("y", padDOutputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add padDNode output failed."),
                    return FAILED);

  padDNode = graph.AddNode(padDDesc);
  FUSION_PASS_CHECK(padDNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "padDNode is null, fusion failed."),
                    return PARAM_INVALID);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to AddPadDNode");
  return SUCCESS;
}

Status MulAddAddFusionPass::AddAndDeleteEdge(vector<ge::NodePtr>& fusedNodes, ge::NodePtr& fusedMulAddAddNode,
                                             ge::NodePtr& mulPadDNode, ge::NodePtr& padDNode,
                                             ge::NodePtr& transdataDstNode) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into AddAndDeleteEdge");

  ge::OutDataAnchorPtr transNodeInAnchor = transdataDstNode->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(transNodeInAnchor, fusedMulAddAddNode->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add input data1 edge failed."),
                    return FAILED);

  if (mulPadDNode != nullptr) {
    ge::OutDataAnchorPtr mulNodeInAnchor = fusedNodes[0]->GetInDataAnchor(0)->GetPeerOutAnchor();
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mulNodeInAnchor, mulPadDNode->GetInDataAnchor(0)) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add mulNodeInAnchor edge failed."),
                      return FAILED);

    ge::OutDataAnchorPtr mulPadDNodeOutAnchor = mulPadDNode->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mulPadDNodeOutAnchor, fusedMulAddAddNode->GetInDataAnchor(1)) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add input data1 edge failed."),
                      return FAILED);
  } else {
    ge::OutDataAnchorPtr mulNodeInAnchor = fusedNodes[0]->GetInDataAnchor(0)->GetPeerOutAnchor();
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mulNodeInAnchor, fusedMulAddAddNode->GetInDataAnchor(1)) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add input data1 edge failed."),
                      return FAILED);
  }

  ge::OutDataAnchorPtr add1NodeInAnchor = fusedNodes[1]->GetInDataAnchor(1)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(add1NodeInAnchor, padDNode->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add padd input data edge failed."),
                    return FAILED);

  ge::OutDataAnchorPtr padDNodeOutAnchor = padDNode->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(padDNodeOutAnchor, fusedMulAddAddNode->GetInDataAnchor(2)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add input data2 edge failed."),
                    return FAILED);

  ge::OutDataAnchorPtr add2NodeInAnchor = fusedNodes[3]->GetInDataAnchor(0)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(add2NodeInAnchor, fusedMulAddAddNode->GetInDataAnchor(3)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add input data3 edge failed."),
                    return FAILED);

  for (auto inDataAnchor : fusedNodes[3]->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fusedNodes[3]->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."),
                      return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(fusedMulAddAddNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."),
                      return FAILED);
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to AddAndDeleteEdge");
  return SUCCESS;
}

Status MulAddAddFusionPass::FusionUnalignedScense(ge::ComputeGraph& graph, vector<ge::NodePtr>& fusedNodes,
                                                  vector<ge::NodePtr>& newNodes, ge::NodePtr& transdataDstNode) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into FusionUnalignedScense");

  ge::NodePtr mulPadDNode = nullptr;
  if (!alignFlag) {
    FUSION_PASS_CHECK(AddMulPadDNode(graph, fusedNodes[0], mulPadDNode) != SUCCESS,
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "add mulPadD node failed!"), return NOT_CHANGED);
  }

  ge::NodePtr padDNode = nullptr;
  FUSION_PASS_CHECK(AddPadDNode(graph, fusedNodes[1], padDNode) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "add addPadD Node failed!"), return NOT_CHANGED);

  ge::OpDescPtr fusedMulAddAddDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((fusedMulAddAddDesc = std::make_shared<ge::OpDesc>(
                               fusedNodes[0]->GetName() + '_' + "FusedMulAddAdd", "FusedMulAddAdd")),
                          return INTERNAL_ERROR);

  ge::GeTensorDesc inputputDesc0 = transdataDstNode->GetOpDesc()->GetOutputDesc(0);
  ge::GeTensorDesc inputputDesc1;
  if (!alignFlag) {
    inputputDesc1 = mulPadDNode->GetOpDesc()->GetOutputDesc(0);
  } else {
    inputputDesc1 = fusedNodes[0]->GetOpDesc()->GetInputDesc(0);
  }
  ge::GeTensorDesc inputputDesc2 = padDNode->GetOpDesc()->GetOutputDesc(0);
  ge::GeTensorDesc inputputDesc3 = fusedNodes[3]->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc outputDesc = fusedNodes[3]->GetOpDesc()->GetOutputDesc(0);

  FUSION_PASS_CHECK(fusedMulAddAddDesc->AddInputDesc("x1", inputputDesc0) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add fusedMulAddAdd input x1 failed."),
                    return FAILED);
  FUSION_PASS_CHECK(fusedMulAddAddDesc->AddInputDesc("x2", inputputDesc1) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add fusedMulAddAdd input x2 failed."),
                    return FAILED);
  FUSION_PASS_CHECK(fusedMulAddAddDesc->AddInputDesc("x3", inputputDesc2) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add fusedMulAddAdd input x3 failed."),
                    return FAILED);
  FUSION_PASS_CHECK(fusedMulAddAddDesc->AddInputDesc("x4", inputputDesc3) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add fusedMulAddAdd input x4 failed."),
                    return FAILED);
  FUSION_PASS_CHECK(fusedMulAddAddDesc->AddOutputDesc("y", outputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add fusedMulAddAdd output y failed."),
                    return FAILED);

  ge::NodePtr fusedMulAddAddNode = graph.AddNode(fusedMulAddAddDesc);
  FUSION_PASS_CHECK(fusedMulAddAddNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedMulAddAddNode is null, fusion failed."),
                    return FAILED);

  FUSION_PASS_CHECK(
      AddAndDeleteEdge(fusedNodes, fusedMulAddAddNode, mulPadDNode, padDNode, transdataDstNode) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add and delete edge failed."), return FAILED);

  FUSION_PASS_CHECK(CheckFusedControlAnchor(fusedNodes[3], fusedMulAddAddNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add control anchor failed."), return FAILED);

  if (mulPadDNode != nullptr) {
    newNodes.push_back(mulPadDNode);
  }
  newNodes.push_back(padDNode);
  newNodes.push_back(fusedMulAddAddNode);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to FusionUnalignedScense");
  return SUCCESS;
}  // namespace fe

Status MulAddAddFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into MulAddAddFusionPass");

  ge::NodePtr mulNode = GetNodeFromMapping(PATTERN_MUL, mapping);
  FUSION_PASS_CHECK(mulNode == nullptr, OP_LOGI("mul Node is null."), return NOT_CHANGED);
  ge::NodePtr addNode1 = GetNodeFromMapping(PATTERN_ADD_1, mapping);
  FUSION_PASS_CHECK(addNode1 == nullptr, OP_LOGI("add1 Node is null."), return NOT_CHANGED);
  ge::NodePtr transdataNode = GetNodeFromMapping(PATTERN_TRANSDATA, mapping);
  FUSION_PASS_CHECK(transdataNode == nullptr, OP_LOGI("transdata Node is null."), return NOT_CHANGED);
  ge::NodePtr addNode2 = GetNodeFromMapping(PATTERN_ADD_2, mapping);
  FUSION_PASS_CHECK(addNode2 == nullptr, OP_LOGI("add2 Node is null."), return NOT_CHANGED);

  vector<ge::NodePtr> fusedNodes = {mulNode, addNode1, transdataNode, addNode2};

  ge::NodePtr transdataDstNode = nullptr;
  FUSION_PASS_CHECK(CheckFusedNode(fusedNodes, transdataDstNode) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Does not fit the fusion scene"), return NOT_CHANGED);
  FUSION_PASS_CHECK(FusionUnalignedScense(graph, fusedNodes, newNodes, transdataDstNode) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Unaligned Scense fusion failed"), return FAILED);

  for (size_t index = 0; index < fusedNodes.size(); index++) {
    FUSION_PASS_CHECK(RemoveFusedNode(graph, fusedNodes[index]) != ge::GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove [%s]",
                                                     fusedNodes[index]->GetName().c_str()),
                      return FAILED);
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to MulAddAddFusionPass");
  return SUCCESS;
}
REGISTER_PASS("ZMulAddAddFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, MulAddAddFusionPass);
}  // namespace fe