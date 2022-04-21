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

#include "concatv2d_aligned_fusion_pass.h"

#include <math.h>

#include <algorithm>
#include <iostream>
#include <map>

#include "error_util.h"
#include "external/graph/operator_factory.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "securec.h"
#include "tbe_ops_pass_util.h"

using namespace ge;
namespace fe {
static const std::string PATTERN_CONCATV2D = "ConcatV2D";
static const std::string PATTERN_MATMULV2 = "MatMulV2";
static const std::string PATTERN_RELU = "Relu";
static const std::string PATTERN_CONSTANT = "Constant";
static const char* CONCATV2D = "ConcatV2D";
static const char* CONCATV2 = "ConcatV2";
static const char* RELU = "Relu";
static const char* CONSTANT = "Constant";
static const char* CONST = "Const";
static const char* MATMULV2 = "MatMulV2";

static const int64_t ALIGN_UNIT_16 = 16;
static const int32_t ALLOWED_BLK_N = 2;
static const int32_t ALLOWED_DIM_NUM = 2;

static const int32_t CONCATV2D_1_IDX = 0;
static const int32_t RELU_IDX = 1;
static const int32_t CONSTANT_IDX = 2;
static const int32_t MATMULV2_IDX = 3;

static const int32_t PAD1_IDX = 0;
static const int32_t SPLITV_IDX = 1;
static const int32_t PAD2_IDX = 2;
static const int32_t CONCATV2_2_IDX = 3;

static const int32_t SPLIT_DIM_IDX = 2;
static const int32_t CONCAT_DIM_IDX = 2;

static int64_t splitOffset;

/*!
 * @brief Define pattern.
 * The graph struct need to adapt and target is shown as follows:
 *    preNode1  PreNode2         PreNode
 *         \     /                  |          Constant
 *        ConcatV2D                PadD            |
 *            |               ==>    \    /      SplitV
 *           Relu  Constant         ConcatV2D     /  |
 *             \    /                   |       PadD |
 *             MatMulV2                Relu       \  |
 *                |                       \      ConcatV2
 *             PostNode                    \       /
 *                                         MatMulV2
 *                                             |
 * @return vector<FusionPattern*> All valid patterns.
 */
vector<FusionPattern*> ConcatV2DAlignedFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ConcatV2DAlignedFusion");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create a pattern object."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_CONCATV2D, {CONCATV2D, CONCATV2})
      .AddOpDesc(PATTERN_RELU, {RELU})
      .AddOpDesc(PATTERN_CONSTANT, {CONSTANT, CONST})
      .AddOpDesc(PATTERN_MATMULV2, {MATMULV2})
      .SetInputs(PATTERN_RELU, {PATTERN_CONCATV2D})
      .SetInputs(PATTERN_MATMULV2, {PATTERN_RELU, PATTERN_CONSTANT})
      .SetOutput(PATTERN_MATMULV2);
  patterns.push_back(pattern);
  return patterns;
}

Status ConcatV2DAlignedFusionPass::CheckFusedNodes(vector<ge::NodePtr>& fusedNodes) const {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckFusedNodes start");

  FUSION_PASS_CHECK(fusedNodes.size() <= MATMULV2_IDX,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "FusedNodes size should be greater than %d, cur is %d. not changed.",
                            MATMULV2_IDX, fusedNodes.size()),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(HasUnKnowShape(fusedNodes[CONCATV2D_1_IDX]) || HasUnKnowShape(fusedNodes[RELU_IDX]) ||
                        HasUnKnowShape(fusedNodes[MATMULV2_IDX]),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "ConcatV2DAlignedFusion do not support dynamic shape. not changed."),
                    return NOT_CHANGED);

  OpDescPtr concatV2OpDesc = fusedNodes.at(CONCATV2D_1_IDX)->GetOpDesc();
  FUSION_PASS_CHECK(concatV2OpDesc == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to get op desc. not changed."),
                    return NOT_CHANGED);

  int32_t concatN = -1;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetInt(concatV2OpDesc, "N", concatN),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to get op attr. not changed."), return NOT_CHANGED);
  FUSION_PASS_CHECK(concatN != ALLOWED_BLK_N,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Attr N of ConcatV2D should be %d, cur is %d. not changed.",
                            ALLOWED_BLK_N, concatN),
                    return NOT_CHANGED);

  auto x0Shape = concatV2OpDesc->GetInputDesc(0).GetShape();
  FUSION_PASS_CHECK((x0Shape.GetDimNum() != ALLOWED_DIM_NUM),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Shape size of x0 should be %d. not changed.", ALLOWED_DIM_NUM),
                    return NOT_CHANGED);

  int64_t dimSize = x0Shape.GetDim(1);
  FUSION_PASS_CHECK((dimSize % ALIGN_UNIT_16 == 0),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Last dim size of x0 have aligned, not changed."),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK((concatV2OpDesc->GetInputDesc(1).GetShape().GetDim(1) % ALIGN_UNIT_16 != 0),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Last dim size of x1 is not aligned, not changed."),
                    return NOT_CHANGED);

  splitOffset = dimSize;

  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckFusedNodes successful. KeyInfo:[%ld].", splitOffset);

  return SUCCESS;
}

Status ConcatV2DAlignedFusionPass::UpdateShapes(ge::NodePtr beginNode, const ge::NodePtr endNode) const {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "UpdateShapes start.");

  while (beginNode != endNode) {
    auto inputs = beginNode->GetInDataNodes();
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto node = inputs.at(i);
      auto nodeShape = node->GetOpDesc()->MutableOutputDesc(0)->GetShape();
      auto nodeOriShape = node->GetOpDesc()->MutableOutputDesc(0)->GetOriginShape();
      beginNode->GetOpDesc()->MutableInputDesc(i)->SetShape(nodeShape);
      beginNode->GetOpDesc()->MutableInputDesc(i)->SetOriginShape(nodeOriShape);
    }
    FUSION_PASS_CHECK(beginNode->InferShapeAndType() != ge::GRAPH_SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to infer shape. %s.", beginNode->GetName().c_str()),
                      return FAILED);
    beginNode = beginNode->GetOutDataNodes().at(0);
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "UpdateShapes successful.");

  return SUCCESS;
}

Status ConcatV2DAlignedFusionPass::UpdateEdges(vector<ge::NodePtr>& fusedNodes, vector<ge::NodePtr>& newNodes) const {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "UpdateEdges start.");

  OutDataAnchorPtr concatV2DPeerOutAnchor = fusedNodes[CONCATV2D_1_IDX]->GetInDataAnchor(0)->GetPeerOutAnchor();
  InDataAnchorPtr concatV2DInAnchor0 = fusedNodes[CONCATV2D_1_IDX]->GetInDataAnchor(0);
  InDataAnchorPtr padDInAnchor0 = newNodes[PAD1_IDX]->GetInDataAnchor(0);
  OutDataAnchorPtr padDOutAnchor0 = newNodes[PAD1_IDX]->GetOutDataAnchor(0);

  FUSION_PASS_CHECK(
      ge::GraphUtils::RemoveEdge(concatV2DPeerOutAnchor, concatV2DInAnchor0) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove edge. ConcatV2D In[x0]<->Peer."),
      return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(concatV2DPeerOutAnchor, padDInAnchor0) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge. Peer<->PadD."),
                    return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(padDOutAnchor0, concatV2DInAnchor0) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge. PadD<->Peer."),
                    return FAILED);

  OutDataAnchorPtr constOutAnchor0 = fusedNodes[CONSTANT_IDX]->GetOutDataAnchor(0);
  InDataAnchorPtr matMulV2InAnchor1 = fusedNodes[MATMULV2_IDX]->GetInDataAnchor(1);

  FUSION_PASS_CHECK(
      ge::GraphUtils::RemoveEdge(constOutAnchor0, matMulV2InAnchor1) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove edge. MatMulV2 In[1]<->Peer."),
      return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(constOutAnchor0, newNodes[SPLITV_IDX]->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge. SplitV In[0]<->Peer."),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(newNodes[SPLITV_IDX]->GetOutDataAnchor(0),
                                            newNodes[PAD2_IDX]->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge. padD In[0]<->Peer."),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(newNodes[SPLITV_IDX]->GetOutDataAnchor(1),
                                            newNodes[CONCATV2_2_IDX]->GetInDataAnchor(1)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge. padD In[0]<->Peer."),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(newNodes[PAD2_IDX]->GetOutDataAnchor(0),
                                            newNodes[CONCATV2_2_IDX]->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge. padD In[0]<->Peer."),
                    return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(newNodes[MATMULV2_IDX]->GetOutDataAnchor(0), matMulV2InAnchor1) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge. padD In[0]<->Peer."),
                    return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "UpdateEdges successful.");

  return SUCCESS;
}

Status ConcatV2DAlignedFusionPass::CreatePadDNode(ge::ComputeGraph& graph, const ge::OutDataAnchorPtr& preAnchor,
                                                  size_t padDim, ge::NodePtr& padDNode) const {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CreatePadDNode start.");

  auto preNode = preAnchor->GetOwnerNode();
  FUSION_PASS_CHECK(preNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to get node."),
                    return FAILED);

  ge::OpDescPtr padOpDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((padOpDesc = std::make_shared<ge::OpDesc>(preNode->GetName() + "/PadD", "PadD")),
                          return INTERNAL_ERROR);

  auto preTensorDesc = preNode->GetOpDesc()->GetOutputDesc(preAnchor->GetIdx());
  vector<int64_t> preTensorDims = preTensorDesc.GetShape().GetDims();
  FUSION_PASS_CHECK(preTensorDims.size() <= padDim,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to create node. (%d vs %d).", preTensorDims.size(), padDim),
                    return FAILED);

  vector<vector<int64_t>> paddingsValue;
  for (size_t dim = 0; dim < preTensorDims.size(); ++dim) {
    paddingsValue.push_back({0, 0});
  }
  int64_t padValue = ALIGN_UNIT_16 - preTensorDims.at(padDim) % ALIGN_UNIT_16;
  paddingsValue[padDim][1] = padValue;

  FUSION_PASS_CHECK(padOpDesc->AddInputDesc("x", preTensorDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add op input desc."),
                    return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetListListInt(padOpDesc, "paddings", paddingsValue),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to set op attr."), return FAILED);

  vector<int64_t> padDOutDims(preTensorDims);
  padDOutDims[padDim] += padValue;

  auto padDOutTensorDesc = preTensorDesc;
  ge::GeShape padDOutShape(padDOutDims);
  padDOutTensorDesc.SetShape(padDOutShape);
  padDOutTensorDesc.SetOriginShape(padDOutShape);

  FUSION_PASS_CHECK(padOpDesc->AddOutputDesc("y", padDOutTensorDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add op output desc."),
                    return FAILED);

  padDNode = graph.AddNode(padOpDesc);
  FUSION_PASS_CHECK(padDNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add node."),
                    return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "CreatePadDNode successful. KeyInfo:[%d, %ld].", padDim, padValue);

  return SUCCESS;
}

Status ConcatV2DAlignedFusionPass::CreateSplitVNode(ge::ComputeGraph& graph, const ge::OutDataAnchorPtr& preAnchor,
                                                    size_t splitDim, ge::NodePtr& splitVNode) const {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CreateSplitVNode begin.");

  auto preNode = preAnchor->GetOwnerNode();

  std::shared_ptr<ge::OpDesc> splitVOpDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((splitVOpDesc = std::make_shared<ge::OpDesc>(preNode->GetName() + "/SplitV", "SplitV")),
                          return FAILED);
  FUSION_PASS_CHECK(splitVOpDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create node."), return FAILED);

  auto preTensorDesc = preNode->GetOpDesc()->GetOutputDesc(preAnchor->GetIdx());
  splitVOpDesc->AddInputDesc("x", preTensorDesc);

  auto xDims = preTensorDesc.GetShape().GetDims();
  int64_t splitDimSize = xDims[splitDim];
  FUSION_PASS_CHECK(splitDimSize <= splitOffset,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create node. (%ld vs %ld).",
                                                   splitDimSize, splitOffset),
                    return FAILED);

  std::vector<int64_t> y0Dims = xDims;
  y0Dims[splitDim] = splitOffset;
  auto y0TensorDesc = preTensorDesc;
  y0TensorDesc.SetShape(ge::GeShape(y0Dims));
  y0TensorDesc.SetOriginShape(ge::GeShape(y0Dims));
  splitVOpDesc->AddOutputDesc("y0", y0TensorDesc);

  std::vector<int64_t> y1Dims = xDims;
  y1Dims[splitDim] = splitDimSize - splitOffset;
  auto y1TensorDesc = preTensorDesc;
  y1TensorDesc.SetShape(ge::GeShape(y1Dims));
  y1TensorDesc.SetOriginShape(ge::GeShape(y1Dims));
  splitVOpDesc->AddOutputDesc("y1", y1TensorDesc);

  ge::GeTensorDesc sizeSplitsTensorDesc(ge::GeShape({ALLOWED_BLK_N}), ge::FORMAT_ND, DT_INT64);
  splitVOpDesc->AddInputDesc("size_splits", sizeSplitsTensorDesc);

  ge::GeTensorDesc splitDimTensorDesc(ge::GeShape(std::vector<int64_t>({})), ge::FORMAT_ND, DT_INT32);
  splitVOpDesc->AddInputDesc("split_dim", splitDimTensorDesc);

  ge::AttrUtils::SetInt(splitVOpDesc, "num_split", ALLOWED_BLK_N);

  vector<string> constVec = {"size_splits", "split_dim"};
  splitVOpDesc->SetOpInferDepends(constVec);

  splitVNode = graph.AddNode(splitVOpDesc);
  FUSION_PASS_CHECK(splitVNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add node."),
                    return FAILED);

  std::shared_ptr<ge::OpDesc> sizeSplitsOpDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (sizeSplitsOpDesc = std::make_shared<ge::OpDesc>(preNode->GetName() + "/size_splits", "Const")), return FAILED);
  FUSION_PASS_CHECK(sizeSplitsOpDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create node."), return FAILED);

  ge::GeTensorPtr sizeSplitsTensor = nullptr;
  ge::Operator::OpListInt sizeSplitsList = {y0Dims[splitDim], y1Dims[splitDim]};
  FUSION_PASS_MAKE_SHARED(
      (sizeSplitsTensor = std::make_shared<ge::GeTensor>(
           sizeSplitsTensorDesc, reinterpret_cast<uint8_t*>(sizeSplitsList.data()), ALLOWED_BLK_N * sizeof(int64_t))),
      return PARAM_INVALID);

  ge::AttrUtils::SetTensor(sizeSplitsOpDesc, ge::ATTR_NAME_WEIGHTS, sizeSplitsTensor);
  sizeSplitsOpDesc->AddOutputDesc(sizeSplitsTensor->GetTensorDesc());

  auto sizeSplitsNode = graph.AddNode(sizeSplitsOpDesc);
  FUSION_PASS_CHECK(sizeSplitsNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add node."), return FAILED);

  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(sizeSplitsNode->GetOutDataAnchor(0), splitVNode->GetInDataAnchor(1)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge."), return FAILED);

  std::shared_ptr<ge::OpDesc> splitDimOpDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((splitDimOpDesc = std::make_shared<ge::OpDesc>(preNode->GetName() + "/split_dim", "Const")),
                          return FAILED);
  FUSION_PASS_CHECK(splitDimOpDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create node."), return FAILED);

  ge::GeTensorPtr splitDimTensor = nullptr;
  FUSION_PASS_MAKE_SHARED((splitDimTensor = std::make_shared<ge::GeTensor>(
                               splitDimTensorDesc, reinterpret_cast<uint8_t*>(&splitDim), sizeof(splitDim))),
                          return PARAM_INVALID);

  ge::AttrUtils::SetTensor(splitDimOpDesc, ge::ATTR_NAME_WEIGHTS, splitDimTensor);
  splitDimOpDesc->AddOutputDesc(splitDimTensor->GetTensorDesc());

  auto splitDimNode = graph.AddNode(splitDimOpDesc);
  FUSION_PASS_CHECK(splitDimNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add node."), return FAILED);

  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(splitDimNode->GetOutDataAnchor(0), splitVNode->GetInDataAnchor(SPLIT_DIM_IDX)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge."), return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "CreateSplitVNode successful. KeyInfo:[%d, %ld, %ld].", splitDim, splitDimSize,
          splitOffset);

  return SUCCESS;
}

Status ConcatV2DAlignedFusionPass::CreateConcatV2Node(ge::ComputeGraph& graph,
                                                      const vector<ge::OutDataAnchorPtr>& preAnchors, size_t concatDim,
                                                      ge::NodePtr& concatV2Node) const {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CreateConcatV2Node start.");

  auto preNode = preAnchors.at(0)->GetOwnerNode();

  std::shared_ptr<ge::OpDesc> concatV2OpDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((concatV2OpDesc = std::make_shared<ge::OpDesc>(preNode->GetName() + "/ConcatV2", "ConcatV2")),
                          return FAILED);
  FUSION_PASS_CHECK(concatV2OpDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create node."), return FAILED);

  auto preTensorDesc = preNode->GetOpDesc()->GetOutputDesc(preAnchors.at(0)->GetIdx());
  FUSION_PASS_CHECK(concatV2OpDesc->AddInputDesc("x0", preTensorDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add input desc."), return FAILED);

  vector<int64_t> yDims = preTensorDesc.GetShape().GetDims();
  auto yTensorDesc = preTensorDesc;

  auto preAnchorNum = preAnchors.size();
  FUSION_PASS_CHECK(preAnchorNum < ALLOWED_BLK_N,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to create node. (%d, %d).", preAnchorNum, ALLOWED_BLK_N),
                    return NOT_CHANGED);

  for (size_t i = 1; i < preAnchorNum; ++i) {
    preNode = preAnchors.at(i)->GetOwnerNode();
    preTensorDesc = preNode->GetOpDesc()->GetOutputDesc(preAnchors.at(i)->GetIdx());
    string name = "x" + std::to_string(i);
    FUSION_PASS_CHECK(concatV2OpDesc->AddInputDesc(name, preTensorDesc) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add input desc."),
                      return FAILED);
    yDims.at(concatDim) += preTensorDesc.GetShape().GetDims().at(concatDim);
  }

  yTensorDesc.SetShape(ge::GeShape(yDims));
  yTensorDesc.SetOriginShape(ge::GeShape(yDims));
  concatV2OpDesc->AddOutputDesc("y", yTensorDesc);

  ge::GeTensorDesc concatDimTensorDesc(ge::GeShape(std::vector<int64_t>({})), ge::FORMAT_ND, DT_INT32);
  concatV2OpDesc->AddInputDesc("concat_dim", concatDimTensorDesc);

  ge::AttrUtils::SetInt(concatV2OpDesc, "N", preAnchorNum);

  vector<string> constVec = {"concat_dim"};
  concatV2OpDesc->SetOpInferDepends(constVec);

  concatV2Node = graph.AddNode(concatV2OpDesc);
  FUSION_PASS_CHECK(concatV2Node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add node."), return FAILED);

  std::shared_ptr<ge::OpDesc> concatDimOpDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((concatDimOpDesc = std::make_shared<ge::OpDesc>(preNode->GetName() + "/concat_dim", "Const")),
                          return FAILED);
  FUSION_PASS_CHECK(concatDimOpDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create node."), return FAILED);

  ge::GeTensorPtr concatDimTensor = nullptr;
  FUSION_PASS_MAKE_SHARED((concatDimTensor = std::make_shared<ge::GeTensor>(
                               concatDimTensorDesc, reinterpret_cast<uint8_t*>(&concatDim), sizeof(concatDim))),
                          return PARAM_INVALID);

  ge::AttrUtils::SetTensor(concatDimOpDesc, ge::ATTR_NAME_WEIGHTS, concatDimTensor);
  concatDimOpDesc->AddOutputDesc(concatDimTensor->GetTensorDesc());

  auto concatDimNode = graph.AddNode(concatDimOpDesc);
  FUSION_PASS_CHECK(concatDimNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add node."), return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(concatDimNode->GetOutDataAnchor(0),
                                            concatV2Node->GetInDataAnchor(CONCAT_DIM_IDX)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge."), return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "CreateConcatV2Node successful. KeyInfo:[%d, %d].", concatDim, preAnchorNum);

  return SUCCESS;
}

Status ConcatV2DAlignedFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "ConcatV2DAlignedFusionPass start.");

  NodePtr concatV2DNode = GetNodeFromMapping(PATTERN_CONCATV2D, mapping);
  FUSION_PASS_CHECK(concatV2DNode == nullptr, OP_LOGI("ConcatV2 Node is null."), return NOT_CHANGED);
  NodePtr reluNode = GetNodeFromMapping(PATTERN_RELU, mapping);
  FUSION_PASS_CHECK(reluNode == nullptr, OP_LOGI("Relu Node is null."), return NOT_CHANGED);
  NodePtr constantNode = GetNodeFromMapping(PATTERN_CONSTANT, mapping);
  FUSION_PASS_CHECK(constantNode == nullptr, OP_LOGI("Constant Node is null."), return NOT_CHANGED);
  NodePtr matmulV2Node = GetNodeFromMapping(PATTERN_MATMULV2, mapping);
  FUSION_PASS_CHECK(matmulV2Node == nullptr, OP_LOGI("MatmulV2 Node is null."), return NOT_CHANGED);

  vector<ge::NodePtr> fusedNodes = {concatV2DNode, reluNode, constantNode, matmulV2Node};

  FUSION_PASS_CHECK(CheckFusedNodes(fusedNodes) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Does not fit the fusion scene"), return NOT_CHANGED);

  NodePtr newPadDNode = nullptr;
  OutDataAnchorPtr padDPreAnchor = concatV2DNode->GetInDataAnchor(0)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(CreatePadDNode(graph, padDPreAnchor, 1, newPadDNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create node."), return FAILED);
  newNodes.push_back(newPadDNode);

  NodePtr newSplitVNode = nullptr;
  OutDataAnchorPtr splitVPreAnchor = constantNode->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(CreateSplitVNode(graph, splitVPreAnchor, 0, newSplitVNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create node."), return FAILED);
  newNodes.push_back(newSplitVNode);

  NodePtr newPadDNode2 = nullptr;
  OutDataAnchorPtr padDPreAnchor2 = newSplitVNode->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(CreatePadDNode(graph, padDPreAnchor2, 0, newPadDNode2) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create node."), return FAILED);
  newNodes.push_back(newPadDNode2);

  NodePtr newConcatV2Node = nullptr;
  vector<ge::OutDataAnchorPtr> concatV2AllPreAnchor = {newPadDNode2->GetOutDataAnchor(0),
                                                       newSplitVNode->GetOutDataAnchor(1)};
  FUSION_PASS_CHECK(CreateConcatV2Node(graph, concatV2AllPreAnchor, 0, newConcatV2Node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create node."), return FAILED);
  newNodes.push_back(newConcatV2Node);

  FUSION_PASS_CHECK(UpdateEdges(fusedNodes, newNodes) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to update edge."), return FAILED);

  FUSION_PASS_CHECK(UpdateShapes(concatV2DNode, matmulV2Node->GetOutDataNodes().at(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to update shape."), return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "ConcatV2DAlignedFusionPass successful.");

  return SUCCESS;
}

REGISTER_PASS("ZConcatV2DAlignedFusionPass", BUILT_IN_GRAPH_PASS, ConcatV2DAlignedFusionPass);
}  // namespace fe
