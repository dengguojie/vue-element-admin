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
 * \file derelu_fusion_pass.cpp
 * \brief derelu fusion pass
 */
#include "derelu_fusion_pass.h"
#include <string>
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"

namespace fe {
static const string PATTERN_RELUGRAD = "reluGrad";
static const string PATTERN_RELU = "relu";
static const int32_t INT_NUM_TWO = 2;
static const string OP_RELU = "Relu";
static const string RELU_V2 = "ReluV2";
static const string RELUGRAD = "ReluGrad";
static const string RELUGRAD_V2 = "ReluGradV2";
static const string PATTERN_INPUTS = "input";
// unknown shape value
const int64_t UNKNOWN_SHAPE_VALUE = -1;
const int64_t SHAPE_UNKNOWN_DIM_NUM = -2;

bool DreluFusionPass::IsUnknownShape(const ge::GeShape& shape) {
  std::vector<int64_t> dims = shape.GetDims();
  size_t dimsSize = dims.size();
  for (size_t i = 0; i < dimsSize; i++) {
    if (dims[i] == UNKNOWN_SHAPE_VALUE || dims[i] == SHAPE_UNKNOWN_DIM_NUM) {
      return true;
    }
  }
  return false;
}

vector<FusionPattern*> DreluFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define DreluFusionPass pattern begin");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("DreluFusionPass");

  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_RELUGRAD, {RELUGRAD})
      .AddOpDesc(PATTERN_RELU, {OP_RELU})
      .AddOpDesc(PATTERN_INPUTS)
      .SetInputs(PATTERN_RELUGRAD, {PATTERN_INPUTS, PATTERN_RELU})
      .SetOutput(PATTERN_RELUGRAD);
  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define DreluFusionPass pattern end");
  return patterns;
}

Status DreluFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define DreluFusionPass fusion begin");
  ge::NodePtr relu = GetNodeFromMapping(PATTERN_RELU, mapping);
  ge::NodePtr reluGrad = GetNodeFromMapping(PATTERN_RELUGRAD, mapping);
  int64_t reluGradCount = 0;
  for (auto peerInDataAnchor : relu->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    ge::NodePtr nextNode = peerInDataAnchor->GetOwnerNode();
    if (nextNode->GetType() == RELUGRAD) {
      reluGradCount++;
    }
  }
  FUSION_PASS_CHECK(reluGradCount > 1, OP_LOGI(FUSED_OP_TYPE.c_str(), "Relu have multiple output, can not fusion."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(relu == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "relu is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(reluGrad == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reluGrad is null, fusion failed."),
                    return PARAM_INVALID);

  ge::NodePtr relu2 = CreateNode(graph, relu, fusionNodes);
  FUSION_PASS_CHECK(relu2 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "relu2 is null, fusion failed."),
                    return PARAM_INVALID);

  if (relu2->GetOpDesc()->InferShapeAndType() != ge::GRAPH_SUCCESS) {
    FUSION_PASS_CHECK(graph.RemoveNode(relu2) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove ReluV2 failed."),
                      return FAILED);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "ReluV2 InferShapeAndType failed, can not fusion.");
    return NOT_CHANGED;
  }

  FUSION_PASS_CHECK(ReplaceNode(relu, relu2, graph) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "replace node failed"),
                    return FAILED);

  ge::OpDescPtr reluDesc = relu2->GetOpDesc();
  if (reluDesc->GetAllOutputsDescSize() < INT_NUM_TWO) {
    return FAILED;
  }
  if (IsUnknownShape(relu2->GetOpDesc()->GetOutputDesc(1).GetShape())) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "DreluFusionPass cannot be applied for unknown shape.");
    return SUCCESS;
  }
  std::vector<int64_t> relu2Dims = relu2->GetOpDesc()->GetOutputDesc(1).GetShape().GetDims();
  relu2->GetOpDesc()->MutableOutputDesc(1)->SetOriginDataType(relu2->GetOpDesc()->GetOutputDesc(1).GetDataType());

  reluGrad->GetOpDesc()->SetType(RELUGRAD_V2);
  reluGrad->GetOpDesc()->MutableInputDesc(1)->SetShape(ge::GeShape(relu2Dims));
  reluGrad->GetOpDesc()->MutableInputDesc(1)->SetOriginShape(ge::GeShape(relu2Dims));
  reluGrad->GetOpDesc()->MutableInputDesc(1)->SetDataType(ge::DT_UINT8);
  auto reluGradInDataAnchor = reluGrad->GetInDataAnchor(1);
  FUSION_PASS_CHECK(reluGradInDataAnchor == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reluGradInDataAnchor is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(reluGrad == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reluGrad is null, fusion failed."),
                    return PARAM_INVALID);
  auto reluGradPeerOutDataAnchor = reluGradInDataAnchor->GetPeerOutAnchor();

  auto relu2OutDataAnchor = relu2->GetOutDataAnchor(1);

  FUSION_PASS_CHECK(relu2OutDataAnchor == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "relu2OutDataAnchor is null, fusion failed."), return PARAM_INVALID);
  // delete edge
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(reluGradPeerOutDataAnchor, reluGradInDataAnchor) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove inputdata edge error"), return FAILED);
  // add edge
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(relu2OutDataAnchor, reluGradInDataAnchor) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input edge error"), return FAILED);
  reluGrad->GetOpDesc()->MutableInputDesc(1)->SetOriginDataType(reluGrad->GetOpDesc()->GetInputDesc(1).GetDataType());
  fusionNodes.push_back(reluGrad);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define DreluFusionPass fusion end");
  return SUCCESS;
}

Status DreluFusionPass::RemoveNode(ge::NodePtr node, ge::ComputeGraph& graph) {
  // remove input data edge
  for (size_t i = 0; i < node->GetAllInDataAnchors().size(); ++i) {
    auto inDataAnchor = node->GetInDataAnchor(i);
    FUSION_PASS_CHECK(inDataAnchor == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inDataAnchor is null, fusion failed."),
                      return PARAM_INVALID);
    auto preOutDataAnchor = inDataAnchor->GetPeerOutAnchor();
    if (preOutDataAnchor == nullptr) {
      continue;
    }
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(preOutDataAnchor, inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove inputdata edge error"), return FAILED);
  }
  // delete node
  FUSION_PASS_CHECK(graph.RemoveNode(node) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove node failed"),
                    return FAILED);
  return SUCCESS;
}

ge::NodePtr DreluFusionPass::CreateNode(ge::ComputeGraph& graph, ge::NodePtr relu, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr relu2 = nullptr;
  ge::OpDescPtr opDesc = ge::AttrUtils::CopyOpDesc(relu->GetOpDesc());
  opDesc->SetType(RELU_V2);
  ge::GeTensorDesc tensorDesc;
  tensorDesc.SetFormat(relu->GetOpDesc()->GetOutputDesc(0).GetFormat());
  tensorDesc.SetOriginFormat(relu->GetOpDesc()->GetOutputDesc(0).GetOriginFormat());
  opDesc->AddOutputDesc("mask", tensorDesc);
  opDesc->AddInferFunc(nullptr);
  opDesc->AddInferFormatFunc(nullptr);
  opDesc->AddVerifierFunc(nullptr);
  relu2 = graph.AddNode(opDesc);
  fusionNodes.push_back(relu2);
  return relu2;
}

Status DreluFusionPass::ReplaceNode(ge::NodePtr oldNode, ge::NodePtr newNode, ge::ComputeGraph& graph) {
  // input data edge
  for (size_t i = 0; i < oldNode->GetAllInDataAnchors().size(); ++i) {
    auto inDataAnchor = oldNode->GetInDataAnchor(static_cast<int>(i));
    auto inDataAnchorNew = newNode->GetInDataAnchor(static_cast<int>(i));
    FUSION_PASS_CHECK(inDataAnchor == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inDataAnchor is null, fusion failed."),
                      return PARAM_INVALID);
    FUSION_PASS_CHECK(inDataAnchorNew == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inDataAnchorNew is null, fusion failed."), return PARAM_INVALID);

    auto preOutDataAnchor = inDataAnchor->GetPeerOutAnchor();
    if (preOutDataAnchor != nullptr) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(preOutDataAnchor, inDataAnchor) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove inputdata edge error"), return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(preOutDataAnchor, inDataAnchorNew) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input edge error"), return FAILED);
    }
  }
  // input control edge
  auto inCtrlAnchor = oldNode->GetInControlAnchor();
  auto inCtrlAnchorNew = newNode->GetInControlAnchor();
  FUSION_PASS_CHECK(inCtrlAnchor == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inCtrlAnchor is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(inCtrlAnchorNew == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inCtrlAnchorNew is null, fusion failed."), return PARAM_INVALID);

  for (auto preOutCtrlAnchor : inCtrlAnchor->GetPeerOutControlAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(preOutCtrlAnchor, inCtrlAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove input control edge error"), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(preOutCtrlAnchor, inCtrlAnchorNew) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input control edge error"), return FAILED);
  }
  for (auto outPeerDataAnchor : inCtrlAnchor->GetPeerOutDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(outPeerDataAnchor, inCtrlAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove input control edge error"), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(outPeerDataAnchor, inCtrlAnchorNew) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input control edge error"), return FAILED);
  }
  // output data edge
  for (size_t i = 0; i < oldNode->GetAllOutDataAnchors().size(); ++i) {
    auto outDataAnchor = oldNode->GetOutDataAnchor(static_cast<int>(i));
    auto outDataAnchorNew = newNode->GetOutDataAnchor(static_cast<int>(i));
    if (outDataAnchor == nullptr) {
      continue;
    }
    for (auto nextInDataAnchor : outDataAnchor->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(outDataAnchor, nextInDataAnchor) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove outData edge error"), return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(outDataAnchorNew, nextInDataAnchor) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add outData edge error"), return FAILED);
    }
    for (auto nextInCtrlAnchor : outDataAnchor->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(outDataAnchor, nextInCtrlAnchor) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove outData edge error"), return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(outDataAnchorNew, nextInCtrlAnchor) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add outData edge error"), return FAILED);
    }
  }

  // output control edge
  auto outCtrlAnchor = oldNode->GetOutControlAnchor();
  FUSION_PASS_CHECK(outCtrlAnchor == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "outCtrlAnchor is null, fusion failed."),
                    return PARAM_INVALID);
  auto outCtrlAnchorNew = newNode->GetOutControlAnchor();
  FUSION_PASS_CHECK(outCtrlAnchorNew == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "outCtrlAnchorNew is null, fusion failed."), return PARAM_INVALID);
  for (auto nextInCtrlAnchor : outCtrlAnchor->GetPeerInControlAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(outCtrlAnchor, nextInCtrlAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove outControl edge error"), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(outCtrlAnchorNew, nextInCtrlAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add outControl edge error"), return FAILED);
  }
  // delete old node
  FUSION_PASS_CHECK(RemoveNode(oldNode, graph) != SUCCESS, , return FAILED);
  return SUCCESS;
}
REGISTER_PASS("DreluFusionPass", BUILT_IN_GRAPH_PASS, DreluFusionPass);
}  // namespace fe
