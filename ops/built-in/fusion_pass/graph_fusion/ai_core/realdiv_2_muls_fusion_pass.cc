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
 * \file realdiv_2_muls_fusion_pass.cc
 * \brief realdiv fusion pass( --> muls)
 */
#include "realdiv_2_muls_fusion_pass.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace std;
using namespace ge;

namespace fe {
static const char* REALDIV = "RealDiv";
static const char* MULS = "Muls";
static const string PATTERN_REALDIV= "RealDiv";
static const string CONSTANT = "Const";
static const string CONSTANTOP = "Constant";
static const float EPSILON = 1e-6;
static const size_t FLOATBYTES = 4;
static const size_t NUM2 = 2;

vector<FusionPattern*> RealDiv2MulsFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("RealDiv2MulsFusionPass");
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter RealDiv2MulsFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_REALDIV, {REALDIV}).SetOutput(PATTERN_REALDIV);
  patterns.push_back(pattern);
  return patterns;
}

Status RealDiv2MulsFusionPass::IsMatch(ge::NodePtr& realDivNode, float& constValue) const {
  ge::ConstGeTensorPtr constTensor = nullptr;
  ge::InDataAnchorPtr realDivInputAnchorPtr1 = realDivNode->GetInDataAnchor(1);
  ge::OutDataAnchorPtr peerOutAnchorPtr = realDivInputAnchorPtr1->GetPeerOutAnchor();
  ge::NodePtr peerOutNode = peerOutAnchorPtr->GetOwnerNode();
  std::string opType = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(peerOutNode);
  if (opType != CONSTANT && opType != CONSTANTOP) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Realiv second input is not const.");
    return NOT_CHANGED;
  }
  vector<ge::GeTensorPtr> constTensortPtr = ge::OpDescUtils::MutableWeights(peerOutNode);
  FUSION_PASS_CHECK(constTensortPtr.empty(), OP_LOGI(FUSED_OP_TYPE.c_str(), "RealDiv input y is tensor!"),
                   return NOT_CHANGED);
  ge:: ConstGeTensorPtr constTensor0 = constTensortPtr[0];
  size_t constSize = constTensor0->GetData().GetSize();
  ge::DataType constDType = constTensor0->GetTensorDesc().GetDataType();
  if (constDType != ge::DT_FLOAT || constSize != FLOATBYTES) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Realiv second input Type is not float or Scalar.");
    return NOT_CHANGED;
  }
  if (constTensor0->GetData().GetData() != nullptr) {
    float* constDataPtr = nullptr;
    constDataPtr = (float*)constTensor0->GetData().GetData();
    constValue = static_cast<float>(*constDataPtr);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "RealDiv second input Value is %f", constValue);
  } else {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "RealDiv second input tensor is null.");
    return NOT_CHANGED;
  }
  if (fabs(constValue - 0.0) <= EPSILON) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "RealDiv second value is close 0.0.");
    return NOT_CHANGED;
  }

  return SUCCESS;
}

Status RealDiv2MulsFusionPass::ReLinkControlAnchor(ge::NodePtr& realDivNode, ge::NodePtr& mulsNode) {
  InControlAnchorPtr realDivInControlAnchorPtr = realDivNode->GetInControlAnchor();
  InControlAnchorPtr mulsInControlAnchorPtr = mulsNode->GetInControlAnchor();
  if (realDivInControlAnchorPtr != nullptr) {
    for (OutControlAnchorPtr outControlAnchorPtr : realDivInControlAnchorPtr->GetPeerOutControlAnchors()) {
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(outControlAnchorPtr, realDivInControlAnchorPtr),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                      "remove input control edge failed"), return FAILED);
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(outControlAnchorPtr, mulsInControlAnchorPtr),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                      "add input control edge failed"), return FAILED);
    }
  }
  return SUCCESS;
}

Status RealDiv2MulsFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define RealDiv2MulsFusionPass fusion begin");
  ge::NodePtr realDivNode = GetNodeFromMapping(PATTERN_REALDIV, mapping);
  FUSION_PASS_CHECK(realDivNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "realdiv node is null."),
                    return PARAM_INVALID);

  ge::OpDescPtr realDivOpDesc = realDivNode->GetOpDesc();
  FUSION_PASS_CHECK(realDivOpDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "RealDiv OpDesc is null."),
                    return PARAM_INVALID);

  FUSION_PASS_CHECK(realDivNode->GetInDataNodes().size() != NUM2,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "RealDiv input size is not 2."),
                    return PARAM_INVALID);

  ge::GeShape inputShape = realDivOpDesc->GetInputDesc(0).GetShape();
  FUSION_PASS_CHECK(inputShape.IsUnknownShape(),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "RealDiv2Muls FusionPass not support dynamic shape"),
                    return NOT_CHANGED);

  float constValue = 0.0;
  if (IsMatch(realDivNode, constValue) != SUCCESS) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Node[s%] don't match RealDiv2Muls fusion pattern.",
            realDivNode->GetName().c_str());
    return NOT_CHANGED;
  }
  ge::GeTensorDesc inputDesc0 = realDivOpDesc->GetInputDesc(0);
  ge::GeTensorDesc outputDesc = realDivOpDesc->GetOutputDesc(0);
  ge::OpDescPtr mulsOpDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((mulsOpDesc = std::make_shared<ge::OpDesc>(realDivNode->GetName() + "/" + MULS, MULS)),
                           return INTERNAL_ERROR);
  mulsOpDesc->AddInputDesc("x", inputDesc0);
  mulsOpDesc->AddOutputDesc("y", outputDesc);
  ge::AttrUtils::SetFloat(mulsOpDesc, "value", (1.0 / constValue));
  ge::NodePtr mulsNode = graph.AddNode(mulsOpDesc);
  FUSION_PASS_CHECK(mulsNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Created Muls node is null, fusion failed."), return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(realDivNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                    mulsNode->GetInDataAnchor(0)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add muls node input data edge failed."),
                    return FAILED);
  for (InDataAnchorPtr inAnchorPtr : realDivNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    inAnchorPtr->UnlinkAll();
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(mulsNode->GetOutDataAnchor(0), inAnchorPtr),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "Add edge from node:%s to fusion "
                                                     "node:%s 's out edge failed.",
                                                     mulsNode->GetName().c_str(), realDivNode->GetName().c_str()),
                      return FAILED);
  }
  if (ReLinkControlAnchor(realDivNode, mulsNode) != SUCCESS) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "process %s and %s control link failed",
            realDivNode->GetName().c_str(), mulsNode->GetName().c_str());
    return FAILED;
  }
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(realDivNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                          realDivNode->GetInDataAnchor(0)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove RealDiv node in data0 edge failed."),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(realDivNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                                          realDivNode->GetInDataAnchor(1)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove RealDiv node in data1 edge failed."),
                    return FAILED);

  FUSION_PASS_CHECK(graph.RemoveNode(realDivNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove RealDiv node failed."), return FAILED);
  newNodes.push_back(mulsNode);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Node[%s] do RealDiv2Muls fusion success!", mulsNode->GetName().c_str());

  return SUCCESS;
}
REGISTER_PASS("RealDiv2MulsFusionPass", BUILT_IN_GRAPH_PASS, RealDiv2MulsFusionPass);
}  // namespace fe
