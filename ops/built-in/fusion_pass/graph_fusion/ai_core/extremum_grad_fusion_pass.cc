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
 * \file extremum_grad_fusion_pass.cpp
 * \brief Fusion Pass for full structure of MaximumGrad/MinimumGrad(only Dx,
 *   only Dy, Dx & Dy) with/without sum
 */
#include "extremum_grad_fusion_pass.h"
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "framework/common/string_util.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "register/op_registry.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph_optimizer/fusion_common/fusion_statistic_recorder.h"

namespace fe {
static const string ATTR_DATA_TYPE = "T";
static const string ATTR_GRAD_X = "grad_x";
static const string ATTR_GRAD_Y = "grad_y";

static const char GREATER_EQUAL[] = "GreaterEqual";
static const char LESS_EQUAL[] = "LessEqual";
static const char MAXIMUM_GRAD[] = "MaximumGrad";
static const char MINIMUM_GRAD[] = "MinimumGrad";

static const char SCOPE_MAXIMUM_GRAD[] = "Maximum_grad";
static const char SCOPE_MINIMUM_GRAD[] = "Minimum_grad";

static const char PATTERN_EQUAL[] = "equal";
static const char PATTERN_ZEROS[] = "zeros";
static const char PATTERN_DZ[] = "intput_dz";
static const char PATTERN_SELECT_DX[] = "select_dx";
static const char PATTERN_SELECT_DY[] = "select_dy";
static const char PATTERN_SUM_DX[] = "sum_dx";
static const char PATTERN_SUM_DY[] = "sum_dy";

static const char* SUM = "Sum";
static const char* SELECT = "Select";
static const char* REDUCESUMD = "ReduceSumD";
static const char* REDUCESUM = "ReduceSum";

static const int EQUAL_INPUT_NUM = 2;
static const int EQUAL_OUTPUT_NUM = 1;

static const int SELECT_INPUT_NUM = 3;
static const int SELECT_OUTPUT_NUM = 1;

static const int SELECT_INPUT_0 = 0;
static const int SELECT_INPUT_1 = 1;
static const int SELECT_INPUT_2 = 2;

static const int SUM_OUTPUT_NUM = 1;
static const int EQUAL_TO_EXTREMUM_START = 1;
static const int EQUAL_WITH_ONE_OUTPUT_NODE = 1;
static const int EQUAL_WITH_TWO_OUTPUT_NODE = 2;
static const int NAME_SCOPE_BACK_INDEX = 2;

static const string STREAM_LABEL = "_stream_label";

std::vector<FusionPattern*> ExtremumGradFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  return patterns;
}

Status ExtremumGradFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                      std::vector<ge::NodePtr>& fusionNodes) {
  return SUCCESS;
}

Status ExtremumGradFusionPass::Run(ge::ComputeGraph& graph) {
  return Run(graph, nullptr);
}
Status ExtremumGradFusionPass::Run(ge::ComputeGraph& graph, OpsKernelInfoStorePtr opsKernelInfoStorePtr) {
  // Step1: Record all GreaterEqual, LessEqual op in graph
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to match ExtremumGradFusionPass.");
  vector<ge::NodePtr> nodeEquals;
  for (ge::NodePtr &n : graph.GetDirectNode()) {
    if (n->GetOpDesc()->GetType() == GREATER_EQUAL || n->GetOpDesc()->GetType() == LESS_EQUAL) {
      nodeEquals.push_back(n);
    }
  }
  // Step2: For each equal op, do fusion operation
  Status finalRet = NOT_CHANGED;
  // record match and effect times match times == effect times
  int32_t matchTimes = 0;
  for (ge::NodePtr &nodeEqual : nodeEquals) {
    Status ret = RunOnePatternFusion(graph, nodeEqual);
    if (ret == SUCCESS) {
      matchTimes++;
      finalRet = SUCCESS;
    } else if (ret != NOT_CHANGED) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Run one pattern fusion failed.");
      return FAILED;
    }
  }
  // save matchTimes and effectTimes
  FusionInfo fusionInfo(graph.GetSessionID(), to_string(graph.GetGraphID()), GetName(), matchTimes, matchTimes);
  FusionStatisticRecorder::Instance().UpdateGraphFusionMatchTimes(fusionInfo);
  FusionStatisticRecorder::Instance().UpdateGraphFusionEffectTimes(fusionInfo);
  OP_LOGD(FUSED_OP_TYPE.c_str(),
          "SessionId[%d], GraphId[%d], GraphFusionPass[%s]: pattern=undefined, matchedTimes=%d, effectedTimes=%d.",
          graph.GetSessionID(), graph.GetGraphID(), GetName().c_str(), matchTimes, matchTimes);
  return finalRet;
}
/*
 * Check imply type
 *
 */
bool ExtremumGradFusionPass::CheckImplyType() const {
  domi::ImplyType greaterEqualImplyType = domi::OpRegistry::Instance()->GetImplyType(GREATER_EQUAL);
  domi::ImplyType lessEqualImplyType = domi::OpRegistry::Instance()->GetImplyType(LESS_EQUAL);
  if (greaterEqualImplyType != domi::ImplyType::TVM && lessEqualImplyType != domi::ImplyType::TVM) {
    OP_LOGD(FUSED_OP_TYPE.c_str(),
            "There is no GreaterEqual or LessEqual op,"
            "exit ExtremumGradFusionPass.");
    return false;
  }
  return true;
}
/*
 * Check nameA & nameB have same scope or not
 */
bool ExtremumGradFusionPass::CheckNameScope(const string& nameA, const string& nameB) const {
  vector<string> nameVecA = ge::StringUtils::Split(nameA, '/');
  vector<string> nameVecB = ge::StringUtils::Split(nameB, '/');

  if ((nameVecA.size() != nameVecB.size())) {
    return false;
  }

  nameVecA.erase(nameVecA.end() - 1);
  nameVecB.erase(nameVecB.end() - 1);

  return nameVecA == nameVecB;
}

bool ExtremumGradFusionPass::CheckZeroConstantOp(ge::NodePtr nodeZeros) const {
  if (nodeZeros == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "nodeZeros is nullptr");
    return false;
  }

  if (ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(nodeZeros) != "Constant") {
    return false;
  }

  return true;
}

bool ExtremumGradFusionPass::CheckSelectOp(const ge::NodePtr& nodeSelect, const ge::NodePtr& nodeEqual) const {
  if (nodeSelect == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "select node is null");
    return false;
  }

  if (nodeSelect->GetOpDesc()->GetType() != SELECT) {
    return false;
  }

  if (!CheckNameScope(nodeSelect->GetName(), nodeEqual->GetName())) {
    return false;
  }

  if (nodeSelect->GetAllInDataAnchors().size() != SELECT_INPUT_NUM) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Select op should have 3 input, actually is %zu.",
            nodeSelect->GetAllInDataAnchors().size());
    return false;
  }

  if (nodeSelect->GetAllOutDataAnchors().size() != SELECT_OUTPUT_NUM) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Select op should have 1 output, actually is %zu.",
            nodeSelect->GetAllOutDataAnchors().size());
    return false;
  }
  return true;
}

bool ExtremumGradFusionPass::CheckSumOp(ge::NodePtr nodeSum, ge::NodePtr nodeEqual) const {
  if (nodeSum == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "node sum is nullptr");
    return false;
  }

  if (nodeSum->GetOpDesc()->GetType() != SUM && nodeSum->GetOpDesc()->GetType() != REDUCESUMD) {
    return false;
  }

  if (!CheckNameScope(nodeSum->GetName(), nodeEqual->GetName())) {
    return false;
  }

  if (nodeSum->GetAllOutDataAnchors().size() != SUM_OUTPUT_NUM) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Sum op should have 1 output, actually is %zu.",
            nodeSum->GetAllOutDataAnchors().size());
    return false;
  }
  return true;
}

bool ExtremumGradFusionPass::CheckSameZeroNode(ge::NodePtr nodeZeros, const map<string, ge::NodePtr>& recordMap) {
  if (nodeZeros == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "nodezero is nullptr");
    return false;
  }

  auto iterZeros = recordMap.find(PATTERN_ZEROS);
  if (iterZeros == recordMap.end()) {
    if (!CheckZeroConstantOp(nodeZeros)) {
      return false;
    }
  } else if (iterZeros->second != nodeZeros) {
    return false;
  }
  return true;
}

bool ExtremumGradFusionPass::MatchDx(ge::NodePtr nodeSelect, map<string, ge::NodePtr>& recordMap) {
  // Step1: Get GreaterEqual/LessEqual node from record map
  ge::NodePtr nodeEqual = nullptr;
  auto iterEqual = recordMap.find(PATTERN_EQUAL);
  if (iterEqual == recordMap.end() || iterEqual->second == nullptr) {
    return false;
  } else {
    nodeEqual = iterEqual->second;
  }

  // Step2: Check node of dx part in record map cannot be record already
  if ((recordMap.find(PATTERN_SELECT_DX) != recordMap.end()) || (recordMap.find(PATTERN_SUM_DX) != recordMap.end())) {
    return false;
  }

  // Step3: Check nodeSelect's validity (3 input anchor, 1 output anchor, have
  // same scope with nodeEqual)
  if (!CheckSelectOp(nodeSelect, nodeEqual)) {
    return false;
  }

  // Step4: Check select input0 come from PATTERN_EQUAL
  ge::OutDataAnchorPtr input0PeerAnchor = nodeSelect->GetInDataAnchor(SELECT_INPUT_0)->GetPeerOutAnchor();
  if (input0PeerAnchor == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "inputanchor is nullptr");
    return false;
  }

  if (input0PeerAnchor->GetOwnerNode() != nodeEqual) {
    return false;
  }

  // Step5: Check select input1 come from PATTERN_DZ
  ge::OutDataAnchorPtr input1PeerAnchor = nodeSelect->GetInDataAnchor(SELECT_INPUT_1)->GetPeerOutAnchor();
  if (input1PeerAnchor == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "input anchor 1 is nullptr");
    return false;
  }

  ge::NodePtr nodeDz = input1PeerAnchor->GetOwnerNode();
  if (nodeDz == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "node Dz is nullptr");
    return false;
  }

  auto iterDz = recordMap.find(PATTERN_DZ);
  if (iterDz != recordMap.end() && iterDz->second != nodeDz) {
    return false;
  }

  // Step6: Check select input2 come from zero constant op
  ge::OutDataAnchorPtr input2PeerAnchor = nodeSelect->GetInDataAnchor(SELECT_INPUT_2)->GetPeerOutAnchor();
  if (input2PeerAnchor == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "input anchor 2 is nullptr");
    return false;
  }
  ge::NodePtr nodeZeros = input2PeerAnchor->GetOwnerNode();
  if (!CheckSameZeroNode(nodeZeros, recordMap)) {
    return false;
  }

  // Step7: Record PATTERN_SELECT_DX:nodeSelect,
  //         PATTERN_DZ:nodeDz
  //         PATTERN_ZEROS:nodeZeros
  recordMap[PATTERN_SELECT_DX] = nodeSelect;
  recordMap[PATTERN_DZ] = nodeDz;
  recordMap[PATTERN_ZEROS] = nodeZeros;

  // Step8: If there is only one Sum OP after nodeSelect, check whether its
  //    scope same with nodeSelect, if same, record it as PATTERN_SUM_DX
  auto outputNodes = nodeSelect->GetOutDataNodes();
  if (outputNodes.size() == 1) {
    ge::NodePtr nodeSum = outputNodes.at(0);
    if (CheckSumOp(nodeSum, nodeEqual)) {
      recordMap[PATTERN_SUM_DX] = nodeSum;
    }
    // if can't get ReduceSumD, check data type,
    // if type is int32, fusion break off
    ge::GeTensorDesc DataTensor = nodeSum->GetOpDesc()->GetInputDesc(0);
    ge::DataType dataType = DataTensor.GetDataType();
    if (dataType == ge::DT_INT32 && nodeSum->GetOpDesc()->GetType() == REDUCESUM) {
      return false;
    }
  }
  return true;
}

bool ExtremumGradFusionPass::MatchDy(ge::NodePtr nodeSelect, map<string, ge::NodePtr>& recordMap) {
  // Step1: Get GreaterEqual/LessEqual node from record map
  ge::NodePtr nodeEqual = nullptr;
  auto iterEqual = recordMap.find(PATTERN_EQUAL);
  if (iterEqual == recordMap.end() || iterEqual->second == nullptr) {
    return false;
  } else {
    nodeEqual = iterEqual->second;
  }

  // Step2: Check node of dy part in record map cannot be record already
  if ((recordMap.find(PATTERN_SELECT_DY) != recordMap.end()) || (recordMap.find(PATTERN_SUM_DY) != recordMap.end())) {
    return false;
  }

  // Step3: Check nodeSelect's validity (3 input anchor, 1 output anchor)
  if (!CheckSelectOp(nodeSelect, nodeEqual)) {
    return false;
  }

  // Step4: Check select input0 come from PATTERN_EQUAL
  ge::OutDataAnchorPtr input0PeerAnchor = nodeSelect->GetInDataAnchor(SELECT_INPUT_0)->GetPeerOutAnchor();
  if (input0PeerAnchor == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "select node input 0 is null");
    return false;
  }
  if (input0PeerAnchor->GetOwnerNode() != nodeEqual) {
    return false;
  }

  // Step5: Check select input1 come from zero constant op
  ge::OutDataAnchorPtr input1PeerAnchor = nodeSelect->GetInDataAnchor(SELECT_INPUT_1)->GetPeerOutAnchor();
  if (input1PeerAnchor == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "select node input 1 is null");
    return false;
  }

  ge::NodePtr nodeZeros = input1PeerAnchor->GetOwnerNode();
  if (CheckSameZeroNode(nodeZeros, recordMap) == false) {
    return false;
  }

  // Step6: Check select input2 come from PATTERN_DZ
  ge::OutDataAnchorPtr input2PeerAnchor = nodeSelect->GetInDataAnchor(SELECT_INPUT_2)->GetPeerOutAnchor();
  if (input2PeerAnchor == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "select input 2 is null");
    return false;
  }

  ge::NodePtr nodeDz = input2PeerAnchor->GetOwnerNode();
  if (nodeDz == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "node dz is null");
    return false;
  }

  auto iterDz = recordMap.find(PATTERN_DZ);
  if (iterDz != recordMap.end() && iterDz->second != nodeDz) {
    return false;
  }

  // Step7: Record PATTERN_SELECT_DY:nodeSelect,
  //         PATTERN_DZ:nodeDz
  //         PATTERN_ZEROS:nodeZeros
  recordMap[PATTERN_SELECT_DY] = nodeSelect;
  recordMap[PATTERN_DZ] = nodeDz;
  recordMap[PATTERN_ZEROS] = nodeZeros;

  // Step8: If there is only one Sum OP after nodeSelect, check whether its
  //    scope same with nodeSelect, if same, record it as PATTERN_SUM_DY
  auto outputNodes = nodeSelect->GetOutAllNodes();
  if (outputNodes.size() == 1) {
    ge::NodePtr nodeSum = outputNodes.at(0);
    if (CheckSumOp(nodeSum, nodeEqual)) {
      recordMap[PATTERN_SUM_DY] = nodeSum;
    }
    // if can't get ReduceSumD, check data type,
    // if type is int32, fusion break off
    ge::GeTensorDesc DataTensor = nodeSum->GetOpDesc()->GetInputDesc(0);
    ge::DataType dataType = DataTensor.GetDataType();
    if (dataType == ge::DT_INT32 && nodeSum->GetOpDesc()->GetType() == REDUCESUM) {
      return false;
    }
  }
  return true;
}

bool ExtremumGradFusionPass::CheckEqualOp(ge::NodePtr nodeEqual) const {
  if (nodeEqual == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "node is null");
    return false;
  }
  // Check namescope of equal, MaximumGrad should be "**/Maximum_grad/**"
  //                           MinimumGrad should be "**/Minimum_grad/**"
  vector<string> nameScope = ge::StringUtils::Split(nodeEqual->GetName(), '/');
  if (nameScope.size() <= 1) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Name scope should be more than one");
    return false;
  }

  string scopeUpper = nameScope[nameScope.size() - NAME_SCOPE_BACK_INDEX];

  if (nodeEqual->GetOpDesc()->GetType() == GREATER_EQUAL && scopeUpper != SCOPE_MAXIMUM_GRAD) {
    return false;
  } else if (nodeEqual->GetOpDesc()->GetType() == LESS_EQUAL && scopeUpper != SCOPE_MINIMUM_GRAD) {
    return false;
  }

  if (nodeEqual->GetAllInDataAnchors().size() != EQUAL_INPUT_NUM) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Equal op should have 2 input anchor, actual is %zu",
            nodeEqual->GetAllInDataAnchors().size());
    return NOT_CHANGED;
  }

  if (nodeEqual->GetAllOutDataAnchors().size() != EQUAL_OUTPUT_NUM) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Equal op should have 1 output anchor, actual is %zu",
            nodeEqual->GetAllOutDataAnchors().size());
    return NOT_CHANGED;
  }

  return true;
}

Status ExtremumGradFusionPass::RunOnePatternFusion(ge::ComputeGraph& graph, const ge::NodePtr& nodeEqual) {
  map<string, ge::NodePtr> recordMap;
  // Step1: check nodeEqual's validity, two input data anchor, one output data
  // anchor
  if (!CheckEqualOp(nodeEqual)) {
    return NOT_CHANGED;
  }

  recordMap[PATTERN_EQUAL] = nodeEqual;

  ge::OutDataAnchorPtr equalOutAnchor = nodeEqual->GetOutDataAnchor(0);
  if (equalOutAnchor == nullptr) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[equalOutAnchor] must not be null.");
    return fe::PARAM_INVALID;
  }
  // Step2: if nodeEqual has two output select node, check that one should be
  // select_dx,
  //    another one should be select_dy
  if (equalOutAnchor->GetPeerInDataAnchors().size() == EQUAL_WITH_TWO_OUTPUT_NODE) {
    auto peerAnchor0 = equalOutAnchor->GetPeerInDataAnchors().at(0);
    if (peerAnchor0 == nullptr) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[peerAnchor0] must not be null.");
      return fe::PARAM_INVALID;
    }
    ge::NodePtr node0 = peerAnchor0->GetOwnerNode();
    if (node0 == nullptr) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[node0] must not be null.");
      return fe::PARAM_INVALID;
    }

    auto peerAnchor1 = equalOutAnchor->GetPeerInDataAnchors().at(1);
    if (peerAnchor1 == nullptr) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[peerAnchor1] must not be null.");
      return fe::PARAM_INVALID;
    }
    ge::NodePtr node1 = peerAnchor1->GetOwnerNode();
    if (node1 == nullptr) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[node1] must not be null.");
      return fe::PARAM_INVALID;
    }
    if ((MatchDx(node0, recordMap) && MatchDy(node1, recordMap)) ||
        (MatchDx(node1, recordMap) && MatchDy(node0, recordMap))) {
      // Record output nodes anchor vs succeed node anchor map
      RecordOutputAnchorMap(node0);
      RecordOutputAnchorMap(node1);

      // when found all node in pattern, do graph fusion
      vector<ge::NodePtr> fusionNodes;
      Status status = DoFusion(graph, recordMap, fusionNodes);
      if (status == SUCCESS) {
        SetExtemDataDumpAttr(recordMap, fusionNodes);
      }
      return status;
    } else {
      return NOT_CHANGED;
    }
    // Step3: else, nodeEqual should have one output node, check that it should
    // be select_dx or select_dy
  } else if (equalOutAnchor->GetPeerInDataAnchors().size() == EQUAL_WITH_ONE_OUTPUT_NODE) {
    auto peerAnchor0 = equalOutAnchor->GetPeerInDataAnchors().at(0);
    if (peerAnchor0 == nullptr) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[peerAnchor0] must not be null.");
      return fe::PARAM_INVALID;
    }
    ge::NodePtr node0 = peerAnchor0->GetOwnerNode();
    if (node0 == nullptr) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[node0] must not be null.");
      return fe::PARAM_INVALID;
    }

    if (MatchDx(node0, recordMap) || MatchDy(node0, recordMap)) {
      // Record output nodes anchor vs succeed node anchor map
      RecordOutputAnchorMap(node0);

      // when found all node in pattern, do graph fusion
      vector<ge::NodePtr> fusionNodes;
      Status status = DoFusion(graph, recordMap, fusionNodes);
      if (status == SUCCESS) {
        SetExtemDataDumpAttr(recordMap, fusionNodes);
      }
      return status;
    } else {
      return NOT_CHANGED;
    }
  }

  return NOT_CHANGED;
}
void ExtremumGradFusionPass::SetExtemDataDumpAttr(const std::map<string, ge::NodePtr>& recordMap,
                                                  vector<ge::NodePtr>& fusionNodes) {
  std::vector<ge::NodePtr> originalNodes;
  for (auto iter = recordMap.begin(); iter != recordMap.end(); iter++) {
    if (iter->second != nullptr) {
      originalNodes.push_back(iter->second);
    }
  }
  SetDataDumpAttr(originalNodes, fusionNodes);
}
ge::NodePtr ExtremumGradFusionPass::FindNodeInRecordMap(const map<string, ge::NodePtr>& recordMap, string key) {
  ge::NodePtr node = nullptr;
  if (recordMap.find(key) != recordMap.end()) {
    node = recordMap.at(key);
  }

  return node;
}

bool ExtremumGradFusionPass::CheckAttrMatch(const map<string, ge::NodePtr>& recordMap) {
  // attr _stream_label of fusion nodes must be equal
  string streamLabel = "";
  for (auto iter = recordMap.begin(); iter != recordMap.end(); iter++) {
    string streamLabelTmp = "";
    if (!ge::AttrUtils::GetStr(iter->second->GetOpDesc(), STREAM_LABEL, streamLabelTmp)) {
      streamLabelTmp = "null";
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Fusion nodes not have _stream_label attr.");
    }
    if (streamLabel == "") {
      streamLabel = streamLabelTmp;
    } else if (streamLabel != "" && streamLabel != streamLabelTmp) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "_stream_label not equal, pattern matching failed.");
      return false;
    }
  }
  return true;
}

Status ExtremumGradFusionPass::DoFusion(ge::ComputeGraph& graph, const map<string, ge::NodePtr>& recordMap,
                                        vector<ge::NodePtr>& fusionNodes) {
  // Step1: attr _stream_label of fusion nodes must be equal
  if (!CheckAttrMatch(recordMap)) {
    return NOT_CHANGED;
  }
  // Step2: Create MaximumGrad/MinimumGrad OpDesc & Node
  ge::NodePtr nodeEqual = FindNodeInRecordMap(recordMap, PATTERN_EQUAL);
  if (nodeEqual == nullptr) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[nodeEqual] must not be null.");
    return fe::PARAM_INVALID;
  }
  ge::NodePtr selectDxNode = FindNodeInRecordMap(recordMap, PATTERN_SELECT_DX);

  ge::NodePtr outputDxNode = selectDxNode;
  if (recordMap.find(PATTERN_SUM_DX) != recordMap.end()) {
    outputDxNode = recordMap.at(PATTERN_SUM_DX);
  }

  ge::NodePtr selectDyNode = FindNodeInRecordMap(recordMap, PATTERN_SELECT_DY);
  ge::NodePtr outputDyNode = selectDyNode;
  if (recordMap.find(PATTERN_SUM_DY) != recordMap.end()) {
    outputDyNode = recordMap.at(PATTERN_SUM_DY);
  }
  // Step3: Get dz output anchor to select node
  ge::NodePtr dzInputNode = FindNodeInRecordMap(recordMap, PATTERN_DZ);
  if (dzInputNode == nullptr) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[dzInputNode] must not be null.");
    return fe::PARAM_INVALID;
  }
  ge::OutDataAnchorPtr dzInputAnchor = nullptr;
  for (auto &anchor : dzInputNode->GetAllOutDataAnchors()) {
    if (anchor == nullptr) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[anchor] must not be null.");
      return fe::PARAM_INVALID;
    }
    for (auto &peerAnchor : anchor->GetPeerInDataAnchors()) {
      if (peerAnchor == nullptr) {
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[peerAnchor] must not be null.");
        return fe::PARAM_INVALID;
      }
      if (peerAnchor->GetOwnerNode() == selectDxNode || peerAnchor->GetOwnerNode() == selectDyNode) {
        dzInputAnchor = anchor;
        break;
      }
    }
    if (dzInputAnchor != nullptr) {
      break;
    }
  }

  ge::NodePtr extreGradNode = nullptr;
  extreGradNode = CreateExtremumGradNode(graph, nodeEqual, selectDxNode, selectDyNode, recordMap);
  if (extreGradNode == nullptr) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[extreGradNode] must not be null.");
    return fe::PARAM_INVALID;
  }
  // Step4: Relink outerX, outerY, outerDz to MaximumGrad/MinimumGrad intput0,
  // input1, input2,
  Status adjustAnchorRes = AdjustAnchor(dzInputAnchor, nodeEqual, extreGradNode, outputDxNode, outputDyNode);
  FUSION_PASS_CHECK(adjustAnchorRes != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Replace ExtremumGrad node to ComputeGraph failed."),
                    return adjustAnchorRes);

  vector<bool> isInputConst;
  for (auto &anchor : extreGradNode->GetAllInDataAnchors()) {
    auto peerAnchor = anchor->GetPeerOutAnchor();
    auto node = peerAnchor->GetOwnerNode();
    auto outputTensor = node->GetOpDesc()->GetOutputDesc(peerAnchor->GetIdx());
    FUSION_PASS_CHECK(extreGradNode->GetOpDesc()->UpdateInputDesc(anchor->GetIdx(), outputTensor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "update input failed."), return FAILED);

    if (ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(node) == CONSTANT) {
      isInputConst.push_back(true);
    } else {
      isInputConst.push_back(false);
    }
  }
  extreGradNode->GetOpDesc()->SetIsInputConst(isInputConst);
  // Step5: Remove origin node from graph
  Status removeEqual = RemoveNode(graph, recordMap, PATTERN_EQUAL);
  if (removeEqual != SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove Node GreaterEqual/LessEqual from graph failed.");
    return removeEqual;
  }

  Status removeSelectDx = RemoveNode(graph, recordMap, PATTERN_SELECT_DX);
  if (removeSelectDx != SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove select node dx failed.");
    return removeSelectDx;
  }

  Status removeSelectDy = RemoveNode(graph, recordMap, PATTERN_SELECT_DY);
  if (removeSelectDy != SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove select node dy failed.");
    return removeSelectDy;
  }

  Status removeSumDx = RemoveNode(graph, recordMap, PATTERN_SUM_DX);
  if (removeSumDx != SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove sum node dx failed.");
    return removeSumDx;
  }

  Status removeSumDy = RemoveNode(graph, recordMap, PATTERN_SUM_DY);
  if (removeSumDy != SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove sum node dy failed.");
    return removeSumDy;
  }
  fusionNodes.push_back(extreGradNode);
  return SUCCESS;
}

Status ExtremumGradFusionPass::RemoveInputEdges(ge::ComputeGraph& graph, const ge::NodePtr node) const {
  if (node == nullptr) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[node] must not be null.");
    return fe::PARAM_INVALID;
  }
  for (ge::InDataAnchorPtr &anchor : node->GetAllInDataAnchors()) {
    if (anchor == nullptr) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[anchor] must not be null.");
      return fe::PARAM_INVALID;
    }
    ge::OutDataAnchorPtr srcAnchor = anchor->GetPeerOutAnchor();
    if (srcAnchor == nullptr) {
      continue;
    }

    if (ge::GraphUtils::RemoveEdge(srcAnchor, anchor) != ge::GRAPH_SUCCESS) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Disconnect %s input data anchor failed.", node->GetName().c_str());
      return FAILED;
    }

    ge::NodePtr srcNode = srcAnchor->GetOwnerNode();
    if (srcNode == nullptr) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[srcNode] must not be null.");
      return fe::PARAM_INVALID;
    }
    if (srcNode->GetInAllNodes().size() == 0 && srcNode->GetOutAllNodes().size() == 0) {
      // Delete isolated const node from graph
      if (graph.RemoveNode(srcNode) != ge::GRAPH_SUCCESS) {
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove node from graph failed.");
        return FAILED;
      }
    }
  }

  ge::InControlAnchorPtr inCtrlAnchor = node->GetInControlAnchor();
  if (inCtrlAnchor != nullptr) {
    for (ge::OutControlAnchorPtr &srcAnchor : inCtrlAnchor->GetPeerOutControlAnchors()) {
      if (ge::GraphUtils::RemoveEdge(srcAnchor, inCtrlAnchor) != ge::GRAPH_SUCCESS) {
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Disconnect %s input control anchor failed.", node->GetName().c_str());
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

Status ExtremumGradFusionPass::RemoveOutputEdges(ge::NodePtr node) const {
  if (node == nullptr) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[node] must not be null.");
    return fe::PARAM_INVALID;
  }
  for (ge::OutDataAnchorPtr &anchor : node->GetAllOutDataAnchors()) {
    if (anchor == nullptr) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[anchor] must not be null.");
      return fe::PARAM_INVALID;
    }
    for (ge::InDataAnchorPtr &dstAnchor : anchor->GetPeerInDataAnchors()) {
      if (ge::GraphUtils::RemoveEdge(anchor, dstAnchor) != ge::GRAPH_SUCCESS) {
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Disconnect %s output data anchor failed.", node->GetName().c_str());
        return FAILED;
      }
    }
  }

  ge::OutControlAnchorPtr outControlAnchor = node->GetOutControlAnchor();
  if (outControlAnchor != nullptr) {
    for (ge::InControlAnchorPtr &dstAnchor : outControlAnchor->GetPeerInControlAnchors()) {
      if (ge::GraphUtils::RemoveEdge(outControlAnchor, dstAnchor) != ge::GRAPH_SUCCESS) {
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Disconnect %s output control anchor failed.", node->GetName().c_str());
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

Status ExtremumGradFusionPass::RemoveNode(ge::ComputeGraph& graph, const map<string, ge::NodePtr>& recordMap,
                                          string patternName) {
  auto iter = recordMap.find(patternName);
  if (iter == recordMap.end() || iter->second == nullptr) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "pattern name %s not in map", patternName.c_str());
    return SUCCESS;
  }

  ge::NodePtr node = iter->second;
  // Step1: Disconnect all input anchor
  Status removeInputEdges = RemoveInputEdges(graph, node);
  if (removeInputEdges != SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove node:%s input edges failed.", node->GetName().c_str());
    return removeInputEdges;
  }

  // Step2: Disconnect all output anchor
  Status removeOutputEdges = RemoveOutputEdges(node);
  if (removeOutputEdges != SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove node:%s output edges failed.", node->GetName().c_str());
    return removeOutputEdges;
  }

  // Step3: Remove node from graph
  if (graph.RemoveNode(node) != ge::GRAPH_SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove node from graph failed.");
    return FAILED;
  }
  return SUCCESS;
}

ge::NodePtr ExtremumGradFusionPass::CreateExtremumGradNode(ge::ComputeGraph& graph, ge::NodePtr nodeEqual,
                                                           ge::NodePtr selectDxNode, ge::NodePtr selectDyNode,
                                                           const map<string, ge::NodePtr>& recordMap) {
  if (nodeEqual == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "equal node is null");
    return nullptr;
  }

  // Step1: Get Opdesc from input node
  ge::OpDescPtr equalOpDesc = nodeEqual->GetOpDesc();
  if (equalOpDesc == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "equal opdesc is null");
    return nullptr;
  }

  ge::OpDescPtr selectOpDesc = nullptr;
  if (selectDxNode != nullptr) {
    selectOpDesc = selectDxNode->GetOpDesc();
  } else {
    selectOpDesc = selectDyNode->GetOpDesc();
  }
  if (selectOpDesc == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "select opdesc is null");
    return nullptr;
  }

  // Step2: Create extremum grad OpDesc, set OpType & name according nodeEqual
  //    Init "grad_x", "grad_y" = false, extract "T" from selectNode
  ge::OpDescPtr extreGradOpDesc = make_shared<ge::OpDesc>();
  if (extreGradOpDesc == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "extreGradOpDesc is nullptr, create not success!");
    return nullptr;
  }

  if (SetExtreMumGradOpDesc(equalOpDesc, selectOpDesc, extreGradOpDesc) != SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Set OpType, OpName, Attrs to ExtremumGradNode not success.");
    return nullptr;
  }

  // Step3: Set 3 input tensorDesc to extreGradOpDesc
  ge::GeTensorDesc inputTensorDesc1 = equalOpDesc->GetInputDesc(0);
  ge::GeTensorDesc inputTensorDesc2 = equalOpDesc->GetInputDesc(1);
  auto iterDz = recordMap.find(PATTERN_DZ);
  if (iterDz == recordMap.end()) {
    return nullptr;
  }
  ge::GeTensorDesc inputTensorDesc3 = iterDz->second->GetOpDesc()->GetOutputDesc(0);
  if (extreGradOpDesc->AddInputDesc(inputTensorDesc1) != ge::GRAPH_SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input desc[0] to %s not success", extreGradOpDesc->GetName().c_str());
    return nullptr;
  }
  if (extreGradOpDesc->AddInputDesc(inputTensorDesc2) != ge::GRAPH_SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input desc[1] to %s not success", extreGradOpDesc->GetName().c_str());
    return nullptr;
  }

  if (extreGradOpDesc->AddInputDesc(inputTensorDesc3) != ge::GRAPH_SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input desc[2] to %s not success", extreGradOpDesc->GetName().c_str());
    return nullptr;
  }

  // Step4: Set output tensorDesc according mapping result
  if (recordMap.find(PATTERN_SELECT_DX) != recordMap.end()) {
    if (recordMap.find(PATTERN_SUM_DX) != recordMap.end()) {
      ge::NodePtr sumNode = recordMap.at(PATTERN_SUM_DX);
      ge::GeTensorDesc sumOutputTensorDesc = sumNode->GetOpDesc()->GetOutputDesc(0);
      if (extreGradOpDesc->AddOutputDesc(sumOutputTensorDesc) != ge::GRAPH_SUCCESS) {
        OP_LOGW(FUSED_OP_TYPE.c_str(), "add output desc with sumDx output not success");
        return nullptr;
      }
    } else {
      ge::GeTensorDesc dxTensorDesc = selectDxNode->GetOpDesc()->GetOutputDesc(0);
      if (extreGradOpDesc->AddOutputDesc(dxTensorDesc) != ge::GRAPH_SUCCESS) {
        OP_LOGW(FUSED_OP_TYPE.c_str(), "add output desc with selectDx output not success");
        return nullptr;
      }
    }
  }
  if (recordMap.find(PATTERN_SELECT_DY) != recordMap.end()) {
    if (recordMap.find(PATTERN_SUM_DY) != recordMap.end()) {
      ge::NodePtr sumNode = recordMap.at(PATTERN_SUM_DY);
      ge::GeTensorDesc sumOutputTensorDesc = sumNode->GetOpDesc()->GetOutputDesc(0);
      if (extreGradOpDesc->AddOutputDesc(sumOutputTensorDesc) != ge::GRAPH_SUCCESS) {
        OP_LOGW(FUSED_OP_TYPE.c_str(), "add output desc with sumDy output not success");
        return nullptr;
      }
    } else {
      ge::GeTensorDesc dxTensorDesc = selectDyNode->GetOpDesc()->GetOutputDesc(0);
      if (extreGradOpDesc->AddOutputDesc(dxTensorDesc) != ge::GRAPH_SUCCESS) {
        OP_LOGW(FUSED_OP_TYPE.c_str(), "add output desc with selectDy output not success");
        return nullptr;
      }
    }
  }

  if (recordMap.find(PATTERN_SELECT_DX) == recordMap.end() || recordMap.find(PATTERN_SELECT_DY) == recordMap.end()) {
    ge::GeTensorDesc tensorDesc;
    if (extreGradOpDesc->AddOutputDesc(tensorDesc) != ge::GRAPH_SUCCESS) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "add output desc with null tensor desc not success");
      return nullptr;
    }
  }
  string streamLabel = "";
  if (!ge::AttrUtils::GetStr(nodeEqual->GetOpDesc(), STREAM_LABEL, streamLabel)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Fusion nodes not have _stream_label attr.");
  } else {
    if (!ge::AttrUtils::SetStr(extreGradOpDesc, STREAM_LABEL, streamLabel)) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "newNode set _stream_label error, fusion failed.");
      return nullptr;
    }
  }
  // Step5: Create node
  ge::NodePtr extremumNode = graph.AddNode(extreGradOpDesc);

  return extremumNode;
}

Status ExtremumGradFusionPass::SetExtreMumGradOpDesc(ge::OpDescPtr equalOpDesc, ge::OpDescPtr selectOpDesc,
                                                     ge::OpDescPtr extreGradOpDesc) const {
  if (equalOpDesc == nullptr) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[equalOpDesc] must not be null.");
    return fe::PARAM_INVALID;
  }
  if (selectOpDesc == nullptr) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[selectOpDesc] must not be null.");
    return fe::PARAM_INVALID;
  }
  if (extreGradOpDesc == nullptr) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[extreGradOpDesc] must not be null.");
    return fe::PARAM_INVALID;
  }

  // Step1: Set ExtremumGrad OpDesc basic attr
  if (equalOpDesc->GetType() == GREATER_EQUAL) {
    extreGradOpDesc->SetType(MAXIMUM_GRAD);
  } else {
    extreGradOpDesc->SetType(MINIMUM_GRAD);
  }

  // Step2: Set same name scope with select: "**/**/**/**" ->
  // "**/**/**/ExtremumGrad"
  string nameTmp = selectOpDesc->GetName();
  vector<string> namesTmp = ge::StringUtils::Split(nameTmp, '/');
  string extreGradName = "";
  for (size_t i = 0; i < (namesTmp.size() - 1); ++i) {
    extreGradName += (namesTmp[i] + "/");
  }
  extreGradName += extreGradOpDesc->GetType();
  static int extremumGradCount = 0;
  extreGradOpDesc->SetName(extreGradName + to_string(extremumGradCount));
  OP_LOGD(FUSED_OP_TYPE.c_str(), "add %s Op:%s in ExtremumGradFusionPass", extreGradOpDesc->GetType().c_str(),
          extreGradOpDesc->GetName().c_str());
  ++extremumGradCount;

  // Step3: Set ExtremumGrad OpDesc Attrs
  // Extract attr "T" from select node
  if (ge::AttrUtils::HasAttr(selectOpDesc, ATTR_DATA_TYPE)) {
    ge::DataType selectDataType;
    if (!ge::AttrUtils::GetDataType(selectOpDesc, ATTR_DATA_TYPE, selectDataType)) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get datatype from select node failed");
      return FAILED;
    }

    if (!ge::AttrUtils::SetDataType(extreGradOpDesc, ATTR_DATA_TYPE, selectDataType)) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "set datatype to extremum_grad node failed");
      return FAILED;
    }
  } else {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Cannot find attr:T in node:%s", selectOpDesc->GetName().c_str());
    return FAILED;
  }
  // Init "grad_x", "grad_y" attr to ExtremumGrad OP
  if (!ge::AttrUtils::SetBool(extreGradOpDesc, ATTR_GRAD_X, false)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "set Grad_X to extremum_grad node failed");
    return FAILED;
  }

  if (!ge::AttrUtils::SetBool(extreGradOpDesc, ATTR_GRAD_Y, false)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "set Grad_Y to extremum_grad node failed");
    return FAILED;
  }
  return SUCCESS;
}

Status ExtremumGradFusionPass::AdjustAnchor(ge::OutDataAnchorPtr dzInputAnchor, ge::NodePtr nodeEqual,
                                            ge::NodePtr extreGradNode, ge::NodePtr outputDxNode,
                                            ge::NodePtr outputDyNode) const {
  if (nodeEqual == nullptr) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[nodeEqual] must not be null.");
    return fe::PARAM_INVALID;
  }
  if (extreGradNode == nullptr) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[extreGradNode] must not be null.");
    return fe::PARAM_INVALID;
  }
  // Step1: Add link from Dz output to MaximumGrad/MinimumGrad input[0]
  auto extreNodeInput0 = extreGradNode->GetInDataAnchor(0);
  if (ge::GraphUtils::AddEdge(dzInputAnchor, extreNodeInput0) != ge::GRAPH_SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between DzInput to ExtremumGrad input0 failed.");
    return FAILED;
  }
  // Step2: Adjust Equal input[0,1] peer anchor to extreGradNode input[1,2]
  int inAnchorIndex = EQUAL_TO_EXTREMUM_START;
  for (auto &inputAnchor : nodeEqual->GetAllInDataAnchors()) {
    if (inputAnchor == nullptr) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[inputAnchor] must not be null.");
      return fe::PARAM_INVALID;
    }
    auto srcAnchor = inputAnchor->GetPeerOutAnchor();
    if (srcAnchor == nullptr) {
      continue;
    }
    auto dstAnchor = extreGradNode->GetInDataAnchor(inAnchorIndex);
    Status replaceEdgeDstRes = ReplaceEdgeDst(srcAnchor, inputAnchor, dstAnchor);
    if (replaceEdgeDstRes != SUCCESS) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                     "Replace equal_inputAnchors[%d] to extreGradNode_inputAnchors[%d] failed",
                                     inAnchorIndex - EQUAL_TO_EXTREMUM_START, inAnchorIndex);
      return replaceEdgeDstRes;
    }

    ++inAnchorIndex;
  }
  // Step3: Adjust ExtreGrad OutAnchor0 to outputDxNode[0] peer anchor
  int outputIndex = 0;
  if (outputDxNode != nullptr) {
    ge::OutDataAnchorPtr extreGradOutAnchor0 = extreGradNode->GetOutDataAnchor(outputIndex);
    ge::OutDataAnchorPtr oldSrcAnchor = outputDxNode->GetOutDataAnchor(0);
    if (oldSrcAnchor == nullptr) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[oldSrcAnchor] must not be null.");
      return fe::PARAM_INVALID;
    }
    for (ge::InDataAnchorPtr &dstAnchor : oldSrcAnchor->GetPeerInDataAnchors()) {
      Status replaceEdgeSrcRes = ReplaceEdgeSrc(oldSrcAnchor, extreGradOutAnchor0, dstAnchor);
      if (replaceEdgeSrcRes != SUCCESS) {
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Replace edge failed.");
        return replaceEdgeSrcRes;
      }
    }

    if (ge::AttrUtils::SetBool(extreGradNode->GetOpDesc(), ATTR_GRAD_X, true) == false) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "set Grad_X = true to extremum_grad node failed");
      return FAILED;
    }
    outputIndex++;
  }
  // Step4: Adjust ExtreGrad OutAnchor1 to outputDyNode[0] peer anchor
  if (outputDyNode != nullptr) {
    auto extreGradOutAnchor1 = extreGradNode->GetOutDataAnchor(outputIndex);
    auto oldSrcAnchor = outputDyNode->GetOutDataAnchor(0);
    if (oldSrcAnchor == nullptr) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[oldSrcAnchor] must not be null.");
      return fe::PARAM_INVALID;
    }

    for (auto &dstAnchor : oldSrcAnchor->GetPeerInDataAnchors()) {
      Status replaceEdgeSrcRes = ReplaceEdgeSrc(oldSrcAnchor, extreGradOutAnchor1, dstAnchor);
      if (replaceEdgeSrcRes != SUCCESS) {
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Replace edge failed.");
        return replaceEdgeSrcRes;
      }
    }

    if (ge::AttrUtils::SetBool(extreGradNode->GetOpDesc(), ATTR_GRAD_Y, true) == false) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "set Grad_Y = true to extremum_grad node failed");
      return FAILED;
    }
  }

  return SUCCESS;
}

Status ExtremumGradFusionPass::ReplaceEdgeDst(ge::OutDataAnchorPtr src, ge::InDataAnchorPtr dst,
                                              ge::InDataAnchorPtr newDst) const {
  if (ge::GraphUtils::RemoveEdge(src, dst) != ge::GRAPH_SUCCESS ||
      ge::GraphUtils::AddEdge(src, newDst) != ge::GRAPH_SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Replace edge dst Failed.");
    return FAILED;
  }
  return SUCCESS;
}

Status ExtremumGradFusionPass::ReplaceEdgeSrc(ge::OutDataAnchorPtr src, ge::OutDataAnchorPtr newSrc,
                                              ge::InDataAnchorPtr dst) const {
  if (ge::GraphUtils::RemoveEdge(src, dst) != ge::GRAPH_SUCCESS ||
      ge::GraphUtils::AddEdge(newSrc, dst) != ge::GRAPH_SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Replace edge src Failed.");
    return FAILED;
  }
  return SUCCESS;
}
REGISTER_PASS("ExtremumGradFusionPass", BUILT_IN_GRAPH_PASS, ExtremumGradFusionPass);
}  // namespace fe
