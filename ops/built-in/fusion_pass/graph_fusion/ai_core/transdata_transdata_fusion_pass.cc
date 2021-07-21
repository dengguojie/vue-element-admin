/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file transdata_transdata_fusion_pass.cpp
 * \brief
 */
#include "transdata_transdata_fusion_pass.h"

#include <iostream>
#include <vector>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"

using namespace std;
using namespace ge;

namespace fe {
static const string PATTERN_REFORMAT = "ReFormat";
static const string PATTERN_TRANSDATA_1 = "Transdata_1";
static const string PATTERN_TRANSDATA_2 = "Transdata_2";
static const string PATTERN_RESHAPE = "Reshape";
static const string PATTERN_FULLYCONNECTION = "FullyConnection";

/*
 transdata --> transdata --> FullyConnection
*/

Status TransdataTransdataPass::RemoveNode(ge::NodePtr node, ge::ComputeGraph& graph) {
  // remove input data edge
  for (size_t i = 0; i < node->GetAllInDataAnchors().size(); ++i) {
    auto inDataAnchor = node->GetInDataAnchor(i);
    FUSION_PASS_CHECK(inDataAnchor == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inDataAnchor is null, remove node failed."), return FAILED);
    auto preOutDataAnchor = inDataAnchor->GetPeerOutAnchor();
    FUSION_PASS_CHECK(preOutDataAnchor == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "preOutDataAnchor is null, remove node failed."), return FAILED);

    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(preOutDataAnchor, inDataAnchor) != ge::GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove node failed."), return FAILED);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "remove edge %u of node %s", i, node->GetName().c_str());
  }
  // delete the node
  FUSION_PASS_CHECK(graph.RemoveNode(node) != ge::GRAPH_SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove node failed"),
                    return FAILED);
  return SUCCESS;
}

vector<FusionPattern*> TransdataTransdataPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define TransdataTransdataPass pattern begin");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern1 = new (std::nothrow) FusionPattern("TransdataTransdataPass1");
  FUSION_PASS_CHECK(pattern1 == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new an object failed"), return patterns);

  pattern1->AddOpDesc(PATTERN_TRANSDATA_1, {"TransData"})
      .AddOpDesc(PATTERN_TRANSDATA_2, {"TransData"})
      .AddOpDesc(PATTERN_FULLYCONNECTION, {"FullyConnection"})
      .SetInputs(PATTERN_TRANSDATA_2, {PATTERN_TRANSDATA_1})
      .SetInputs(PATTERN_FULLYCONNECTION, {PATTERN_TRANSDATA_2})
      .SetOutput(PATTERN_FULLYCONNECTION);
  patterns.push_back(pattern1);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define TransdataTransdataPass1 pattern end");

  FusionPattern* pattern2 = new (std::nothrow) FusionPattern("TransdataTransdataPass2");
  FUSION_PASS_CHECK(pattern2 == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new an object failed"), return patterns);

  pattern2->AddOpDesc(PATTERN_TRANSDATA_1, {"TransData"})
      .AddOpDesc(PATTERN_TRANSDATA_2, {"TransData"})
      .AddOpDesc(PATTERN_RESHAPE, {"Reshape"})
      .AddOpDesc(PATTERN_REFORMAT, {"ReFormat"})
      .AddOpDesc(PATTERN_FULLYCONNECTION, {"FullyConnection", "FullyConnectionCompress"})
      .SetInputs(PATTERN_RESHAPE, {PATTERN_TRANSDATA_1})
      .SetInputs(PATTERN_REFORMAT, {PATTERN_RESHAPE})
      .SetInputs(PATTERN_TRANSDATA_2, {PATTERN_REFORMAT})
      .SetInputs(PATTERN_FULLYCONNECTION, {PATTERN_TRANSDATA_2})
      .SetOutput(PATTERN_FULLYCONNECTION);
  patterns.push_back(pattern2);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define TransdataTransdataPass2 pattern end");

  return patterns;
}

Status TransdataTransdataPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define TransdataTransdataPass fusion begin");
  ge::NodePtr transData_1 = GetNodeFromMapping(PATTERN_TRANSDATA_1, mapping);
  ge::NodePtr transData_2 = GetNodeFromMapping(PATTERN_TRANSDATA_2, mapping);
  ge::NodePtr reshapeOP = GetNodeFromMapping(PATTERN_RESHAPE, mapping);
  ge::NodePtr reformatOP = GetNodeFromMapping(PATTERN_REFORMAT, mapping);
  ge::NodePtr fullyConnection = GetNodeFromMapping(PATTERN_FULLYCONNECTION, mapping);

  FUSION_PASS_CHECK(transData_1 == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "transData_1 is null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(transData_2 == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "transData_2 is null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(fullyConnection == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fullyConnection is null"),
                    return PARAM_INVALID);

  if ((reshapeOP != nullptr) && (reshapeOP->GetOutDataNodes().size() > 1)){
    return SUCCESS;
  }
  int64_t axis;
  FUSION_PASS_CHECK(
      !ge::AttrUtils::GetInt(fullyConnection->GetOpDesc(), "axis", axis),
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Get node[%s]'s axis attr not success.", fullyConnection->GetName().c_str()),
      return false);
  if (axis != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "node[FullyConnection]'s axis is not 1, not support fusion, TransdataTransdataPass fusion end");
    return SUCCESS;
  }
  ge::GeTensorDesc secondTransDataOutputTensor = transData_2->GetOpDesc()->GetOutputDesc(0);
  if (static_cast<ge::Format>(ge::GetPrimaryFormat(secondTransDataOutputTensor.GetFormat())) != ge::FORMAT_FRACTAL_Z &&
      static_cast<ge::Format>(ge::GetPrimaryFormat(secondTransDataOutputTensor.GetFormat())) != ge::FORMAT_FRACTAL_NZ) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "node[FullyConnection]'s input format is not FRACTAL_Z , not support fusion, TransdataTransdataPass fusion "
            "end");
    return SUCCESS;
  }
  ge::OpDescPtr firstTransDataOpDesc = transData_1->GetOpDesc();
  FUSION_PASS_CHECK(firstTransDataOpDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "transData_1 opdesc is null"),
                    return PARAM_INVALID);
  ge::GeTensorDesc firstTransDataInputTensor = firstTransDataOpDesc->GetInputDesc(0);
  if (firstTransDataInputTensor.GetFormat() != ge::FORMAT_NC1HWC0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "firstTransDataInputTensor format is not NC1HWC0 , not support fusion, TransdataTransdataPass fusion end");
    return SUCCESS;
  }
  vector<int64_t> shapeDims = secondTransDataOutputTensor.GetOriginShape().GetDims();
  if (shapeDims.empty()) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "fullyConnection first input(x) shape is NULL");
    return FAILED;
  }

  transData_1->GetOpDesc()->UpdateOutputDesc(0, secondTransDataOutputTensor);
  ge::AttrUtils::SetStr(transData_1->GetOpDesc(), "dst_format", "FRACTAL_Z");

  auto firstTransDataOutDataAnchor = transData_1->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(firstTransDataOutDataAnchor == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "firstTransDataOutDataAnchor is null"), return FAILED);

  auto secondTransDataOutDataAnchor = transData_2->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(secondTransDataOutDataAnchor == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "secondTransDataOutDataAnchor is null"), return FAILED);
  auto secondTransDataPeerInDataAnchor = secondTransDataOutDataAnchor->GetPeerInDataAnchors();

  // delete transData_2 node
  if (reshapeOP != nullptr)
    FUSION_PASS_CHECK(graph.RemoveNode(reshapeOP) != GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove reshape node failed"), return FAILED);
  if (reformatOP != nullptr)
    FUSION_PASS_CHECK(graph.RemoveNode(reformatOP) != GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove reformat node failed"), return FAILED);
  FUSION_PASS_CHECK(RemoveNode(transData_2, graph) == FAILED,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove transData_2 node failed"), return FAILED);

  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(firstTransDataOutDataAnchor, secondTransDataPeerInDataAnchor.at(0)) != ge::GRAPH_SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add TransData_1 and  fullyConnection edge error"), return FAILED);

  fusionNodes.push_back(transData_1);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define TransdataTransdataPass fusion end");
  return SUCCESS;
}

REGISTER_PASS("TransdataTransdataPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, TransdataTransdataPass);
}  // namespace fe
