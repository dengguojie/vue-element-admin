/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
 * \file transdata_confusion_transpose.cpp
 * \brief
 */
#include "transdata_confusion_transpose_fusion_pass.h"

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
static const string PATTERN_CONFUSIONTRANSPOSE = "ConfusionTransposeD";
static const string PATTERN_TRANSDATA_1 = "Transdata_1";
static const string PATTERN_REFORMAT = "ReFormat";
static const string PATTERN_TRANSDATA_2 = "Transdata_2";

/*
 transdata1 --> confusion_transpose_d --> transdata2
*/

Status TransDataConfusionTransposeFusionPass::RemoveNode(ge::NodePtr node, ge::ComputeGraph& graph) {
  // remove input data edge
  for (size_t i = 0; i < node->GetAllInDataAnchors().size(); ++i) {
    auto inDataAnchor = node->GetInDataAnchor(i);
    FUSION_PASS_CHECK(inDataAnchor == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "inDataAnchor is null, remove node failed."),
                      return FAILED);
    auto preOutDataAnchor = inDataAnchor->GetPeerOutAnchor();
    FUSION_PASS_CHECK(preOutDataAnchor == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "preOutDataAnchor is null, remove node failed."),
                      return FAILED);

    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(preOutDataAnchor, inDataAnchor) != ge::GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove node failed."),
                      return FAILED);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "remove edge %u of node %s", i, node->GetName().c_str());
  }
  // delete the node
  FUSION_PASS_CHECK(graph.RemoveNode(node) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove node failed"),
                    return FAILED);
  return SUCCESS;
}

vector<FusionPattern*> TransDataConfusionTransposeFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define TransDataConfusionTransposeFusionPass pattern begin");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("TransDataConfusionTransposeFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new an object failed"),
                    return patterns);

  pattern->AddOpDesc(PATTERN_TRANSDATA_1, {"TransData"})
      .AddOpDesc(PATTERN_REFORMAT, {"ReFormat"})
      .AddOpDesc(PATTERN_CONFUSIONTRANSPOSE, {"ConfusionTransposeD"})
      .AddOpDesc(PATTERN_TRANSDATA_2, {"TransData"})
      .SetInputs(PATTERN_REFORMAT, {PATTERN_TRANSDATA_1})
      .SetInputs(PATTERN_CONFUSIONTRANSPOSE, {PATTERN_REFORMAT})
      .SetInputs(PATTERN_TRANSDATA_2, {PATTERN_CONFUSIONTRANSPOSE})
      .SetOutput(PATTERN_TRANSDATA_2);
  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define TransDataConfusionTransposeFusionPass pattern end");

  return patterns;
}

Status TransDataConfusionTransposeFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                     vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define TransDataConfusionTransposeFusionPass fusion begin");
  ge::NodePtr transData_1 = GetNodeFromMapping(PATTERN_TRANSDATA_1, mapping);
  ge::NodePtr transData_2 = GetNodeFromMapping(PATTERN_TRANSDATA_2, mapping);
  ge::NodePtr confusionTransposeD = GetNodeFromMapping(PATTERN_CONFUSIONTRANSPOSE, mapping);

  FUSION_PASS_CHECK(transData_1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "transData_1 is null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(transData_2 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "transData_2 is null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(confusionTransposeD == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "confusionTransposeD is null"),
                    return PARAM_INVALID);
  // must be NZ to ND
  ge::OpDescPtr firstTransDataOpDesc = transData_1->GetOpDesc();
  FUSION_PASS_CHECK(firstTransDataOpDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "transData_1 opdesc is null"),
                    return PARAM_INVALID);
  ge::GeTensorDesc firstTransDataInputTensor = firstTransDataOpDesc->GetInputDesc(0);
  ge::GeTensorDesc firstTransDataOutputTensor = firstTransDataOpDesc->GetOutputDesc(0);
  ge::GeShape transDataShape_1 = firstTransDataInputTensor.GetOriginShape();
  if (!(firstTransDataOutputTensor.GetFormat() == ge::FORMAT_ND &&
      ge::GetPrimaryFormat(firstTransDataInputTensor.GetFormat()) == ge::FORMAT_FRACTAL_NZ)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "node[TransData]'s input format is not FRACTAL_NZ , not support fusion,"
            "TransDataConfusionTransposeFusionPass fusion end");
    return NOT_CHANGED;
  }
  // must be ND to NZ
  ge::OpDescPtr secondTransDataOpDesc = transData_2->GetOpDesc();
  FUSION_PASS_CHECK(secondTransDataOpDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "transData_2 opdesc is null"),
                    return PARAM_INVALID);
  ge::GeTensorDesc secondTransDataInputTensor = secondTransDataOpDesc->GetInputDesc(0);
  ge::GeTensorDesc secondTransDataOutputTensor = secondTransDataOpDesc->GetOutputDesc(0);
  ge::GeShape transDataShape_2 = secondTransDataOutputTensor.GetOriginShape();
  if (!(secondTransDataInputTensor.GetFormat() == ge::FORMAT_ND &&
      ge::GetPrimaryFormat(secondTransDataOutputTensor.GetFormat()) == ge::FORMAT_FRACTAL_NZ)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "node[TransData]'s input format is not FRACTAL_NZ,"
            "not support fusion, TransDataConfusionTransposeFusionPass fusion end");
    return NOT_CHANGED;
  }
  // perm must be [0, 2, 1, 3]
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(confusionTransposeD);
  // get perm of confusion_transpose_d
  std::vector<int64_t> permList;
  if (op.GetAttr("perm", permList) != ge::GRAPH_SUCCESS) {
    OP_LOGI(TbeGetName(op).c_str(), "GetOpAttr perm failed!");
    return NOT_CHANGED;
  }
  if (permList.size() != 4 || permList[0] != 0 || permList[1] != 2 || permList[2] != 1 || permList[3] != 3) {
    OP_LOGI(TbeGetName(op).c_str(),
            "length of perm not equal to 4!");
    return NOT_CHANGED;
  }
  size_t firstShapeSize = transDataShape_1.GetDimNum();
  size_t secondShapeSize = transDataShape_2.GetDimNum();
  if (!((firstShapeSize == 3 && secondShapeSize == 4) || (firstShapeSize == 4 && secondShapeSize == 3))) {
    OP_LOGI(TbeGetName(op).c_str(), "the dims of transdata shape not match the fusion condition!");
    return NOT_CHANGED;
  }
  if (firstShapeSize == 3) {
    if (!(transDataShape_1.GetDim(0) == transDataShape_2.GetDim(0) &&
        transDataShape_1.GetDim(1) == transDataShape_2.GetDim(2) &&
        transDataShape_1.GetDim(2) == transDataShape_2.GetDim(1) * transDataShape_2.GetDim(3))) {
      OP_LOGI(TbeGetName(op).c_str(), "the dims of transdata shape not match the fusion condition!");
      return NOT_CHANGED;
    }
  }
  if (firstShapeSize == 4) {
    if (!(transDataShape_2.GetDim(0) == transDataShape_1.GetDim(0) &&
        transDataShape_2.GetDim(1) == transDataShape_1.GetDim(2) &&
        transDataShape_2.GetDim(2) == transDataShape_1.GetDim(1) * transDataShape_1.GetDim(3))) {
      OP_LOGI(TbeGetName(op).c_str(), "the shape of transdata not match the fusion condition!");
      return NOT_CHANGED;
    }
  }
  // add edge from the input of transData_1 to the output of transData_2
  if (transData_2->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The output edge of TransData is [%d].",
            transData_2->GetOutDataAnchor(0)->GetPeerInDataAnchors().size());
    for (InDataAnchorPtr inAnchorPtr : transData_2->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(transData_1->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                           inAnchorPtr),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                "Add edge from fused node:%s's input to fusion node:%s's output failed.",
                                transData_1->GetName().c_str(), transData_2->GetName().c_str()),
                        return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input to fusion node:%s's output success.",
              transData_1->GetName().c_str(), transData_2->GetName().c_str());
    }
  }
  // delete transData and confusionTransposeD node
  FUSION_PASS_CHECK(graph.RemoveNode(transData_1) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove transData_1 node failed"),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(transData_2) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove transData_2 node failed"),
                    return FAILED);
  FUSION_PASS_CHECK(RemoveNode(confusionTransposeD, graph) == FAILED,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove confusionTransposeD node failed"),
                    return FAILED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define TransDataConfusionTransposeFusionPass fusion end");
  return SUCCESS;
}

REGISTER_PASS("TransDataConfusionTransposeFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS,
              TransDataConfusionTransposeFusionPass);
}  // namespace fe
