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
 * \file concat_v2d_fusion_pass.cpp
 * \brief Concatv2d fusion pass(multi Concatv2d --> single Concatv2d)
 */
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>

#include "concat_v2d_fusion_pass.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {
static const string CONCATV2D = "ConcatV2D";
static const std::string PATTERN_FUSEDNODE = "FusedNodeConcat";
static const char ATTR_CONCAT_DIM[] = "concat_dim";
vector<FusionPattern*> Concatv2dFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("Concatv2dFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {CONCATV2D}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);

  return patterns;
}

bool Concatv2dFusionPass::CheckConcatValid(ge::NodePtr node, ge::Format format, ge::GeShape shape, int32_t dimNum) {
  int32_t concatDim = 0;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetInt(node->GetOpDesc(), ATTR_CONCAT_DIM, concatDim),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "There is no concat_dim attr."), return false);

  FUSION_PASS_CHECK(dimNum != concatDim,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "%s 's concat_dim is %ld, target is %ld, can't fuss.",
                            node->GetName().c_str(), concatDim, dimNum),
                    return false);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "%s 's concat_dim is %ld, target is %ld.", node->GetName().c_str(), concatDim, dimNum);

  ge::Format inputFormat = node->GetOpDesc()->GetInputDesc(0).GetFormat();
  FUSION_PASS_CHECK(inputFormat != format,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "%s 's input format is %s, target is %s, can't fuss.",
                            node->GetName().c_str(), ge::TypeUtils::FormatToSerialString(inputFormat).c_str(),
                            ge::TypeUtils::FormatToSerialString(format).c_str()),
                    return false);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "%s 's input format is %s, target is %s.", node->GetName().c_str(),
          ge::TypeUtils::FormatToSerialString(inputFormat).c_str(),
          ge::TypeUtils::FormatToSerialString(format).c_str());

  ge::GeShape inputShape = node->GetOpDesc()->GetInputDesc(0).GetShape();
  int32_t index = 0;
  for (auto dim : inputShape.GetDims()) {
    if (index == dimNum) {
      continue;
    }
    FUSION_PASS_CHECK(dim != shape.GetDim(index),
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "%s 's input %ld dim is %ld, target is %ld, can't fuss.",
                              node->GetName().c_str(), index, dim, shape.GetDim(index)),
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "%s 's input %ld dim is %ld, target is %ld.",
                              node->GetName().c_str(), index, dim, shape.GetDim(index));
                      return false);
    index++;
  }
  uint32_t OutNodesSize = 1;
  FUSION_PASS_CHECK(node->GetOutAllNodes().size() != OutNodesSize,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "%s 's output should be 1, can't fuss.", node->GetName().c_str()),
                    return false);
  return true;
}

Status Concatv2dFusionPass::PatternParse(ge::NodePtr concatv2dNode, vector<ge::NodePtr>& fusedInputNodes,
                                         vector<ge::NodePtr>& concatNodes) {
  int32_t directOutNodeNum;
  directOutNodeNum = concatv2dNode->GetOutAllNodes().size();
  FUSION_PASS_CHECK(directOutNodeNum <= 0,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "There is no need to fusion, out node num %d.", directOutNodeNum),
                    return FAILED);
  int32_t concatDim = 0;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetInt(concatv2dNode->GetOpDesc(), ATTR_CONCAT_DIM, concatDim),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "There is no need to fusion."), return FAILED);

  ge::Format format = concatv2dNode->GetOpDesc()->GetInputDesc(0).GetFormat();
  ge::GeShape shape = concatv2dNode->GetOpDesc()->GetInputDesc(0).GetShape();
  for (auto nodePtr : concatv2dNode->GetInAllNodes()) {
    if (nodePtr->GetType() == CONCATV2D) {
      FUSION_PASS_CHECK(!CheckConcatValid(nodePtr, format, shape, concatDim),
                        OP_LOGI(FUSED_OP_TYPE.c_str(), "There is no need to fusion."), return FAILED);
      concatNodes.push_back(nodePtr);
      for (auto preNode : nodePtr->GetInAllNodes()) {
        FUSION_PASS_CHECK(
            preNode->GetOutAllNodes().size() != 1,
            OP_LOGD(FUSED_OP_TYPE.c_str(), "%s 's output should be 1, can't fuss.", preNode->GetName().c_str()),
            return FAILED);
        fusedInputNodes.push_back(preNode);
      }
    } else {
      FUSION_PASS_CHECK(
          nodePtr->GetOutAllNodes().size() != 1,
          OP_LOGD(FUSED_OP_TYPE.c_str(), "%s 's output should be 1, can't fuss.", nodePtr->GetName().c_str()),
          return FAILED);
      fusedInputNodes.push_back(nodePtr);
    }
  }
  FUSION_PASS_CHECK(!concatNodes.size(), OP_LOGD(FUSED_OP_TYPE.c_str(), "Singel concat, no need fusion."),
                    return FAILED);
  return SUCCESS;
}

Status Concatv2dFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr concatv2dNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);

  FUSION_PASS_CHECK(concatv2dNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed"),
                    return PARAM_INVALID);
  vector<ge::NodePtr> fusedInputNodes;
  vector<ge::NodePtr> concatNodes;

  if (SUCCESS != PatternParse(concatv2dNode, fusedInputNodes, concatNodes)) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "do not need do concatv2d fusion here, concatv2d name %s",
            concatv2dNode->GetName().c_str());
    fusedInputNodes.clear();
    concatNodes.clear();
    return NOT_CHANGED;
  }
  
  if (fusedInputNodes.size() > 63) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "cocnat input tensors number %d more than 63", fusedInputNodes.size());
    fusedInputNodes.clear();
    concatNodes.clear();
    return NOT_CHANGED;
  }

  ge::OpDescPtr fusedConcatv2dOpDesc = AttrUtils::CloneOpDesc(concatv2dNode->GetOpDesc());
  FUSION_PASS_CHECK(
      fusedConcatv2dOpDesc == nullptr,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.", concatv2dNode->GetName().c_str()),
      return PARAM_INVALID);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "fusedConcatv2dOpDesc %s, optye %s, input %ld, output %ld",
          fusedConcatv2dOpDesc->GetName().c_str(), fusedConcatv2dOpDesc->GetType().c_str(),
          fusedConcatv2dOpDesc->GetAllInputsDesc().size(), fusedConcatv2dOpDesc->GetAllOutputsDesc().size());
  fusedConcatv2dOpDesc->SetName(concatv2dNode->GetName());
  fusedConcatv2dOpDesc->SetType(CONCATV2D);

  uint32_t index = 0;
  for (auto inputDesc : fusedConcatv2dOpDesc->GetAllInputsDesc()) {
    FUSION_PASS_CHECK(!ge::OpDescUtils::ClearInputDesc(fusedConcatv2dOpDesc, index),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node:%s's clear %d th input failed.",
                              fusedConcatv2dOpDesc->GetName().c_str(), index),
                      return PARAM_INVALID);
  }

  size_t input_idx = 0;
  for (auto inputNode : fusedInputNodes) {
    uint32_t index = inputNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetIdx();
    ge::GeTensorDesc desc = inputNode->GetOutAllNodes().at(0)->GetOpDesc()->GetInputDesc(index);
    string name = "x" + std::to_string(input_idx);
    fusedConcatv2dOpDesc->AddInputDesc(name, desc);
    input_idx++;
  }

  int64_t num_N_new = fusedInputNodes.size();
  OP_LOGD(FUSED_OP_TYPE.c_str(),"Node:%s's has %ld inputs.", fusedConcatv2dOpDesc->GetName().c_str(),
          num_N_new);
  ge::AttrUtils::SetInt(fusedConcatv2dOpDesc, "N", num_N_new);

  ge::NodePtr fusedConcatv2dNode = graph.AddNode(fusedConcatv2dOpDesc);
  std::map<string, uint32_t> output_name_id = {{"y", 0}};
  fusedConcatv2dNode->GetOpDesc()->UpdateOutputName(output_name_id);

  index = 0;
  for (auto inputNode : fusedInputNodes) {
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(inputNode->GetOutDataAnchor(0),
                                              fusedConcatv2dNode->GetInDataAnchor(index)) != ge::GRAPH_SUCCESS,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "add input edge between %s and %s %d th failed.",
                              inputNode->GetName().c_str(), fusedConcatv2dNode->GetName().c_str(), index),
                      return NOT_CHANGED);
    index++;
  }

  index = 0;
  for (auto outAnchor : concatv2dNode->GetAllOutDataAnchors()) {
    for (InDataAnchorPtr inAnchorPtr : outAnchor->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusedConcatv2dNode->GetOutDataAnchor(index), inAnchorPtr),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from %s to fusion node %s's %d th failed.",
                                concatv2dNode->GetName().c_str(), fusedConcatv2dNode->GetName().c_str(), index),
                        return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from %s to fusion node %s's %d index success.",
              concatv2dNode->GetName().c_str(), fusedConcatv2dNode->GetName().c_str(), index);
    }
  }

  concatNodes.push_back(concatv2dNode);
  for (auto concatNode : concatNodes) {
    for (auto inAnchor : concatNode->GetAllInDataAnchors()) {
      if (inAnchor) {
        inAnchor->UnlinkAll();
      }
    }
  }

  for (auto concatNode : concatNodes) {
    FUSION_PASS_CHECK(graph.RemoveNode(concatNode) == ge::GRAPH_FAILED,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove node %s failed.", concatNode->GetName().c_str()),
                      return FAILED);
  }
  return SUCCESS;
}

REGISTER_PASS("ZConcatv2dFusionPass", BUILT_IN_GRAPH_PASS, Concatv2dFusionPass);
}  // namespace fe
