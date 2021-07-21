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
 * \file lars_v2_fusion_pass.cpp
 * \brief LARS fusion pass(LARS --> lars_v2_update + SquareSumV1 + SquareSumV1)
 */
#include "lars_v2_fusion_pass.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {
static const std::string kPatternFusedNode = "LarsV2";

Status CreateSquareSumAllOp(OpDescPtr src_op_desc, OpDescPtr new_op_desc) {
  GeTensorDesc new_op_output_desc(GeShape(), FORMAT_ND, DT_FLOAT);
  new_op_desc->AddInputDesc("x1", src_op_desc->GetInputDesc("w"));
  new_op_desc->AddInputDesc("x2", src_op_desc->GetInputDesc("g"));
  new_op_desc->AddOutputDesc("y1", new_op_output_desc);
  new_op_desc->AddOutputDesc("y2", new_op_output_desc);
  return SUCCESS;
}

vector<FusionPattern*> LarsV2FusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("LarsFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "New a pattern object failed."), return patterns);
  pattern->AddOpDesc(kPatternFusedNode, {"LarsV2"}).SetOutput(kPatternFusedNode);
  patterns.push_back(pattern);
  return patterns;
}

Status LarsV2FusionPass::Fusion(ComputeGraph& graph, Mapping& mapping, vector<NodePtr>& fusion_nodes) {
  NodePtr fused_node = GetNodeFromMapping(kPatternFusedNode, mapping);
  FUSION_PASS_CHECK(fused_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The fusedNode is null, fusion failed."),
                    return PARAM_INVALID);

  OpDescPtr fused_desc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(fused_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The fusedNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  // set lars_v2_update desc
  OpDescPtr lars_v2_update_desc = AttrUtils::CloneOpDesc(fused_desc);
  FUSION_PASS_CHECK(
      lars_v2_update_desc == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Node:%s's OpDesc is null, fusion failed.", fused_node->GetName().c_str()),
      return PARAM_INVALID);
  lars_v2_update_desc->SetName(fused_desc->GetName() + "/LarsUpdate");
  lars_v2_update_desc->SetType("LarsV2Update");

  // new the square_sum_all node
  OpDescPtr square_sum_all_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (square_sum_all_desc = std::make_shared<ge::OpDesc>(fused_desc->GetName() + "/SquareSumAll", "SquareSumAll")),
      return PARAM_INVALID);

  CreateSquareSumAllOp(fused_desc, square_sum_all_desc);

  lars_v2_update_desc->UpdateInputDesc(2, square_sum_all_desc->GetOutputDesc(0));
  lars_v2_update_desc->UpdateInputDesc(3, square_sum_all_desc->GetOutputDesc(1));
  lars_v2_update_desc->AddInputDesc(4, fused_desc->GetInputDesc(2));
  lars_v2_update_desc->AddInputDesc(5, fused_desc->GetInputDesc(3));

  NodePtr square_sum_all_node = graph.AddNode(square_sum_all_desc);
  NodePtr lars_v2_update_node = graph.AddNode(lars_v2_update_desc);
  fusion_nodes.push_back(square_sum_all_node);
  fusion_nodes.push_back(lars_v2_update_node);

  FUSION_PASS_CHECK(square_sum_all_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The fusionNode:%s is null, fusion failed.",
                            square_sum_all_node->GetName().c_str()),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(lars_v2_update_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The fusionNode:%s is null, fusion failed.",
                            lars_v2_update_node->GetName().c_str()),
                    return PARAM_INVALID);

  // add square_sum_all op input edge
  auto fused_in_data_anchor0 = fused_node->GetInDataAnchor(0);
  FUSION_PASS_CHECK(fused_in_data_anchor0 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The fused_in_data_anchor0 is null, fusion failed."),
                    return PARAM_INVALID);
  auto fused_in_data_anchor1 = fused_node->GetInDataAnchor(1);
  FUSION_PASS_CHECK(fused_in_data_anchor1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The fused_in_data_anchor1 is null, fusion failed."),
                    return PARAM_INVALID);
  auto fused_in_data_anchor2 = fused_node->GetInDataAnchor(2);
  FUSION_PASS_CHECK(fused_in_data_anchor2 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The fused_in_data_anchor2 is null, fusion failed."),
                    return PARAM_INVALID);
  auto fused_in_data_anchor3 = fused_node->GetInDataAnchor(3);
  FUSION_PASS_CHECK(fused_in_data_anchor3 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The fused_in_data_anchor3 is null, fusion failed."),
                    return PARAM_INVALID);

  FUSION_PASS_CHECK(
      GraphUtils::AddEdge(fused_in_data_anchor0->GetPeerOutAnchor(), square_sum_all_node->GetInDataAnchor(0)) !=
          SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add edge from fused node:%s's index 0 to fusion node:%s's index 0 failed.",
              fused_node->GetName().c_str(), square_sum_all_node->GetName().c_str()),
      return FAILED);
  OP_LOGD(kFusedOpType.c_str(), "Add edge from fused node:%s's index 0 to fusion node:%s's index 0.",
          fused_node->GetName().c_str(), square_sum_all_node->GetName().c_str());

  FUSION_PASS_CHECK(
      GraphUtils::AddEdge(fused_in_data_anchor1->GetPeerOutAnchor(), square_sum_all_node->GetInDataAnchor(1)) !=
          SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add edge from fused node:%s's index 1 to fusion node:%s's index 1 failed.",
              fused_node->GetName().c_str(), square_sum_all_node->GetName().c_str()),
      return FAILED);
  OP_LOGD(kFusedOpType.c_str(), "Add edge from fused node:%s's index 1 to fusion node:%s's index 1.",
          fused_node->GetName().c_str(), square_sum_all_node->GetName().c_str());

  // add control anchor between square_sum_all op and  it's input nodes.
  for (unsigned int i = 0; i < fused_node->GetInControlAnchor()->GetPeerOutControlAnchors().size(); i++) {
    FUSION_PASS_CHECK(
        GraphUtils::AddEdge(fused_node->GetInControlAnchor()->GetPeerOutControlAnchors().at(i),
                            square_sum_all_node->GetInControlAnchor()) != SUCCESS,
        VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(),
                "Add edge from fused node:%s's control index[%d] to fusion node:%s's control index failed.",
                fused_node->GetName().c_str(), i, square_sum_all_node->GetName().c_str()),
        return FAILED);
    OP_LOGD(kFusedOpType.c_str(), "Add edge from fused node:%s's control index[%d] to fusion node:%s's control index.",
            fused_node->GetName().c_str(), i, square_sum_all_node->GetName().c_str());
  }

  // add edge for lars_v2_update op input 0
  FUSION_PASS_CHECK(
      GraphUtils::AddEdge(fused_in_data_anchor0->GetPeerOutAnchor(), lars_v2_update_node->GetInDataAnchor(0)) !=
          SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add edge from fused node:%s's index 0 to fusion node:%s's index 0 failed.",
              fused_node->GetName().c_str(), lars_v2_update_node->GetName().c_str()),
      return FAILED);
  OP_LOGD(kFusedOpType.c_str(), "Add edge from fused node:%s's index 0 to fusion node:%s's index 0.",
          fused_node->GetName().c_str(), lars_v2_update_node->GetName().c_str());

  // add edge for lars_v2_update op input 1
  FUSION_PASS_CHECK(
      GraphUtils::AddEdge(fused_in_data_anchor1->GetPeerOutAnchor(), lars_v2_update_node->GetInDataAnchor(1)) !=
          SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add edge from fused node:%s's index 1 to fusion node:%s's index 1 failed.",
              fused_node->GetName().c_str(), lars_v2_update_node->GetName().c_str()),
      return FAILED);
  OP_LOGD(kFusedOpType.c_str(), "Add edge from fused node:%s's index 1 to fusion node:%s's index 1.",
          fused_node->GetName().c_str(), lars_v2_update_node->GetName().c_str());

  // add edge for lars_v2_update op input 2
  FUSION_PASS_CHECK(
      GraphUtils::AddEdge(square_sum_all_node->GetOutDataAnchor(0), lars_v2_update_node->GetInDataAnchor(2)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add edge from fused node:%s's index 0 to fusion node:%s's index 2 failed.",
              square_sum_all_node->GetName().c_str(), lars_v2_update_node->GetName().c_str()),
      return FAILED);
  OP_LOGD(kFusedOpType.c_str(), "Add edge from fused node:%s's index 0 to fusion node:%s's index 3",
          square_sum_all_node->GetName().c_str(), lars_v2_update_node->GetName().c_str());

  // add edge for lars_v2_update op input 3
  FUSION_PASS_CHECK(
      GraphUtils::AddEdge(square_sum_all_node->GetOutDataAnchor(1), lars_v2_update_node->GetInDataAnchor(3)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add edge from fused node:%s's index 1 to fusion node:%s's index 3 failed.",
              square_sum_all_node->GetName().c_str(), lars_v2_update_node->GetName().c_str()),
      return FAILED);
  OP_LOGD(kFusedOpType.c_str(), "Add edge from fused node:%s's index 0 to fusion node:%s's index 3",
          square_sum_all_node->GetName().c_str(), lars_v2_update_node->GetName().c_str());

  // add edge for lars_v2_update op input 4
  FUSION_PASS_CHECK(
      GraphUtils::AddEdge(fused_in_data_anchor2->GetPeerOutAnchor(), lars_v2_update_node->GetInDataAnchor(4)) !=
          SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add edge from fused node:%s's index 2 to fusion node:%s's index 4 failed.",
              fused_node->GetName().c_str(), lars_v2_update_node->GetName().c_str()),
      return FAILED);
  OP_LOGD(kFusedOpType.c_str(), "Add edge from fused node:%s's index 2 to fusion node:%s's index 4.",
          fused_node->GetName().c_str(), lars_v2_update_node->GetName().c_str());

  // add edge for lars_v2_update op input 5
  FUSION_PASS_CHECK(
      GraphUtils::AddEdge(fused_in_data_anchor3->GetPeerOutAnchor(), lars_v2_update_node->GetInDataAnchor(5)) !=
          SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add edge from fused node:%s's index 3 to fusion node:%s's index 5 failed.",
              fused_node->GetName().c_str(), lars_v2_update_node->GetName().c_str()),
      return FAILED);
  OP_LOGD(kFusedOpType.c_str(), "Add edge from fused node:%s's index 3 to fusion node:%s's index 5.",
          fused_node->GetName().c_str(), lars_v2_update_node->GetName().c_str());

  // add lars_v2_update out edge
  auto fused_out_data_anchor0 = fused_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(fused_out_data_anchor0 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The fused_out_data_anchor0 is null, fusion failed."),
                    return PARAM_INVALID);
  if (fused_out_data_anchor0->GetPeerInDataAnchors().size() > 0) {
    OP_LOGI(kFusedOpType.c_str(), "The size of LARS out_data_anchor peer in_data_anchors is [%u].",
            fused_out_data_anchor0->GetPeerInDataAnchors().size());
    for (InDataAnchorPtr in_anchor_ptr : fused_out_data_anchor0->GetPeerInDataAnchors()) {
      in_anchor_ptr->UnlinkAll();
      FUSION_PASS_CHECK(
          SUCCESS != GraphUtils::AddEdge(lars_v2_update_node->GetOutDataAnchor(0), in_anchor_ptr),
          VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add edge from fused node:%s's index to fusion node:%s's 1st index failed.",
                  fused_node->GetName().c_str(), lars_v2_update_node->GetName().c_str()),
          return FAILED);
      OP_LOGD(kFusedOpType.c_str(), "Add edge from fused node:%s's 1st index to fusion node:%s's 1st index.",
              fused_node->GetName().c_str(), lars_v2_update_node->GetName().c_str());
    }
  }

  // add control anchor between lars_v2_update and out node of lars.
  for (auto lars_v2_out_control_node : fused_node->GetOutControlNodes()) {
    FUSION_PASS_CHECK(
        GraphUtils::AddEdge(lars_v2_update_node->GetOutControlAnchor(), lars_v2_out_control_node->GetInControlAnchor()),
        VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add edge between node %s. and node %s failed.",
                lars_v2_update_node->GetName().c_str(), lars_v2_out_control_node->GetName().c_str()),
        return FAILED);
  }

  if (fused_node->GetInControlAnchor() != nullptr) {
    fused_node->GetInControlAnchor()->UnlinkAll();
  }
  for (auto in_anchor : fused_node->GetAllInDataAnchors()) {
    if (in_anchor != nullptr) {
      in_anchor->UnlinkAll();
    }
  }
  FUSION_PASS_CHECK(GRAPH_SUCCESS != graph.RemoveNode(fused_node),
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Remove fusedNode node[%s] failed", fused_node->GetName().c_str()),
                    return FAILED);

  OP_LOGI(kFusedOpType.c_str(), "LarsFusionPass graph fusion success!");
  return SUCCESS;
}
REGISTER_PASS("LarsFusionPass", BUILT_IN_GRAPH_PASS, LarsV2FusionPass);
}  // namespace fe
