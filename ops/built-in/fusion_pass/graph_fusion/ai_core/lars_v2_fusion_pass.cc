/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @brief LARS fusion pass(LARS --> lars_v2_update + SquareSumV1 + SquareSumV1)
 *
 * @author t00474913
 */

#include "lars_v2_fusion_pass.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "op_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {
static const char *FUSED_NODE = "LarsV2";

static const std::string PATTERN_FUSEDNODE = "LarsV2";

Status CreateSquareSumAllOp(ge::OpDescPtr src_op_desc,
                           ge::OpDescPtr new_op_desc) {
  ge::GeTensorDesc new_op_output_desc(GeShape(), ge::FORMAT_ND, ge::DT_FLOAT);

  new_op_desc->AddInputDesc("x1", src_op_desc->GetInputDesc("w"));
  new_op_desc->AddInputDesc("x2", src_op_desc->GetInputDesc("g"));
  new_op_desc->AddOutputDesc("y1", new_op_output_desc);
  new_op_desc->AddOutputDesc("y2", new_op_output_desc);

  return SUCCESS;
}

vector<FusionPattern *> LarsV2FusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern = new (std::nothrow) FusionPattern("LarsFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
           return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE})
      .SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

Status LarsV2FusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                                vector<ge::NodePtr> &fusionNodes) {
  ge::NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fused_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
           return PARAM_INVALID);

  ge::OpDescPtr fused_desc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(fused_desc == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode's OpDesc is null, fusion failed."),
           return PARAM_INVALID);

  // set lars_v2_update desc
  ge::OpDescPtr lars_v2_update_desc = AttrUtils::CloneOpDesc(fused_desc);
  FUSION_PASS_CHECK(lars_v2_update_desc == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.",
                   fused_node->GetName().c_str()),
           return PARAM_INVALID);
  lars_v2_update_desc->SetName(fused_desc->GetName() + "/LarsUpdate");
  lars_v2_update_desc->SetType("LarsV2Update");

  // new the square_sum_all node
  ge::OpDescPtr square_sum_all_desc = std::make_shared<ge::OpDesc>(
      fused_desc->GetName() + "/SquareSumAll", "SquareSumAll");

  CreateSquareSumAllOp(fused_desc, square_sum_all_desc);


  lars_v2_update_desc->UpdateInputDesc(2,
                                       square_sum_all_desc->GetOutputDesc(0));
  lars_v2_update_desc->UpdateInputDesc(3,
                                       square_sum_all_desc->GetOutputDesc(1));
  lars_v2_update_desc->AddInputDesc(4, fused_desc->GetInputDesc(2));
  lars_v2_update_desc->AddInputDesc(5, fused_desc->GetInputDesc(3));

  ge::NodePtr square_sum_all_node = graph.AddNode(square_sum_all_desc);
  ge::NodePtr lars_v2_update_node = graph.AddNode(lars_v2_update_desc);
  fusionNodes.push_back(square_sum_all_node);
  fusionNodes.push_back(lars_v2_update_node);

  FUSION_PASS_CHECK(square_sum_all_node == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                   square_sum_all_node->GetName().c_str()),
           return PARAM_INVALID);
  FUSION_PASS_CHECK(lars_v2_update_node == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                   lars_v2_update_node->GetName().c_str()),
           return PARAM_INVALID);

  // add square_sum_all op input edge
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(
                          fused_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                          square_sum_all_node->GetInDataAnchor(0)),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index 0 to fusion node:%s's "
                   "index 0 failed.",
                   fused_node->GetName().c_str(),
                   square_sum_all_node->GetName().c_str()),
           return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index 0 to fusion node:%s's index 0.",
          fused_node->GetName().c_str(),
          square_sum_all_node->GetName().c_str());

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(
                          fused_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
                          square_sum_all_node->GetInDataAnchor(1)),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index 1 to fusion node:%s's "
                   "index 1 failed.",
                   fused_node->GetName().c_str(),
                   square_sum_all_node->GetName().c_str()),
           return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index 1 to fusion node:%s's index 1.",
          fused_node->GetName().c_str(),
          square_sum_all_node->GetName().c_str());

  // add control anchor between square_sum_all op and  it's input nodes.
  for (unsigned int i = 0;
       i < fused_node->GetInControlAnchor()->GetPeerOutControlAnchors().size();
       i++) {
    FUSION_PASS_CHECK(
        SUCCESS !=
            ge::GraphUtils::AddEdge(
                fused_node->GetInControlAnchor()->GetPeerOutControlAnchors().at(
                    i),
                square_sum_all_node->GetInControlAnchor()),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's control index[%d] to fusion "
                "node:%s's control index failed.",
                fused_node->GetName().c_str(), i,
                square_sum_all_node->GetName().c_str()),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(),
        "Add edge from fused node:%s's control index[%d] to fusion node:%s's "
        "control index.",
        fused_node->GetName().c_str(), i,
        square_sum_all_node->GetName().c_str());
  }

  // add edge for lars_v2_update op input 0
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(
                          fused_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                          lars_v2_update_node->GetInDataAnchor(0)),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index 0 to fusion node:%s's "
                   "index 0 failed.",
                   fused_node->GetName().c_str(),
                   lars_v2_update_node->GetName().c_str()),
           return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index 0 to fusion node:%s's index 0.",
          fused_node->GetName().c_str(),
          lars_v2_update_node->GetName().c_str());

  // add edge for lars_v2_update op input 1
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(
                          fused_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
                          lars_v2_update_node->GetInDataAnchor(1)),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index 1 to fusion node:%s's "
                   "index 1 failed.",
                   fused_node->GetName().c_str(),
                   lars_v2_update_node->GetName().c_str()),
           return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index 1 to fusion node:%s's index 1.",
          fused_node->GetName().c_str(),
          lars_v2_update_node->GetName().c_str());

  // add edge for lars_v2_update op input 2
  FUSION_PASS_CHECK(SUCCESS !=
               ge::GraphUtils::AddEdge(square_sum_all_node->GetOutDataAnchor(0),
                                       lars_v2_update_node->GetInDataAnchor(2)),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index 0 to fusion node:%s's "
                   "index 2 failed.",
                   square_sum_all_node->GetName().c_str(),
                   lars_v2_update_node->GetName().c_str()),
           return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index 0 to fusion node:%s's index 3",
          square_sum_all_node->GetName().c_str(),
          lars_v2_update_node->GetName().c_str());

  // add edge for lars_v2_update op input 3
  FUSION_PASS_CHECK(SUCCESS !=
               ge::GraphUtils::AddEdge(square_sum_all_node->GetOutDataAnchor(1),
                                       lars_v2_update_node->GetInDataAnchor(3)),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index 1 to fusion node:%s's "
                   "index 3 failed.",
                   square_sum_all_node->GetName().c_str(),
                   lars_v2_update_node->GetName().c_str()),
           return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index 0 to fusion node:%s's index 3",
          square_sum_all_node->GetName().c_str(),
          lars_v2_update_node->GetName().c_str());

  // add edge for lars_v2_update op input 4
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(
                          fused_node->GetInDataAnchor(2)->GetPeerOutAnchor(),
                          lars_v2_update_node->GetInDataAnchor(4)),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index 2 to fusion node:%s's "
                   "index 4 failed.",
                   fused_node->GetName().c_str(),
                   lars_v2_update_node->GetName().c_str()),
           return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index 2 to fusion node:%s's index 4.",
          fused_node->GetName().c_str(),
          lars_v2_update_node->GetName().c_str());

  // add edge for lars_v2_update op input 5
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(
                          fused_node->GetInDataAnchor(3)->GetPeerOutAnchor(),
                          lars_v2_update_node->GetInDataAnchor(5)),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index 3 to fusion node:%s's "
                   "index 5 failed.",
                   fused_node->GetName().c_str(),
                   lars_v2_update_node->GetName().c_str()),
           return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index 3 to fusion node:%s's index 5.",
          fused_node->GetName().c_str(),
          lars_v2_update_node->GetName().c_str());

  // add lars_v2_update out edge
  if (fused_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The size of LARS out_data_anchor peer in_data_anchors is [%d].",
            fused_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size());
    for (InDataAnchorPtr inAnchorPtr :
         fused_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(
                         lars_v2_update_node->GetOutDataAnchor(0), inAnchorPtr),
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index to fusion node:%s's 1st "
                  "index failed.",
                  fused_node->GetName().c_str(),
                  lars_v2_update_node->GetName().c_str()),
          return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(),
          "Add edge from fused node:%s's 1st index to fusion node:%s's 1st "
          "index.",
          fused_node->GetName().c_str(),
          lars_v2_update_node->GetName().c_str());
    }
  }

  // add control anchor between lars_v2_update and out node of lars.
  for (auto lars_v2_out_control_node : fused_node->GetOutControlNodes()) {
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(lars_v2_update_node->GetOutControlAnchor(),
                                lars_v2_out_control_node->GetInControlAnchor()),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                lars_v2_update_node->GetName().c_str(),
                lars_v2_out_control_node->GetName().c_str()),
        return FAILED);
  }

  if (fused_node->GetInControlAnchor() != nullptr) {
    fused_node->GetInControlAnchor()->UnlinkAll();
  }
  for (auto inAnchor : fused_node->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(fused_node),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "remove fusedNode node[%s] failed",
                   fused_node->GetName().c_str()),
           return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "LarsFusionPass graph fusion success!");
  return SUCCESS;
}
REGISTER_PASS("LarsFusionPass", BUILT_IN_GRAPH_PASS, LarsV2FusionPass);
}  // namespace fe
