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
 * \file mul_addn_l2loss_fusion_pass.cpp
 * \brief mul addn(two input) fusion pass(mul --> addn)
 */
#include "mul_addn_l2loss_fusion_pass.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/fusion_common/graph_pass_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "register/op_registry.h"

using namespace ge;
namespace fe {
static const string kFusionNodePatternIdAddn = "AddN";
static const string kFusionNodePatternIdMul = "Mul";
static const std::string kStreamLabel = "_stream_label";

vector<FusionPattern*> MulAddNL2LossFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("MulAddNL2LossFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(kFusedOpType.c_str(), "New a pattern object failed."), return patterns);

  pattern->AddOpDesc(kFusionNodePatternIdAddn, {"AddN"})
      .AddOpDesc(kFusionNodePatternIdMul, {"Mul"})
      .SetInputs(kFusionNodePatternIdAddn, {kFusionNodePatternIdMul})
      .SetOutput(kFusionNodePatternIdAddn);
  patterns.push_back(pattern);

  return patterns;
}

Status MulAddNL2LossFusionPass::Fusion(ComputeGraph& graph, Mapping& mapping, vector<NodePtr>& new_nodes) {
  OP_LOGD(kFusedOpType.c_str(), "Enter graph fusion MulAddNFusionPass!");
  NodePtr addn_node = GetNodeFromMapping(kFusionNodePatternIdAddn, mapping);
  NodePtr mul_node = GetNodeFromMapping(kFusionNodePatternIdMul, mapping);

  FUSION_PASS_CHECK(addn_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "The addn_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mul_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "The mul_node is null, fusion failed."),
                    return PARAM_INVALID);
  auto in_data_anchor0 = mul_node->GetInDataAnchor(0);
  FUSION_PASS_CHECK(in_data_anchor0 == nullptr,
                    OP_LOGE(kFusedOpType.c_str(), "The in_data_anchor0 is null, fusion failed."), return PARAM_INVALID);
  auto peer_out_anchor0 = in_data_anchor0->GetPeerOutAnchor();
  FUSION_PASS_CHECK(peer_out_anchor0 == nullptr,
                    OP_LOGE(kFusedOpType.c_str(), "The peer_out_anchor0 is null, fusion failed."),
                    return PARAM_INVALID);
  NodePtr variable_node = peer_out_anchor0->GetOwnerNode();
  FUSION_PASS_CHECK(variable_node->GetType() != "Variable" && variable_node->GetType() != "TransData",
                    OP_LOGI(kFusedOpType.c_str(), "Mul node[%s]'s 1st input must be a variable, actuall is [%s].",
                            mul_node->GetName().c_str(), variable_node->GetType().c_str()),
                    return NOT_CHANGED);
  if (variable_node->GetType() == "TransData") {
    FUSION_PASS_CHECK(variable_node->GetInDataNodes().empty(),
                      OP_LOGE(kFusedOpType.c_str(), "The size of in_data_node is 0, fusion failed."),
                      return PARAM_INVALID);
    FUSION_PASS_CHECK(
        variable_node->GetInDataNodes().at(0)->GetType() != "Variable",
        OP_LOGI(kFusedOpType.c_str(), "TransData node[%s]'s 1st input must be a variable, actuall is [%s].",
                variable_node->GetName().c_str(), variable_node->GetInDataNodes().at(0)->GetType().c_str()),
        return NOT_CHANGED);
  }
  OP_LOGI(kFusedOpType.c_str(), "Get variable node[%s] success.", variable_node->GetName().c_str());
  auto variable_out_nodes = variable_node->GetOutAllNodes();
  NodePtr l2loss_node = nullptr;
  for (auto variable_out_node : variable_out_nodes) {
    if (variable_out_node->GetType() == "L2Loss") {
      l2loss_node = variable_out_node;
      break;
    }
  }

  FUSION_PASS_CHECK(
      l2loss_node == nullptr,
      OP_LOGI(kFusedOpType.c_str(), "Variable node[%s] don't have l2loss out node.", variable_node->GetName().c_str()),
      return NOT_CHANGED);
  OP_LOGI(kFusedOpType.c_str(), "Get l2loss node[%s] success.", l2loss_node->GetName().c_str());
  FUSION_PASS_CHECK(
      l2loss_node->GetOutAllNodes().size() != 1,
      OP_LOGI(kFusedOpType.c_str(), "L2Loss node[%s] should only have one output node, but actually is [%u].",
              l2loss_node->GetName().c_str(), l2loss_node->GetOutAllNodes().size()),
      return NOT_CHANGED);
  auto out_data_anchor0 = l2loss_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(out_data_anchor0 == nullptr,
                    OP_LOGE(kFusedOpType.c_str(), "The out_data_anchor0 is null, fusion failed."),
                    return PARAM_INVALID);
  auto first_peer_anchor = out_data_anchor0->GetFirstPeerAnchor();
  FUSION_PASS_CHECK(first_peer_anchor == nullptr,
                    OP_LOGE(kFusedOpType.c_str(), "The first_peer_anchor is null, fusion failed."),
                    return PARAM_INVALID);
  NodePtr l2loss_out_node = first_peer_anchor->GetOwnerNode();
  FUSION_PASS_CHECK(
      l2loss_out_node->GetType() != "AddN",
      OP_LOGI(kFusedOpType.c_str(), "L2Loss node[%s]'s output node must be AddN.", l2loss_out_node->GetName().c_str()),
      return NOT_CHANGED);

  string addn_stream_label;
  string mul_stream_label;
  string l2loss_stream_label;
  AttrUtils::GetStr(addn_node->GetOpDesc(), kStreamLabel, addn_stream_label);
  AttrUtils::GetStr(mul_node->GetOpDesc(), kStreamLabel, mul_stream_label);
  AttrUtils::GetStr(l2loss_node->GetOpDesc(), kStreamLabel, l2loss_stream_label);
  OP_LOGI(kFusedOpType.c_str(), "AddN node's stream label is %s", addn_stream_label.c_str());
  OP_LOGI(kFusedOpType.c_str(), "Mul node's stream label is %s", mul_stream_label.c_str());
  OP_LOGI(kFusedOpType.c_str(), "L2Loss node's stream label is %s", l2loss_stream_label.c_str());
  if (addn_stream_label != mul_stream_label || l2loss_stream_label != mul_stream_label) {
    OP_LOGI(kFusedOpType.c_str(), "Node [%s], node [%s] and node [%s] don't hava same stream label.",
            addn_node->GetName().c_str(), mul_node->GetName().c_str(), l2loss_node->GetName().c_str());
    return NOT_CHANGED;
  } else {
    OP_LOGI(kFusedOpType.c_str(), "AddN and Mul have same _stream_label attr.");
  }

  FUSION_PASS_CHECK(
      addn_node->GetType() != "AddN",
      OP_LOGI(kFusedOpType.c_str(), "Get the addn_node type is [%s], which is not AddN.", addn_node->GetType().c_str()),
      return NOT_CHANGED);

  FUSION_PASS_CHECK(mul_node->GetOutDataNodes().size() != 1,
                    OP_LOGI(kFusedOpType.c_str(), "Output node of Mul node size is [%u], which not equal to 1.",
                            mul_node->GetOutDataNodes().size()),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(mul_node->GetInDataNodes().size() != 2,
                    OP_LOGI(kFusedOpType.c_str(), "Input node of Mul node size is [%u], which not equal to 2.",
                            mul_node->GetInDataNodes().size()),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(addn_node->GetInDataNodes().size() != 2,
                    OP_LOGI(kFusedOpType.c_str(), "Input node of AddN size is [%u], which not equal to 2.",
                            addn_node->GetInDataNodes().size()),
                    return NOT_CHANGED);

  for (auto mul_in_node : mul_node->GetInAllNodes()) {
    bool check_mul_in_control_nodes{false};
    if (NodeUtils::GetInConstNodeTypeCrossSubgraph(mul_in_node) != "Constant") {
      continue;
    }
    if (mul_in_node->GetInControlAnchor() == nullptr ||
        mul_in_node->GetInControlAnchor()->GetPeerOutControlAnchors().size() == 0) {
      continue;
    }
    check_mul_in_control_nodes =
        mul_in_node->GetInControlAnchor()->GetPeerOutControlAnchors().size() == 1 &&
        mul_in_node->GetInControlAnchor()->GetPeerOutControlAnchors().at(0)->GetOwnerNode()->GetType() == "NoOp";
    if (!check_mul_in_control_nodes) {
      OP_LOGI(kFusedOpType.c_str(), "Mul's input node[%s] has control anchor from node[%s], no need to do fusion.",
              mul_in_node->GetName().c_str(),
              mul_in_node->GetInControlAnchor()->GetPeerOutControlAnchors().at(0)->GetOwnerNode()->GetName().c_str());
      return NOT_CHANGED;
    }
  }

  uint32_t lossscale_input_index{0};
  for (uint32_t index = 0; index < mul_node->GetAllInDataAnchors().size(); index++) {
    NodePtr input_node_mul = mul_node->GetInDataAnchor(index)->GetPeerOutAnchor()->GetOwnerNode();
    if (NodeUtils::GetInConstNodeTypeCrossSubgraph(input_node_mul) == "Constant") {
      lossscale_input_index = index;
      break;
    }
  }

  // get mul Constant input
  GeTensorDesc const_input = mul_node->GetOpDesc()->GetInputDesc(lossscale_input_index);
  vector<int64_t> dim_info = const_input.GetShape().GetDims();
  size_t dims_size = dim_info.size();
  FUSION_PASS_CHECK(
      !(dims_size == 0 || (dims_size == 1 && dim_info[0] == 1)),
      OP_LOGI(kFusedOpType.c_str(),
              "The const input of Mul node must be scalar or dims=(1,), but dims size is [%u] and dims[0] is [%lld]",
              dims_size, dim_info[0]),
      return NOT_CHANGED);

  if (domi::OpRegistry::Instance()->GetImplyType(mul_node->GetType()) == domi::ImplyType::CCE) {
    OP_LOGI(kFusedOpType.c_str(), "Op %s is CCE, no need to fusion.", mul_node->GetName().c_str());
    return NOT_CHANGED;
  }
  if (domi::OpRegistry::Instance()->GetImplyType(addn_node->GetType()) == domi::ImplyType::CCE) {
    OP_LOGI(kFusedOpType.c_str(), "Op %s is CCE, no need to fusion.", addn_node->GetName().c_str());
    return NOT_CHANGED;
  }
  if (domi::OpRegistry::Instance()->GetImplyType(l2loss_node->GetType()) == domi::ImplyType::CCE) {
    OP_LOGI(kFusedOpType.c_str(), "Op %s is CCE, no need to fusion.", l2loss_node->GetName().c_str());
    return NOT_CHANGED;
  }
  int mul_index{0};
  bool find_addn{false};
  auto out_dataanchor_list = mul_node->GetAllOutDataAnchors();
  for (auto out_data_anchor : out_dataanchor_list) {
    for (auto peer_indata_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      if (peer_indata_anchor->GetOwnerNode()->GetType() == "AddN") {
        mul_index = peer_indata_anchor->GetIdx();
        if (mul_index != 0) {
          OP_LOGI(kFusedOpType.c_str(), "The 0th input of AddN is not Mul node, do not need fusion.");
          continue;
        }
        FUSION_PASS_CHECK(GraphUtils::RemoveEdge(out_data_anchor, peer_indata_anchor),
                          OP_LOGE(kFusedOpType.c_str(), "Remove edge failed."), return FAILED);
        find_addn = true;
      }
    }
  }
  FUSION_PASS_CHECK(!find_addn, OP_LOGI(kFusedOpType.c_str(), "The output node of Mul node is not AddN."),
                    return NOT_CHANGED);

  // add control anchor between input node of Mul and out node of Mul.
  for (auto input_node_mul : mul_node->GetInAllNodes()) {
    for (auto mul_out_control_node : mul_node->GetOutControlNodes()) {
      FUSION_PASS_CHECK(
          GraphUtils::AddEdge(input_node_mul->GetOutControlAnchor(), mul_out_control_node->GetInControlAnchor()),
          OP_LOGE(kFusedOpType.c_str(), "Add edge between node %s. and node %s failed.",
                  input_node_mul->GetName().c_str(), mul_out_control_node->GetName().c_str()),
          return FAILED);
    }
  }

  std::map<string, uint32_t> addn_input_names = addn_node->GetOpDesc()->GetAllInputName();
  for (auto index_name : addn_input_names) {
    OP_LOGI(kFusedOpType.c_str(), "The addn_node's input name is :%s, input index is %u", index_name.first.c_str(),
            index_name.second);
  }

  if (mul_node->GetInDataAnchor(lossscale_input_index)->GetPeerOutAnchor() != nullptr) {
    NodePtr input_node_mul = mul_node->GetInDataAnchor(lossscale_input_index)->GetPeerOutAnchor()->GetOwnerNode();
    // In order to prevent the lossscale nodes from being deleted when the mul
    // node is deleted, you need to unassociate them first.
    FUSION_PASS_CHECK(GraphUtils::RemoveEdge(mul_node->GetInDataAnchor(lossscale_input_index)->GetPeerOutAnchor(),
                                             mul_node->GetInDataAnchor(lossscale_input_index)),
                      OP_LOGE(kFusedOpType.c_str(), "Remove edge failed."), return FAILED);
    FUSION_PASS_CHECK(addn_node->AddLinkFrom(input_node_mul),
                      OP_LOGE(kFusedOpType.c_str(), "Add link between node %s. and node %s failed.",
                              input_node_mul->GetName().c_str(), addn_node->GetName().c_str()),
                      return FAILED);
  }

  std::map<string, uint32_t> addn_new_input_names{{"x1", 0}, {"x2", 1}, {"x3", 2}};

  addn_node->GetOpDesc()->UpdateInputName(addn_new_input_names);
  if (mul_node->GetInDataAnchor(1 - lossscale_input_index)->GetPeerOutAnchor() != nullptr) {
    NodePtr input_node_mul = mul_node->GetInDataAnchor(1 - lossscale_input_index)->GetPeerOutAnchor()->GetOwnerNode();
    FUSION_PASS_CHECK(GraphUtils::AddEdge(mul_node->GetInDataAnchor(1 - lossscale_input_index)->GetPeerOutAnchor(),
                                          addn_node->GetInDataAnchor(mul_index)),
                      OP_LOGE(kFusedOpType.c_str(), "Add edge between node %s and node %s failed.",
                              input_node_mul->GetName().c_str(), addn_node->GetName().c_str()),
                      return FAILED);
  }

  FUSION_PASS_CHECK(graph.RemoveNode(mul_node),
                    OP_LOGE(kFusedOpType.c_str(), "Remove node %s failed.", mul_node->GetName().c_str()),
                    return FAILED);

  addn_node->GetOpDesc()->SetType("FusedMulAddNL2loss");
  std::vector<NodePtr> original_nodes;
  original_nodes.push_back(mul_node);
  original_nodes.push_back(addn_node);
  GraphPassUtil::RecordOriginalNames(original_nodes, addn_node);
  for (auto index_name : addn_node->GetOpDesc()->GetAllInputName()) {
    OP_LOGI(kFusedOpType.c_str(), "The addn_node's input name is :%s, input index is %u", index_name.first.c_str(),
            index_name.second);
  }

  OP_LOGI(kFusedOpType.c_str(), "MulAddNFusionPass graph fusion success!");

  OpDescPtr fusion_desc = AttrUtils::CopyOpDesc(addn_node->GetOpDesc());
  fusion_desc->AddOutputDesc("y2", l2loss_node->GetOpDesc()->GetOutputDesc(0));
  fusion_desc->SetName(addn_node->GetOpDesc()->GetName() + "/FusedMulAddNL2loss");
  NodePtr fusion_node = graph.AddNode(fusion_desc);
  OP_LOGI(kFusedOpType.c_str(), "Node[%s] has [%u] output desc.", fusion_node->GetName().c_str(),
          fusion_node->GetOpDesc()->GetAllOutputsDesc().size());
  // l2loss output link to fusion_node output
  if (out_data_anchor0->GetPeerInDataAnchors().size() > 0) {
    for (InDataAnchorPtr in_anchor_ptr : out_data_anchor0->GetPeerInDataAnchors()) {
      in_anchor_ptr->UnlinkAll();
      FUSION_PASS_CHECK(
          GraphUtils::AddEdge(fusion_node->GetOutDataAnchor(1), in_anchor_ptr) != SUCCESS,
          OP_LOGE(kFusedOpType.c_str(), "Add edge from node:%s's 2nd index to node:%s's 1st index failed.",
                  fusion_node->GetName().c_str(), l2loss_node->GetName().c_str()),
          return FAILED);
      OP_LOGI(kFusedOpType.c_str(), "Add edge from node:%s's 2nd index to node:%s's 1st index.",
              fusion_node->GetName().c_str(), l2loss_node->GetName().c_str());
    }
  }
  for (unsigned int i = 0; i < addn_node->GetAllInDataAnchors().size(); i++) {
    FUSION_PASS_CHECK(

        GraphUtils::AddEdge(addn_node->GetInDataAnchor(i)->GetPeerOutAnchor(), fusion_node->GetInDataAnchor(i)) !=
            SUCCESS,
        OP_LOGE(kFusedOpType.c_str(), "Add edge from fused node:%s's index[%u] to fusion node:%s's index[%u] failed.",
                addn_node->GetName().c_str(), i, fusion_node->GetName().c_str(), i),
        return FAILED);
    OP_LOGI(kFusedOpType.c_str(), "Add edge from fused node:%s's index[%u] to fusion node:%s's index[%u].",
            addn_node->GetName().c_str(), i, fusion_node->GetName().c_str(), i);
  }
  FUSION_PASS_CHECK(addn_node->GetInControlAnchor() == nullptr,
                    OP_LOGE(kFusedOpType.c_str(), "The GetInControlAnchor of addn_node is null, fusion failed."),
                    return PARAM_INVALID);
  for (unsigned int i = 0; i < addn_node->GetInControlAnchor()->GetPeerOutControlAnchors().size(); i++) {
    FUSION_PASS_CHECK(
        GraphUtils::AddEdge(addn_node->GetInControlAnchor()->GetPeerOutControlAnchors().at(i),
                            fusion_node->GetInControlAnchor()) != SUCCESS,
        OP_LOGE(kFusedOpType.c_str(),
                "Add edge from fused node:%s's control index[%u] to fusion node:%s's control index failed.",
                addn_node->GetName().c_str(), i, fusion_node->GetName().c_str()),
        return FAILED);
    OP_LOGI(kFusedOpType.c_str(), "Add edge from fused node:%s's control index[%u] to fusion node:%s's control index.",
            addn_node->GetName().c_str(), i, fusion_node->GetName().c_str());
  }
  if (addn_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    OP_LOGI(kFusedOpType.c_str(), "The size of addn_node is [%u].",
            addn_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size());
    for (InDataAnchorPtr in_anchor_ptr : addn_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      in_anchor_ptr->UnlinkAll();
      FUSION_PASS_CHECK(
          GraphUtils::AddEdge(fusion_node->GetOutDataAnchor(0), in_anchor_ptr) != SUCCESS,
          OP_LOGE(kFusedOpType.c_str(), "Add edge from fused node:%s's index to fusion node:%s's 1st index failed.",
                  addn_node->GetName().c_str(), fusion_node->GetName().c_str()),
          return FAILED);
      OP_LOGI(kFusedOpType.c_str(), "Add edge from fused node:%s's 1st index to fusion node:%s's 1st index.",
              addn_node->GetName().c_str(), fusion_node->GetName().c_str());
    }
  }
  if (addn_node->GetInControlAnchor() != nullptr) {
    addn_node->GetInControlAnchor()->UnlinkAll();
  }
  for (auto in_anchor : addn_node->GetAllInDataAnchors()) {
    if (in_anchor != nullptr) {
      in_anchor->UnlinkAll();
    }
  }
  FUSION_PASS_CHECK(graph.RemoveNode(addn_node) != GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "Remove addn_node node[%s] failed", addn_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(l2loss_node) != GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "Remove l2loss_node node[%s] failed", l2loss_node->GetName().c_str()),
                    return FAILED);
  OP_LOGI(kFusedOpType.c_str(), "MulAddNL2LossFusionPass graph fusion success!");
  return SUCCESS;
}

REGISTER_PASS("MulAddNL2LossFusionPass", BUILT_IN_GRAPH_PASS, MulAddNL2LossFusionPass);
}  // namespace fe
