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
 * \file keras_momentum_lossscale_fusion_pass.cpp
 * \brief keras_momentum lossscale fusion pass(mul --> keras_momentum)
 */
#include "keras_momentum_lossscale_fusion_pass.h"

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
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

/*
 * before:
 * lossscale(reciprocal)(type: Var or const)
 *             \
 *              \Mul--->ApplyKerasMomentum
 *              /
 *             /
 *        grad
 *
 * after:
 * lossscale(reciprocal)
 *             \
 *              \ApplyKerasMomentum
 *              /
 *             /
 *        grad
 *
 */

namespace fe {
static const string kkerasMomentumLossscalePatternIdMomentum = "ApplyKerasMomentum";
static const string kkMomentumLossscalePatternIdMul = "Mul";
static const std::string kkStreamLabel = "_stream_label";

vector<FusionPattern*> KerasMomentumLossscaleFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("KerasMomentumLossscaleFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(kkFusedOpType.c_str(), "New a pattern object failed."), return patterns);

  pattern->AddOpDesc(kkerasMomentumLossscalePatternIdMomentum, {"ApplyKerasMomentum"})
      .AddOpDesc(kkMomentumLossscalePatternIdMul, {"Mul"})
      .SetInputs(kkerasMomentumLossscalePatternIdMomentum, {kkMomentumLossscalePatternIdMul})
      .SetOutput(kkerasMomentumLossscalePatternIdMomentum);
  patterns.push_back(pattern);

  return patterns;
}

Status KerasMomentumLossscaleFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                           vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(kkFusedOpType.c_str(), "Enter graph fusion KerasMomentumLossscaleFusionPass!");
  ge::NodePtr momentum_node = GetNodeFromMapping(kkerasMomentumLossscalePatternIdMomentum, mapping);
  ge::NodePtr mul_node = GetNodeFromMapping(kkMomentumLossscalePatternIdMul, mapping);

  FUSION_PASS_CHECK(momentum_node == nullptr,
                    OP_LOGE(kkFusedOpType.c_str(), "The momentum_node is null, fusion failed."), return PARAM_INVALID);
  FUSION_PASS_CHECK(mul_node == nullptr, OP_LOGE(kkFusedOpType.c_str(), "The mul_node is null, fusion failed."),
                    return PARAM_INVALID);

  string momentum_stream_label;
  string mul_stream_label;
  ge::AttrUtils::GetStr(momentum_node->GetOpDesc(), kkStreamLabel, momentum_stream_label);
  ge::AttrUtils::GetStr(mul_node->GetOpDesc(), kkStreamLabel, mul_stream_label);
  OP_LOGI(kkFusedOpType.c_str(), "Momentum node's stream label is %s", momentum_stream_label.c_str());
  OP_LOGI(kkFusedOpType.c_str(), "Mul node's stream label is %s", mul_stream_label.c_str());
  FUSION_PASS_CHECK(momentum_stream_label != mul_stream_label,
                    OP_LOGW(kkFusedOpType.c_str(), "Node [%s] and node [%s] don't hava same stream label.",
                            momentum_node->GetName().c_str(), mul_node->GetName().c_str()),
                    return NOT_CHANGED);
  OP_LOGI(kkFusedOpType.c_str(), "Momentum and Mul have same _stream_label attr.");

  FUSION_PASS_CHECK(momentum_node->GetType() != "ApplyKerasMomentum",
                    OP_LOGW(kkFusedOpType.c_str(), "Get the momentum_node type is [%s], which is not apply momentum.",
                            momentum_node->GetType().c_str()),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(mul_node->GetOutDataNodes().size() != 1,
                    OP_LOGW(kkFusedOpType.c_str(), "Output node of Mul node size is [%d], which not equal to 1.",
                            mul_node->GetOutDataNodes().size()),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(mul_node->GetInDataNodes().size() != 2,
                    OP_LOGW(kkFusedOpType.c_str(), "Input node of Mul node size is [%d], which not equal to 2.",
                            mul_node->GetInDataNodes().size()),
                    return NOT_CHANGED);

  uint32_t lossscale_input_index{0};
  for (uint32_t index = 0; index < mul_node->GetAllInDataAnchors().size(); index++) {
    auto in_data_anchor_iter = mul_node->GetInDataAnchor(index);
    FUSION_PASS_CHECK(in_data_anchor_iter == nullptr,
                      OP_LOGE(kkFusedOpType.c_str(), "The in_data_anchor_iter is null, fusion failed."),
                      return PARAM_INVALID);
    auto peer_out_anchor_iter = in_data_anchor_iter->GetPeerOutAnchor();
    FUSION_PASS_CHECK(peer_out_anchor_iter == nullptr,
                      OP_LOGE(kkFusedOpType.c_str(), "The peer_out_anchor_iter is null, fusion failed."),
                      return PARAM_INVALID);
    ge::NodePtr input_node_mul = peer_out_anchor_iter->GetOwnerNode();
    if (ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(input_node_mul) == "Constant") {
      lossscale_input_index = index;
      break;
    }
  }
  for (unsigned int i = 0; i < mul_node->GetInControlAnchor()->GetPeerOutControlAnchors().size(); i++) {
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(mul_node->GetInControlAnchor()->GetPeerOutControlAnchors().at(i),
                                           momentum_node->GetInControlAnchor()) != GRAPH_SUCCESS,
        OP_LOGE(kkFusedOpType.c_str(),
                "Add edge from fused node:%s's control index[%u] to fusion node:%s's control index failed.",
                mul_node->GetName().c_str(), i, momentum_node->GetName().c_str()),
        return FAILED);
    OP_LOGD(kkFusedOpType.c_str(), "Add edge from fused node:%s's control index[%u] to fusion node:%s's control index.",
            mul_node->GetName().c_str(), i, momentum_node->GetName().c_str());
  }
  // get mul Constant input
  // mul_node->GetOpDesc() would not be nullptr there
  ge::GeTensorDesc const_input = mul_node->GetOpDesc()->GetInputDesc(lossscale_input_index);
  vector<int64_t> dim_info = const_input.GetShape().GetDims();
  int dims_size = dim_info.size();
  FUSION_PASS_CHECK(
      !(dims_size == 0 || (dims_size == 1 && dim_info[0] == 1)),
      OP_LOGW(kkFusedOpType.c_str(),
              "The const input of Mul node must be scalar or dims=(1,), but dims size is [%u] and dims[0] not 1",
              dims_size),
      return NOT_CHANGED);

  int grad_index{0};
  bool find_apply_momentum{false};
  auto out_dataanchor_list = mul_node->GetAllOutDataAnchors();
  for (auto out_data_anchor : out_dataanchor_list) {
    for (auto peer_indata_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      if (peer_indata_anchor->GetOwnerNode()->GetType() == "ApplyKerasMomentum") {
        grad_index = peer_indata_anchor->GetIdx();
        if (grad_index != 3) {
          OP_LOGI(kkFusedOpType.c_str(),
                  "The input mul node %s of ResourceApplyKerasMomentum is not lossscale node, do not need fusion.",
                  mul_node->GetName().c_str());
          continue;
        }
        FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(out_data_anchor, peer_indata_anchor) != GRAPH_SUCCESS,
                          OP_LOGE(kkFusedOpType.c_str(), "Remove edge failed."), return FAILED);
        find_apply_momentum = true;
      }
    }
  }

  FUSION_PASS_CHECK(!find_apply_momentum,
                    OP_LOGW(kkFusedOpType.c_str(), "The output node of Mul node is not ResourceApplyKerasMomentum."),
                    return NOT_CHANGED);

  // add control anchor between input node of Mul and out node of Mul.
  for (auto input_node_mul : mul_node->GetInAllNodes()) {
    for (auto mul_out_control_node : mul_node->GetOutControlNodes()) {
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(input_node_mul->GetOutControlAnchor(),
                                                mul_out_control_node->GetInControlAnchor()) != GRAPH_SUCCESS,
                        OP_LOGE(kkFusedOpType.c_str(), "Add edge between node %s. and node %s failed.",
                                input_node_mul->GetName().c_str(), mul_out_control_node->GetName().c_str()),
                        return FAILED);
    }
  }
  // momentum_node->GetOpDesc() would not be nullptr there
  std::map<string, uint32_t> momentum_input_names = momentum_node->GetOpDesc()->GetAllInputName();
  for (auto index_name : momentum_input_names) {
    OP_LOGI(kkFusedOpType.c_str(), "The momentum_node's input name is :%s, input index is %u", index_name.first.c_str(),
            index_name.second);
  }
  // grad is the first input of mul, the index is 0, the second input of the
  // lossscale is mul, and the corresponding index is 1.
  auto mul_node_in_data_auchor = mul_node->GetInDataAnchor(lossscale_input_index);
  FUSION_PASS_CHECK(mul_node_in_data_auchor == nullptr,
                    OP_LOGE(kkFusedOpType.c_str(), "The mul_node_in_data_auchor is null, fusion failed."),
                    return PARAM_INVALID);
  if (mul_node_in_data_auchor->GetPeerOutAnchor() != nullptr) {
    ge::NodePtr input_node_mul = mul_node_in_data_auchor->GetPeerOutAnchor()->GetOwnerNode();
    // In order to prevent the lossscale nodes from being deleted when the mul
    // node is deleted, you need to unassociate them first.
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(mul_node_in_data_auchor->GetPeerOutAnchor(),
                                                 mul_node_in_data_auchor) != GRAPH_SUCCESS,
                      OP_LOGE(kkFusedOpType.c_str(), "Remove edge failed."), return FAILED);
    FUSION_PASS_CHECK(momentum_node->AddLinkFrom(input_node_mul) != GRAPH_SUCCESS,
                      OP_LOGE(kkFusedOpType.c_str(), "Add link between node %s. and node %s failed.",
                              input_node_mul->GetName().c_str(), momentum_node->GetName().c_str()),
                      return FAILED);
  }
  map<string, uint32_t>::iterator iter = momentum_input_names.find("grad");
  if (iter != momentum_input_names.end()) {
    momentum_input_names.erase(iter);
  }
  momentum_input_names["x1"] = 3;
  momentum_input_names["x2"] = 5;
  // UpdateInputName do not need check result
  momentum_node->GetOpDesc()->UpdateInputName(momentum_input_names);
  mul_node_in_data_auchor = mul_node->GetInDataAnchor(1 - lossscale_input_index);
  FUSION_PASS_CHECK(mul_node_in_data_auchor == nullptr,
                    OP_LOGE(kkFusedOpType.c_str(), "The mul_node_in_data_auchor is null, fusion failed."),
                    return PARAM_INVALID);
  if (mul_node_in_data_auchor->GetPeerOutAnchor() != nullptr) {
    ge::NodePtr input_node_mul = mul_node_in_data_auchor->GetPeerOutAnchor()->GetOwnerNode();
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mul_node_in_data_auchor->GetPeerOutAnchor(),
                                              momentum_node->GetInDataAnchor(grad_index)) != GRAPH_SUCCESS,
                      OP_LOGE(kkFusedOpType.c_str(), "Add edge between node %s and node %s failed.",
                              input_node_mul->GetName().c_str(), momentum_node->GetName().c_str()),
                      return FAILED);
  }

  FUSION_PASS_CHECK(graph.RemoveNode(mul_node) != GRAPH_SUCCESS,
                    OP_LOGE(kkFusedOpType.c_str(), "Remove node %s failed.", mul_node->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(!ge::AttrUtils::SetBool(momentum_node->GetOpDesc(), ge::APPLYMENTUM_ATTR_IS_GRAPH_FUSION, true),
                    OP_LOGE(kkFusedOpType.c_str(), "Set backward mode attr failed"), return FAILED);
  momentum_node->GetOpDesc()->SetType("FusedMulApplyKerasMomentum");
  for (auto index_name : momentum_node->GetOpDesc()->GetAllInputName()) {
    OP_LOGI(kkFusedOpType.c_str(), "The momentum_node's input name is :%s, input index is %u", index_name.first.c_str(),
            index_name.second);
  }
  fusion_nodes.push_back(momentum_node);
  auto doFusion = std::getenv("SWITCH_FUSION");
  int doFusionFlag = (doFusion != nullptr) ? std::strtol(doFusion, nullptr, 10) : 0;
  if (doFusionFlag) {
    OP_LOGI(kkFusedOpType.c_str(), "KerasMomentumLossscaleFusionPass 5 inputs to 6 inputs begin!");

    // get 0th input node of momentum
    auto momentum_node_in_data_auchor = momentum_node->GetInDataAnchor(0);
    FUSION_PASS_CHECK(momentum_node_in_data_auchor == nullptr,
                      OP_LOGE(kkFusedOpType.c_str(), "The momentum_node_in_data_auchor is null, fusion failed."),
                      return PARAM_INVALID);
    auto momentum_node_first_peer_anchor = momentum_node_in_data_auchor->GetFirstPeerAnchor();
    FUSION_PASS_CHECK(momentum_node_first_peer_anchor == nullptr,
                      OP_LOGE(kkFusedOpType.c_str(), "The momentum_node_first_peer_anchor is null, fusion failed."),
                      return PARAM_INVALID);
    auto var_node = momentum_node_first_peer_anchor->GetOwnerNode();
    auto var_desc = var_node->GetOpDesc();

    // create copy_var_node, update dtype from fp32 to fp16, set attr
    auto copy_desc_ptr = ge::AttrUtils::CopyOpDesc(var_desc);
    FUSION_PASS_CHECK(copy_desc_ptr == nullptr,
                      OP_LOGE(kkFusedOpType.c_str(), "Node:copy_var_node's OpDesc is null, fusion failed."),
                      return FAILED);
    auto tmp_desc = copy_desc_ptr->GetOutputDesc(0);
    tmp_desc.SetOriginDataType(ge::DT_FLOAT16);
    tmp_desc.SetDataType(ge::DT_FLOAT16);
    FUSION_PASS_CHECK(
        copy_desc_ptr->UpdateOutputDesc(0, tmp_desc) != GRAPH_SUCCESS,
        OP_LOGE(kkFusedOpType.c_str(), "Fail to update output desc[0] of op[%s].", copy_desc_ptr->GetName().c_str()),
        return FAILED);
    copy_desc_ptr->SetName(var_desc->GetName() + "_copy");
    FUSION_PASS_CHECK(!ge::AttrUtils::SetStr(copy_desc_ptr, "_copy_from_var_node", var_desc->GetName()),
                      OP_LOGE(kkFusedOpType.c_str(), "SetStr _copy_from_var_node failed"), return FAILED);

    FUSION_PASS_CHECK(!ge::AttrUtils::SetStr(copy_desc_ptr, "_src_var_name", var_desc->GetName()),
                      OP_LOGE(kkFusedOpType.c_str(), "SetStr _src_var_name failed"), return FAILED);
    FUSION_PASS_CHECK(!ge::AttrUtils::SetBool(copy_desc_ptr, "_copy_value", false),
                      OP_LOGE(kkFusedOpType.c_str(), "SetBool _copy_value failed"), return FAILED);
    ge::NodePtr copy_var_node = graph.AddNode(copy_desc_ptr);
    FUSION_PASS_CHECK(copy_var_node == nullptr,
                      OP_LOGE(kkFusedOpType.c_str(), "The copy_var_node is null, add node failed."), return FAILED);
    FUSION_PASS_CHECK(momentum_node->AddLinkFrom(copy_var_node) != GRAPH_SUCCESS,
                      OP_LOGE(kkFusedOpType.c_str(), "Fail to add link between op[%s] and op[%s] failed.",
                              momentum_node->GetName().c_str(), copy_var_node->GetName().c_str()),
                      return FAILED);

    // create copy_ref_node,
    auto ref_copy_desc_ptr = ge::AttrUtils::CopyOpDesc(copy_var_node->GetOpDesc());
    FUSION_PASS_CHECK(ref_copy_desc_ptr == nullptr,
                      OP_LOGE(kkFusedOpType.c_str(), "The ref_copy_desc_ptr's opDesc is null, fusion failed."),
                      return FAILED);
    auto ref_desc_tmp = copy_var_node->GetOpDesc()->GetOutputDesc(0);
    FUSION_PASS_CHECK(ref_copy_desc_ptr->UpdateInputDesc(0, ref_desc_tmp) != GRAPH_SUCCESS,
                      OP_LOGE(kkFusedOpType.c_str(), "UpdateInputDesc failed."), return FAILED);
    ref_copy_desc_ptr->SetName("Ref_" + copy_var_node->GetOpDesc()->GetName());
    // copy_var_node->GetOpDesc() would not be nullptr there
    FUSION_PASS_CHECK(
        !ge::AttrUtils::SetStr(ref_copy_desc_ptr, "ref_var_src_var_name", copy_var_node->GetOpDesc()->GetName()),
        OP_LOGE(kkFusedOpType.c_str(), "SetStr ref_var_src_var_name failed."), return FAILED);
    ge::NodePtr copy_ref_node = graph.AddNode(ref_copy_desc_ptr);
    FUSION_PASS_CHECK(copy_ref_node == nullptr,
                      OP_LOGE(kkFusedOpType.c_str(), "The copy_ref_node is null, add node failed."), return FAILED);

    // create new momentum node, add 1th output for newmomentum node;
    auto redy_op_desc = copy_ref_node->GetOpDesc();
    FUSION_PASS_CHECK(redy_op_desc == nullptr, OP_LOGE(kkFusedOpType.c_str(), "The redy_op_desc is null"),
                      return FAILED);
    auto redy_desc = redy_op_desc->GetInputDesc(0);
    auto old_desc = momentum_node->GetOpDesc();
    FUSION_PASS_CHECK(old_desc == nullptr,
                      OP_LOGE(kkFusedOpType.c_str(), "The momentum_node's opDesc is null, fusion failed."),
                      return FAILED);
    ge::OpDescPtr new_desc = ge::AttrUtils::CopyOpDesc(old_desc);
    FUSION_PASS_CHECK(new_desc == nullptr,
                      OP_LOGE(kkFusedOpType.c_str(), "New moment node's opDesc is null, fusion failed."), return FAILED);
    new_desc->SetType("FusedMulApplyMomentumExtern");
    FUSION_PASS_CHECK(new_desc->AddOutputDesc("var_copy", redy_desc) != GRAPH_SUCCESS,
                      OP_LOGE(kkFusedOpType.c_str(), "AddOutputDesc failed."), return FAILED);
    ge::NodePtr new_node = graph.AddNode(new_desc);
    FUSION_PASS_CHECK(new_node == nullptr, OP_LOGE(kkFusedOpType.c_str(), "The new_node is null, add node failed."),
                      return FAILED);

    // add in edges
    for (uint32_t i = 0; i < momentum_node->GetAllInDataAnchors().size(); i++) {
      auto momentum_node_in_data_anchor = momentum_node->GetInDataAnchor(i);
      FUSION_PASS_CHECK(momentum_node_in_data_anchor == nullptr,
                        OP_LOGE(kkFusedOpType.c_str(), "The momentum_node_in_data_anchor is null"), return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(momentum_node_in_data_anchor->GetPeerOutAnchor(),
                                                new_node->GetInDataAnchor(i)) != GRAPH_SUCCESS,
                        OP_LOGE(kkFusedOpType.c_str(), "Fail to add input edge for new momentum_node."), return FAILED);
    }

    // add out edges
    auto momentum_node_out_data_anchor = momentum_node->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(momentum_node_out_data_anchor == nullptr,
                      OP_LOGE(kkFusedOpType.c_str(), "The momentum_node_out_data_anchor is null"), return FAILED);
    for (auto in_anchor : momentum_node_out_data_anchor->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(in_anchor == nullptr, OP_LOGE(kkFusedOpType.c_str(), "The in_anchor is null, add node failed."),
                        return FAILED);
      in_anchor->UnlinkAll();
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(new_node->GetOutDataAnchor(0), in_anchor) != GRAPH_SUCCESS,
                        OP_LOGE(kkFusedOpType.c_str(), "Fail to add output[0] edge for new momentum_node."),
                        return FAILED);
    }

    // update 1th output of new momentum
    auto new_node_desc = new_node->GetOpDesc();
    FUSION_PASS_CHECK(new_node_desc == nullptr, OP_LOGE(kkFusedOpType.c_str(), "The new_node_desc is null."),
                      return FAILED);
    FUSION_PASS_CHECK(new_node_desc->UpdateOutputDesc(1, redy_desc) != GRAPH_SUCCESS,
                      OP_LOGE(kkFusedOpType.c_str(), "UpdateOutputDesc failed"), return FAILED);
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(new_node->GetOutDataAnchor(1), copy_ref_node->GetInDataAnchor(0)) != GRAPH_SUCCESS,
        OP_LOGE(kkFusedOpType.c_str(), "Fail to add output[1] edge for new momentum_node."), return FAILED);

    // deal in_control edge
    for (unsigned int i = 0; i < momentum_node->GetInControlAnchor()->GetPeerOutControlAnchors().size(); i++) {
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(momentum_node->GetInControlAnchor()->GetPeerOutControlAnchors().at(i),
                                  new_node->GetInControlAnchor()) != GRAPH_SUCCESS,
          OP_LOGE(kkFusedOpType.c_str(),
                  "Add in_edge from fused node:%s's control index[%u] to fusion node:%s's control index failed.",
                  momentum_node->GetName().c_str(), i, new_node->GetName().c_str()),
          return FAILED);
    }

    // deal out_control edge
    for (unsigned int i = 0; i < momentum_node->GetOutControlAnchor()->GetPeerInControlAnchors().size(); i++) {
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(new_node->GetOutControlAnchor(),
                                  momentum_node->GetOutControlAnchor()->GetPeerInControlAnchors().at(i)) !=
              GRAPH_SUCCESS,
          OP_LOGE(kkFusedOpType.c_str(),
                  "Add out_edge from fused node:%s's control index[%u] to fusion node:%s's control index failed.",
                  momentum_node->GetName().c_str(), i, new_node->GetName().c_str()),
          return FAILED);
    }

    // add dst edges for copy_var_node
    auto var_outanchor = var_node->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(var_outanchor == nullptr, OP_LOGE(kkFusedOpType.c_str(), "The var_outanchor is null."),
                      return FAILED);
    for (auto in_anchor : var_outanchor->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(in_anchor == nullptr, OP_LOGE(kkFusedOpType.c_str(), "The in_anchor is null."), return FAILED);
      if (in_anchor->GetOwnerNode()->GetOpDesc()->GetType() == "Conv2D" ||
          in_anchor->GetOwnerNode()->GetOpDesc()->GetType() == "Conv2DBackpropInputD" ||
          in_anchor->GetOwnerNode()->GetOpDesc()->GetType() == "MatMul") {
        in_anchor->UnlinkAll();
        FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(copy_var_node->GetOutDataAnchor(0), in_anchor) != GRAPH_SUCCESS,
                          OP_LOGE(kkFusedOpType.c_str(), "Fail to add output edge for copy_var_node."), return FAILED);
      }
    }

    // delete old data_edges
    for (auto in_anchor : momentum_node->GetAllInDataAnchors()) {
      if (in_anchor != nullptr) {
        in_anchor->UnlinkAll();
      }
    }

    // delete old in_control_edges
    if (momentum_node->GetInControlAnchor() != nullptr) {
      momentum_node->GetInControlAnchor()->UnlinkAll();
    }

    // delete old out_control_edges
    if (momentum_node->GetOutControlAnchor() != nullptr) {
      momentum_node->GetOutControlAnchor()->UnlinkAll();
    }

    FUSION_PASS_CHECK(graph.RemoveNode(momentum_node) != GRAPH_SUCCESS,
                      OP_LOGE(kkFusedOpType.c_str(), "Remove node %s failed.", momentum_node->GetName().c_str()),
                      return FAILED);
    OP_LOGI(kkFusedOpType.c_str(), "Copy refVar success!");
    // Record fusion nodes
    fusion_nodes.pop_back();
    fusion_nodes.push_back(new_node);
  }

  OP_LOGI(kkFusedOpType.c_str(), "KerasMomentumLossscaleFusionPass graph fusion success!");
  return SUCCESS;
}

REGISTER_PASS("A_KerasMomentumLossscaleFusionPass", BUILT_IN_GRAPH_PASS, KerasMomentumLossscaleFusionPass);
}  // namespace fe
