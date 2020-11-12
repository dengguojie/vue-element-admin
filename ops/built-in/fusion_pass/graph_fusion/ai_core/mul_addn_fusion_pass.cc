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
 * \file mul_addn_fusion_pass.cpp
 * \brief mul addn(two input) fusion pass(mul --> addn)
 */
#include "mul_addn_fusion_pass.h"

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

namespace fe {
static const string kFusionNodePatternIdAddn = "AddN";
static const string kFusionNodePatternIdMul = "Mul";
static const std::string kStreamLabel = "_stream_label";

vector<FusionPattern*> MulAddNPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("MulAddNPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(kFusedOpType.c_str(), "New a pattern object failed."), return patterns);

  pattern->AddOpDesc(kFusionNodePatternIdAddn, {"AddN"})
      .AddOpDesc(kFusionNodePatternIdMul, {"Mul"})
      .SetInputs(kFusionNodePatternIdAddn, {kFusionNodePatternIdMul})
      .SetOutput(kFusionNodePatternIdAddn);
  patterns.push_back(pattern);

  return patterns;
}

Status MulAddNPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(kFusedOpType.c_str(), "Enter graph fusion MulAddNFusionPass!");
  ge::NodePtr addn_node = GetNodeFromMapping(kFusionNodePatternIdAddn, mapping);
  ge::NodePtr mul_node = GetNodeFromMapping(kFusionNodePatternIdMul, mapping);

  FUSION_PASS_CHECK(addn_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "The addn_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mul_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "The mul_node is null, fusion failed."),
                    return PARAM_INVALID);

  string addn_stream_label;
  string mul_stream_label;
  ge::AttrUtils::GetStr(addn_node->GetOpDesc(), kStreamLabel, addn_stream_label);
  ge::AttrUtils::GetStr(mul_node->GetOpDesc(), kStreamLabel, mul_stream_label);
  OP_LOGI(kFusedOpType.c_str(), "AddN node's stream label is %s", addn_stream_label.c_str());
  OP_LOGI(kFusedOpType.c_str(), "Mul node's stream label is %s", mul_stream_label.c_str());
  if (addn_stream_label != mul_stream_label) {
    OP_LOGI(kFusedOpType.c_str(), "Node [%s] and node [%s] don't hava same stream label.", addn_node->GetName().c_str(),
            mul_node->GetName().c_str());
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

  uint32_t lossscale_input_index{0};
  for (uint32_t index = 0; index < mul_node->GetAllInDataAnchors().size(); index++) {
    ge::NodePtr input_node_mul = mul_node->GetInDataAnchor(index)->GetPeerOutAnchor()->GetOwnerNode();
    if (ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(input_node_mul) == "Constant") {
      lossscale_input_index = index;
      break;
    }
  }

  // get mul Constant input
  ge::GeTensorDesc const_input = mul_node->GetOpDesc()->GetInputDesc(lossscale_input_index);
  vector<int64_t> dim_info = const_input.GetShape().GetDims();
  size_t dims_size = dim_info.size();
  FUSION_PASS_CHECK(
      !(dims_size == 0 || (dims_size == 1 && dim_info[0] == 1)),
      OP_LOGI(kFusedOpType.c_str(),
              "The const input of Mul node must be scalar or dims=(1,), but dims size is [%u] and dims[0] is [%lld]",
              dims_size, dim_info[0]),
      return NOT_CHANGED);

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
        FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(out_data_anchor, peer_indata_anchor),
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
          ge::GraphUtils::AddEdge(input_node_mul->GetOutControlAnchor(), mul_out_control_node->GetInControlAnchor()),
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
    ge::NodePtr input_node_mul = mul_node->GetInDataAnchor(lossscale_input_index)->GetPeerOutAnchor()->GetOwnerNode();
    // In order to prevent the lossscale nodes from being deleted when the mul
    // node is deleted, you need to unassociate them first.
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(mul_node->GetInDataAnchor(lossscale_input_index)->GetPeerOutAnchor(),
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
    ge::NodePtr input_node_mul =
        mul_node->GetInDataAnchor(1 - lossscale_input_index)->GetPeerOutAnchor()->GetOwnerNode();
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mul_node->GetInDataAnchor(1 - lossscale_input_index)->GetPeerOutAnchor(),
                                              addn_node->GetInDataAnchor(mul_index)),
                      OP_LOGE(kFusedOpType.c_str(), "Add edge between node %s and node %s failed.",
                              input_node_mul->GetName().c_str(), addn_node->GetName().c_str()),
                      return FAILED);
  }

  FUSION_PASS_CHECK(graph.RemoveNode(mul_node),
                    OP_LOGE(kFusedOpType.c_str(), "Remove node %s failed.", mul_node->GetName().c_str()),
                    return FAILED);

  addn_node->GetOpDesc()->SetType("FusedMulAddN");
  for (auto index_name : addn_node->GetOpDesc()->GetAllInputName()) {
    OP_LOGI(kFusedOpType.c_str(), "The addn_node's input name is :%s, input index is %u", index_name.first.c_str(),
            index_name.second);
  }

  // Record fusion nodes
  fusion_nodes.push_back(addn_node);

  OP_LOGI(kFusedOpType.c_str(), "MulAddNFusionPass graph fusion success!");
  return SUCCESS;
}
REGISTER_PASS("MulAddNPass", BUILT_IN_GRAPH_PASS, MulAddNPass);
}  // namespace fe
