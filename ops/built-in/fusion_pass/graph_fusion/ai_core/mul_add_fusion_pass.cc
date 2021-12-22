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
 * \file mul_add_fusion_pass.cpp
 * \brief mul add fusion pass(mul --> add)
 */
#include "mul_add_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

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

static const string FUSION_NODE_PATTERN_ID_ADD = "FusedNodeAdd";
static const string FUSION_NODE_PATTERN_ID_MUL = "FusedNodeMul";
static const char* ADD = "Add";
static const char* MUL = "Mul";
static const std::string PATTERN_INPUTS = "input";
static const std::string STREAM_LABEL = "_stream_label";

vector<FusionPattern*> MulAddFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("MulAddFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(FUSION_NODE_PATTERN_ID_ADD, {ADD})
      .AddOpDesc(FUSION_NODE_PATTERN_ID_MUL, {MUL})
      .SetInputs(FUSION_NODE_PATTERN_ID_ADD, {FUSION_NODE_PATTERN_ID_MUL})
      .SetOutput(FUSION_NODE_PATTERN_ID_ADD);

  FusionPattern* pattern_ext = new (std::nothrow) FusionPattern("MulAddFusionPass");
  FUSION_PASS_CHECK(pattern_ext == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "new a pattern object failed."),
                    return patterns);

  pattern_ext->AddOpDesc(FUSION_NODE_PATTERN_ID_ADD, {ADD})
      .AddOpDesc(FUSION_NODE_PATTERN_ID_MUL, {MUL})
      .AddOpDesc(PATTERN_INPUTS)
      .SetInputs(FUSION_NODE_PATTERN_ID_ADD, {PATTERN_INPUTS, FUSION_NODE_PATTERN_ID_MUL})
      .SetOutput(FUSION_NODE_PATTERN_ID_ADD);

  patterns.push_back(pattern);
  patterns.push_back(pattern_ext);
  return patterns;
}

Status MulAddFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr add_node = GetNodeFromMapping(FUSION_NODE_PATTERN_ID_ADD, mapping);
  ge::NodePtr mul_node = GetNodeFromMapping(FUSION_NODE_PATTERN_ID_MUL, mapping);

  FUSION_PASS_CHECK(add_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "add_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mul_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "mul_node is null, fusion failed."),
                    return PARAM_INVALID);

  // exclude lamb and adam
  std::string mul_name = mul_node->GetName();
  if (string::npos != mul_name.find("adam") || string::npos != mul_name.find("lamb") ||
      string::npos == mul_name.find("bert")) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "fusedmuladd fusion pass only used in bert network.");
    return NOT_CHANGED;
  }

  ge::OpDescPtr mul_desc = mul_node->GetOpDesc();
  ge::OpDescPtr add_desc = add_node->GetOpDesc();
  FUSION_PASS_CHECK(mul_desc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "mul_desc is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(add_desc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "add_desc is null, fusion failed."),
                    return PARAM_INVALID);

  // check input output and link relation
  FUSION_PASS_CHECK(mul_node->GetInDataNodes().size() != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Input node of Mul node size is [%lu], which not equal to 2.",
                            mul_node->GetInDataNodes().size()),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(mul_node->GetOutDataNodes().size() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Output node of Mul node size is [%d], which not equal to 1.",
                            mul_node->GetOutDataNodes().size()),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(
      mul_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() != 1,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "The Mul node should only have 1 output anchor that link to other nodes."),
      return NOT_CHANGED);

  FUSION_PASS_CHECK(mul_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetType() != ADD,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "The Mul node should only have 1 output anchor that link to Add"),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(add_node->GetInDataNodes().size() != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Input node of Add size is [%lu], which not equal to 2.",
                            add_node->GetInDataNodes().size()),
                    return NOT_CHANGED);

  // check stream label
  string add_stream_label;
  string mul_stream_label;
  ge::AttrUtils::GetStr(add_node->GetOpDesc(), STREAM_LABEL, add_stream_label);
  ge::AttrUtils::GetStr(mul_node->GetOpDesc(), STREAM_LABEL, mul_stream_label);
  if (add_stream_label != mul_stream_label) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Node [%s] and node [%s] don't hava same stream label.", add_node->GetName().c_str(),
            mul_node->GetName().c_str());
    return NOT_CHANGED;
  }

  //change mul type for add find the other side
  mul_desc->SetType("MulBeforeAdd");
  // link changed
  // add_node second input link to mul_node third input
  uint32_t add_input_index = 1;

  if (add_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetType() == "MulBeforeAdd") {
    add_input_index = 0;
  }

  mul_desc->SetType("FusedMulAdd");

  ge::NodePtr input_node_add = add_node->GetInDataAnchor(add_input_index)->GetPeerOutAnchor()->GetOwnerNode();

  if (input_node_add->GetAllOutDataAnchors().size() != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "add pre-node has %lu output, do not fused",
            input_node_add->GetAllOutDataAnchors().size());
    mul_desc->SetType("Mul");
    return NOT_CHANGED;
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "add pre-node has only 1 output, will fused it");
  FUSION_PASS_CHECK(mul_node->AddLinkFrom(input_node_add),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add add's input to mul failed."),
                                                   return FAILED);

  // set name and output desc
  mul_desc->SetName(mul_desc->GetName() + "/FusedMulAdd");
  mul_desc->UpdateOutputDesc(0, add_desc->GetOutputDesc(0));

  // add_node output link to mul_node output
  if (add_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    for (ge::InDataAnchorPtr &inAnchorPtr : add_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(mul_node->GetOutDataAnchor(0), inAnchorPtr),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                         "Add edge from fused node:%s's index to fusion node:%s's 1st index failed.",
                  add_node->GetName().c_str(), add_node->GetName().c_str()),
          return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's 1st index to fusion node:%s's 1st index.",
              add_node->GetName().c_str(), add_node->GetName().c_str());
    }
  }
  add_node->GetOutDataAnchor(0)->UnlinkAll();
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(add_node),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove node:[%s] failed.",
                                                   add_desc->GetName().c_str()),
                    return FAILED);

  std::map<string, uint32_t> input_name_id = {{"x1", 0}, {"x2", 1}, {"x3", 2}};
  mul_node->GetOpDesc()->UpdateInputName(input_name_id);

  // Record fusion nodes
  fusionNodes.push_back(mul_node);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "MulAddFusionPass graph fusion success!");
  return SUCCESS;
}
REGISTER_PASS("MulAddFusionPass", BUILT_IN_GRAPH_PASS, MulAddFusionPass);
}  // namespace fe
