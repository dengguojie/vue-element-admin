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
 * \file gelu_fusion_pass.cpp
 * \brief gelu usion pass
 * gelu(x) = 0.5*x*(1.0+tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))))
 * case1:                           | case2:
 *        x                         |              x
 *     /  |  \                      |           /  |   \
 *    |   |   Pow0                  |          |   |   Pow0
 *    |   |    |                    |          |   |    |
 *    |   |   Mul0        x         |          |   |   Mul0          x
 *    |    \   /          |         |          |    \   /            |
 *    |     Add0    ==>  Gelu       |          |     Add0    ==>    Gelu
 *    |      |            |         |          |      |              |
 *    |     Mul1          y         |          |     Mul1            y
 *    |      |                      |          |      |
 *    |     Tanh0                   |          |     Tanh0
 *    |      |                      |          |      |
 *    |     Add1                   const(0.5)*Mul2   Add1
 *    |      |                      |          |      |
 *    |     Mul2*const(0.5)         |          |      |
 *    \      /                      |          \     /
 *      Mul3                        |            Mul3
 *        |                         |              |
 *        y                         |              y
 */
#include "gelu_fusion_pass.h"

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
static const char* POW = "Pow";
static const char* ADD = "Add";
static const char* MUL = "Mul";
static const char* TANH = "Tanh";
static const std::string PATTERN_POW0 = "FusedNodePow0";
static const std::string PATTERN_MUL0 = "FusedNodeMul0";
static const std::string PATTERN_ADD0 = "FusedNodeAdd0";
static const std::string PATTERN_MUL1 = "FusedNodeMul1";
static const std::string PATTERN_TANH0 = "FusedNodeTanh0";
static const std::string PATTERN_ADD1 = "FusedNodeAdd1";
static const std::string PATTERN_MUL2 = "FusedNodeMul2";
static const std::string PATTERN_MUL3 = "FusedNodeMul3";
static const std::string PATTERN_INPUT = "Input0";
static const std::string GELU = "Gelu";

vector<FusionPattern*> GeluFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern1 = new (std::nothrow) FusionPattern("GeluFusionPass");
  FUSION_PASS_CHECK(pattern1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern1->AddOpDesc(PATTERN_POW0, {POW})
      .AddOpDesc(PATTERN_MUL0, {MUL})
      .AddOpDesc(PATTERN_ADD0, {ADD})
      .AddOpDesc(PATTERN_MUL1, {MUL})
      .AddOpDesc(PATTERN_TANH0, {TANH})
      .AddOpDesc(PATTERN_ADD1, {ADD})
      .AddOpDesc(PATTERN_MUL2, {MUL})
      .AddOpDesc(PATTERN_MUL3, {MUL})
      .AddOpDesc(PATTERN_INPUT)
      .SetInputs(PATTERN_POW0, {PATTERN_INPUT})
      .SetInputs(PATTERN_MUL0, {PATTERN_POW0})
      .SetInputs(PATTERN_ADD0, {PATTERN_INPUT, PATTERN_MUL0})
      .SetInputs(PATTERN_MUL1, {PATTERN_ADD0})
      .SetInputs(PATTERN_TANH0, {PATTERN_MUL1})
      .SetInputs(PATTERN_ADD1, {PATTERN_TANH0})
      .SetInputs(PATTERN_MUL2, {PATTERN_ADD1})
      .SetInputs(PATTERN_MUL3, {PATTERN_INPUT, PATTERN_MUL2})
      .SetOutput(PATTERN_MUL3);
  patterns.push_back(pattern1);

  FusionPattern* pattern2 = new (std::nothrow) FusionPattern("GeluFusionPass");
  FUSION_PASS_CHECK(pattern2 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern2->AddOpDesc(PATTERN_POW0, {POW})
      .AddOpDesc(PATTERN_MUL0, {MUL})
      .AddOpDesc(PATTERN_ADD0, {ADD})
      .AddOpDesc(PATTERN_MUL1, {MUL})
      .AddOpDesc(PATTERN_TANH0, {TANH})
      .AddOpDesc(PATTERN_ADD1, {ADD})
      .AddOpDesc(PATTERN_MUL2, {MUL})
      .AddOpDesc(PATTERN_MUL3, {MUL})
      .AddOpDesc(PATTERN_INPUT)
      .SetInputs(PATTERN_POW0, {PATTERN_INPUT})
      .SetInputs(PATTERN_MUL0, {PATTERN_POW0})
      .SetInputs(PATTERN_ADD0, {PATTERN_INPUT, PATTERN_MUL0})
      .SetInputs(PATTERN_MUL1, {PATTERN_ADD0})
      .SetInputs(PATTERN_TANH0, {PATTERN_MUL1})
      .SetInputs(PATTERN_ADD1, {PATTERN_TANH0})
      .SetInputs(PATTERN_MUL2, {PATTERN_INPUT})
      .SetInputs(PATTERN_MUL3, {PATTERN_MUL2, PATTERN_ADD1})
      .SetOutput(PATTERN_MUL3);
  patterns.push_back(pattern2);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define GeluFusionPass pattern end");
  return patterns;
}

Status GeluFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  // get all nodes
  ge::NodePtr pow0_node = GetNodeFromMapping(PATTERN_POW0, mapping);
  ge::NodePtr mul0_node = GetNodeFromMapping(PATTERN_MUL0, mapping);
  ge::NodePtr add0_node = GetNodeFromMapping(PATTERN_ADD0, mapping);
  ge::NodePtr mul1_node = GetNodeFromMapping(PATTERN_MUL1, mapping);
  ge::NodePtr tanh0_node = GetNodeFromMapping(PATTERN_TANH0, mapping);
  ge::NodePtr add1_node = GetNodeFromMapping(PATTERN_ADD1, mapping);
  ge::NodePtr mul2_node = GetNodeFromMapping(PATTERN_MUL2, mapping);
  ge::NodePtr mul3_node = GetNodeFromMapping(PATTERN_MUL3, mapping);
  FUSION_PASS_CHECK(pow0_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "pow0_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mul0_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mul0_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(add0_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add0_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mul1_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mul1_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(tanh0_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "tanh0_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(add1_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add1_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mul2_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mul2_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mul3_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mul3_node is null, fusion failed."),
                    return PARAM_INVALID);

  // check input and output link relation
  FUSION_PASS_CHECK(pow0_node->GetInDataNodes().size() != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Input node of Pow0 node size is [%lu], which not equal to 2.",
                            pow0_node->GetInDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(pow0_node->GetOutDataNodes().size() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Output node of Pow0 node size is [%d], which not equal to 1.",
                            pow0_node->GetOutDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(mul0_node->GetInDataNodes().size() != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Input node of Mul0 node size is [%lu], which not equal to 2.",
                            mul0_node->GetInDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(mul0_node->GetOutDataNodes().size() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Output node of Mul0 node size is [%d], which not equal to 1.",
                            mul0_node->GetOutDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(add0_node->GetInDataNodes().size() != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Input node of Add0 node size is [%lu], which not equal to 2.",
                            add0_node->GetInDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(add0_node->GetOutDataNodes().size() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Output node of Add0 node size is [%d], which not equal to 1.",
                            add0_node->GetOutDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(mul1_node->GetInDataNodes().size() != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Input node of Mul1 node size is [%lu], which not equal to 2.",
                            mul1_node->GetInDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(mul1_node->GetOutDataNodes().size() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Output node of Mul1 node size is [%d], which not equal to 1.",
                            mul1_node->GetOutDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(tanh0_node->GetInDataNodes().size() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Input node of Tanh0 node size is [%lu], which not equal to 2.",
                            tanh0_node->GetInDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(tanh0_node->GetOutDataNodes().size() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Output node of Tanh0 node size is [%d], which not equal to 1.",
                            tanh0_node->GetOutDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(add1_node->GetInDataNodes().size() != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Input node of Add1 node size is [%lu], which not equal to 2.",
                            add1_node->GetInDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(add1_node->GetOutDataNodes().size() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Output node of Add1 node size is [%d], which not equal to 1.",
                            add1_node->GetOutDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(mul2_node->GetInDataNodes().size() != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Input node of Mul2 node size is [%lu], which not equal to 2.",
                            mul2_node->GetInDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(mul2_node->GetOutDataNodes().size() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Output node of Mul2 node size is [%d], which not equal to 1.",
                            mul2_node->GetOutDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(mul3_node->GetInDataNodes().size() != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Input node of Mul3 node size is [%lu], which not equal to 2.",
                            mul3_node->GetInDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(mul3_node->GetOutDataNodes().size() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Output node of Mul3 node size is [%d], which not equal to 1.",
                            mul3_node->GetOutDataNodes().size()),
                    return NOT_CHANGED);

  // copy Opdesc
  std::shared_ptr<ge::OpDesc> gelu_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((gelu_desc = std::make_shared<ge::OpDesc>(pow0_node->GetName() + "/" + GELU, GELU)),
                          return FAILED);
  FUSION_PASS_CHECK(gelu_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "gelu_desc is null, fusion failed."),
                    return PARAM_INVALID);

  // add input
  ge::GeTensorDesc input_desc = pow0_node->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(gelu_desc->AddInputDesc(input_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input failed."), return FAILED);

  // add output
  ge::GeTensorDesc output_desc = mul3_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(gelu_desc->AddOutputDesc(output_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output failed."), return FAILED);

  // add gelu node
  ge::NodePtr gelu_node = graph.AddNode(gelu_desc);
  fusionNodes.push_back(gelu_node);

  // connect input edge
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(pow0_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            gelu_node->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(
                        FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                        pow0_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                        gelu_node->GetName().c_str()),
                    return FAILED);

  // connect output edge
  for (auto& inDataAnchor : mul3_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(mul3_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."),
                      return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(gelu_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."),
                      return FAILED);
  }

  // set node type
  gelu_node->GetOpDesc()->SetType(GELU);

  std::map<string, uint32_t> input_name_id = {{"x", 0}};
  gelu_node->GetOpDesc()->UpdateInputName(input_name_id);
  std::map<string, uint32_t> output_name_id = {{"y", 0}};
  gelu_node->GetOpDesc()->UpdateOutputName(output_name_id);

  // delete fused nodes
  FUSION_PASS_CHECK(graph.RemoveNode(pow0_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove pow0_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mul0_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mul0_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(add0_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove add0_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mul1_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mul1_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(tanh0_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove tanh0_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(add1_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove add1_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mul2_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mul2_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mul3_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mul3_node failed."), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "GeluFusionPass graph fusion success!");
  return SUCCESS;
}
REGISTER_PASS("GeluFusionPass", BUILT_IN_GRAPH_PASS, GeluFusionPass);
}  // namespace fe
