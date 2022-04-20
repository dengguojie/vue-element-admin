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
 * \file softmax_grad_fusion_pass.cc
 */

#include "softmax_grad_fusion_pass.h"

#include <memory>
#include <string>
#include <vector>

#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph/utils/graph_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "error_util.h"

namespace fe {

static const string PATTERN_MUL0 = "mul0";
static const string PATTERN_REDUCESUMD = "reduce_sum_d";
static const string PATTERN_SUB = "sub";
static const string PATTERN_MUL1 = "mul1";

static const string MUL = "Mul";
static const string REDUCESUMD = "ReduceSumD";
static const string SUB = "Sub";
static const string SoftmaxGrad = "SoftmaxGrad";
static const string PATTERN_INPUT0 = "Input0";
static const string PATTERN_INPUT1 = "Input1";
static const uint8_t OUTPUT_NODE_NUM = 1;
static const uint8_t OUTPUT_NODE_NUM_2 = 2;
static const uint8_t PATTERN_1 = 1;
static const uint8_t PATTERN_2 = 2;

/**
 * @brief
 * strideslicegrad    transdata
 *        |       \  /       |
 *        |        mul       |
 *        |        |         |    strideslicegrad     transdata
 *        |    reducesumd    |              \         /
 *        |        |         |       ===>   softmaxgrad
 *        ---------sub       |                   |
 *                 |         |              biasaddgrad
 *                 mul--------
 *                 |
 *              biasaddgrad
 *
 *
 *  grad_x = softmax*(grad_softmax - sum(grad_softmax * softmax))
    pattern1
    input1 input0
       \   /
        mul                                    input0    input1
         |  \                                    \        /
         |  reducesumd  input0       ----->     softmax_grad
         |          \    /                            |
         |           mul1                           grad
         |------\    /
                 sub
                  |
                grad
 *  grad_x = softmax*(grad_softmax - sum(grad_softmax * softmax))
 */

std::vector<FusionPattern*> SoftmaxGradFusionPass::DefinePatterns() {
  std::vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("SoftmaxGradFusionPass");
  if (pattern == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "pattern is nullptr,Create pattern failed.");
    return patterns;
  }
  FusionPattern* pattern1 = new (std::nothrow) FusionPattern("SoftmaxGradFusionPass");
  if (pattern1 == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "pattern1 is nullptr,Create pattern failed.");
    return patterns;
  }
  pattern->AddOpDesc(PATTERN_MUL0, {MUL})
      .AddOpDesc(PATTERN_REDUCESUMD, {REDUCESUMD})
      .AddOpDesc(PATTERN_SUB, {SUB})
      .AddOpDesc(PATTERN_MUL1, {MUL})
      .SetInputs(PATTERN_REDUCESUMD, {PATTERN_MUL0})
      .SetInputs(PATTERN_SUB, {PATTERN_REDUCESUMD})
      .SetInputs(PATTERN_MUL1, {PATTERN_SUB})
      .SetOutput(PATTERN_MUL1);
  patterns.push_back(pattern);
  pattern1->AddOpDesc(PATTERN_MUL0, {MUL})
      .AddOpDesc(PATTERN_MUL1, {MUL})
      .AddOpDesc(PATTERN_REDUCESUMD, {REDUCESUMD})
      .AddOpDesc(PATTERN_SUB, {SUB})
      .AddOpDesc(PATTERN_INPUT0)
      .AddOpDesc(PATTERN_INPUT1)
      .SetInputs(PATTERN_MUL0, {PATTERN_INPUT1, PATTERN_INPUT0})
      .SetInputs(PATTERN_REDUCESUMD, {PATTERN_MUL0})
      .SetInputs(PATTERN_MUL1, {PATTERN_REDUCESUMD, PATTERN_INPUT0})
      .SetInputs(PATTERN_SUB, {PATTERN_MUL0, PATTERN_MUL1})
      .SetOutput(PATTERN_SUB);
  patterns.push_back(pattern1);
  return patterns;
}

bool SoftmaxGradFusionPass::IsMatch(ge::NodePtr mul_node0, ge::NodePtr reducesumd_node, ge::NodePtr sub_node,
                                    ge::NodePtr mul_node1) const {
  std::string mul0_name = mul_node0->GetName();
  std::string mul0_input0_name = mul_node0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName();
  std::string mul0_input1_name = mul_node0->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName();
  std::string sub_input0_name = sub_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName();
  std::string mul1_input0_name = mul_node1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName();
  std::string sum_input0_name = reducesumd_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName();

  uint8_t pattern = (mul0_name == sub_input0_name) ? PATTERN_2 : PATTERN_1;
  if ((pattern == PATTERN_1) && (mul0_input1_name != sub_input0_name)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "mul0_input0 are not same with sub_input0, not change");
    return false;
  }

  if (mul0_input0_name != mul1_input0_name) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "mul0_input1 and mul1_input1 are not same, not change");
    return false;
  }

  auto subOutputDataNodes = sub_node->GetOutDataNodes();
  auto mulOutputDataNodes = mul_node0->GetOutDataNodes();
  auto mul1OutputDataNodes = mul_node1->GetOutDataNodes();
  auto sumOutputDataNodes = reducesumd_node->GetOutDataNodes();
  if(sumOutputDataNodes.size() != OUTPUT_NODE_NUM) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "sum output is not 1, not change.");
    return false;
  }  
  if (pattern == PATTERN_1) {
    if((mulOutputDataNodes.size() != OUTPUT_NODE_NUM) || (mul1OutputDataNodes.size() != OUTPUT_NODE_NUM)) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "this pattern does not meet the fusion condition.");
      return false;
    }
  }
  if (pattern == PATTERN_2) {
    if((mulOutputDataNodes.size() != OUTPUT_NODE_NUM_2) || (mul1OutputDataNodes.size() != OUTPUT_NODE_NUM)) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "this pattern does not meet the fusion condition.");
      return false;
    }
    if ((mul0_name != sum_input0_name)) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "mul0_name and sum_input0_name are not same, not change");
      return false;
    }
  }

  return true;
}

Status SoftmaxGradFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, std::vector<ge::NodePtr>& fusionNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "SoftmaxGradFusionPass start");
  // get node
  ge::NodePtr mul_node0 = GetNodeFromMapping(PATTERN_MUL0, mapping);
  ge::NodePtr reducesumd_node = GetNodeFromMapping(PATTERN_REDUCESUMD, mapping);
  ge::NodePtr sub_node = GetNodeFromMapping(PATTERN_SUB, mapping);
  ge::NodePtr mul_node1 = GetNodeFromMapping(PATTERN_MUL1, mapping);

  FUSION_PASS_CHECK(mul_node0 == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "mul_node0 node is null"), return NOT_CHANGED);
  FUSION_PASS_CHECK(reducesumd_node == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "reducesumd_node node is null"),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(sub_node == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "sub_node node is null"), return NOT_CHANGED);
  FUSION_PASS_CHECK(mul_node1 == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "mul_node1 node is null"), return NOT_CHANGED);

  // check is match for fusion
  if (!IsMatch(mul_node0, reducesumd_node, sub_node, mul_node1)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "input don't match fusion pattern.");
    return NOT_CHANGED;
  }

  // create softmaxgrad node
  std::string softmaxgradNodeName = mul_node0->GetName() + "_" + "softmax_grad";
  ge::OpDescPtr softmaxgradOpdesc;
  FUSION_PASS_MAKE_SHARED((softmaxgradOpdesc = std::make_shared<ge::OpDesc>(softmaxgradNodeName, SoftmaxGrad)),
                          return NOT_CHANGED);
  FUSION_PASS_CHECK(softmaxgradOpdesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "softmaxgradOpdesc is null, fusion failed."),
                    return NOT_CHANGED);
  ge::GeTensorDesc gradsoftmaxInputDesc = mul_node0->GetOpDesc()->GetInputDesc(1);
  ge::GeTensorDesc softmaxInputDesc = mul_node0->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc gradxOutputDesc = mul_node1->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(softmaxgradOpdesc->AddInputDesc("grad_softmax", gradsoftmaxInputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add softmaxgrad input 0 desc failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(softmaxgradOpdesc->AddInputDesc("softmax", softmaxInputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add softmaxgrad input 1 desc failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(softmaxgradOpdesc->AddOutputDesc("grad_x", gradxOutputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add softmaxgrad output desc failed."),
                    return NOT_CHANGED);
  ge::NodePtr softmaxgradNode = graph.AddNode(softmaxgradOpdesc);
  softmaxgradNode->GetOpDesc()->SetType(SoftmaxGrad);
  fusionNodes.push_back(softmaxgradNode);

  // link anchor for softmaxgrad
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(mul_node0->GetInDataAnchor(0)->GetPeerOutAnchor(), softmaxgradNode->GetInDataAnchor(0)) !=
          SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between softmax and softmaxgradNode failed."),
      return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mul_node0->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            softmaxgradNode->GetInDataAnchor(1)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Add edge between strideslicegrad and softmaxgradNode failed."),
                    return FAILED);
  for (auto& inDataAnchor : mul_node1->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(mul_node1->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."),
                      return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(softmaxgradNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."),
                      return FAILED);
  }

  // remove node
  FUSION_PASS_CHECK(graph.RemoveNode(mul_node0) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mul_node0 failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(reducesumd_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove reducesumd_node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(sub_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove sub_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mul_node1) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mul_node1 failed."), return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "softmax_grad fusion success");

  return SUCCESS;
}

REGISTER_PASS("SoftmaxGradFusionPass", BUILT_IN_GRAPH_PASS, SoftmaxGradFusionPass);
}  // namespace fe
