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
 * \file confusion_mul_grad_fusion_pass.cpp
 * \brief confusion_mul_grad fusion pass( --> confusion_mul_grad)
 */
#include "confusion_mul_grad_fusion_pass.h"

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace std;
using namespace ge;

namespace fe {
static const char* REDUCESUMD = "ReduceSumD";
static const char* MUL = "Mul";
static const string PATTERN_MUL = "Mul";
static const string PATTERN_MUL1 = "Mul1";
static const string PATTERN_REDUCESUMD = "ReduceSumD";
static const string AXIS = "axes";
static const string KEEPDIMS = "keep_dims";
static const string CONFUSIONMULGRAD = "ConfusionMulGrad";

/*         (0)input1 input2  input3(0)
              |      /\       /
              |     /  \     /
              |    /    \   /
 ctr edge--- mul /      mul1  ---ctr edge
              |          |
              |          |
              output    reducesumD---ctr edge
                         |
                         |
                       output1
*/

vector<FusionPattern*> MulGradFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define ConfusionMulGrad pattern begin");
  vector<FusionPattern*> patterns;

  // tf confusion_matrix subgraph fused to tbe confusion_matrix
  FusionPattern* pattern = new (std::nothrow) FusionPattern("MulGradFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_REDUCESUMD, {REDUCESUMD})
      .AddOpDesc(PATTERN_MUL1, {MUL})
      .SetInputs(PATTERN_REDUCESUMD, {PATTERN_MUL1})
      .SetOutput(PATTERN_REDUCESUMD);

  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define MulGradFusionPass pattern end");
  return patterns;
}

Status MulGradFusionPass::CheckPeerMul1InDataAnchors(const ge::OutDataAnchorPtr& outputAnchor,
                                                     const size_t& expectedNum) {
  FUSION_PASS_CHECK(outputAnchor == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "outputAnchor must not be null"),
                    return PARAM_INVALID);
  if (outputAnchor->GetPeerInDataAnchors().size() == expectedNum) {
    return SUCCESS;
  }
  return FAILED;
}

Status MulGradFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define MulGradFusionPass fusion begin");

  ge::NodePtr mul1_node = GetNodeFromMapping(PATTERN_MUL1, mapping);
  ge::NodePtr sum_node = GetNodeFromMapping(PATTERN_REDUCESUMD, mapping);

  FUSION_PASS_CHECK(sum_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "sum node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mul1_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mul1 node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(
      CheckPeerMul1InDataAnchors(mul1_node->GetOutDataAnchor(0), 1) != SUCCESS,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "%s contains more than one peer input", mul1_node->GetName().c_str()),
      return NOT_CHANGED);

  if (mul1_node->GetName().find("bert/encoder") == string::npos &&
      mul1_node->GetName().find("loss/gradMul") == string::npos &&
      mul1_node->GetName().find("loss-BertPretrainingLoss/gradMul") == string::npos) {
    return NOT_CHANGED;
  }

  ge::NodePtr input2_node = mul1_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();
  FUSION_PASS_CHECK(input2_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "Mul node is null."), return NOT_CHANGED);

  // get another mul node
  OP_LOGD(FUSED_OP_TYPE.c_str(), "mul1 node name [%s], mul1Node type[%s].", mul1_node->GetName().c_str(),
          mul1_node->GetType().c_str());
  auto input2_node_out_nodes = input2_node->GetOutAllNodes();
  ge::NodePtr mul_node = nullptr;
  for (auto input2_node_out_node : input2_node_out_nodes) {
    if (input2_node_out_node->GetName() == mul1_node->GetName()) {
      continue;
    }

    ge::NodePtr mul_out_node = input2_node_out_node->GetOutDataAnchor(0)->GetFirstPeerAnchor()->GetOwnerNode();

    if (input2_node_out_node->GetType() == MUL) {
      mul_node = input2_node_out_node;
      OP_LOGD(FUSED_OP_TYPE.c_str(), "find mul node sucess, node name is [%s]'s ,type is [%s]",
              mul_node->GetName().c_str(), mul_node->GetType().c_str());
      break;
    }
  }
  FUSION_PASS_CHECK(
      mul_node == nullptr,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "input2_node node[%s] don't have mul out node.", input2_node->GetName().c_str()),
      return NOT_CHANGED);

  // define attrs of input edge based on orignal info
  ge::GeTensorDesc input_desc1 = mul_node->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc input_desc2 = mul_node->GetOpDesc()->GetInputDesc(1);
  ge::GeTensorDesc input_desc3 = mul1_node->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc output_desc1 = mul_node->GetOpDesc()->GetOutputDesc(0);
  ge::GeTensorDesc output_desc2 = sum_node->GetOpDesc()->GetOutputDesc(0);

  // define the attr confusion_mul_grad_node of fused op node
  ge::OpDescPtr confusion_mul_grad_op;
  FUSION_PASS_MAKE_SHARED((confusion_mul_grad_op = std::make_shared<ge::OpDesc>(
                               mul1_node->GetName() + "/" + CONFUSIONMULGRAD, CONFUSIONMULGRAD)),
                          return INTERNAL_ERROR);
  confusion_mul_grad_op->AddInputDesc("input0", input_desc1);
  confusion_mul_grad_op->AddInputDesc("input1", input_desc2);
  confusion_mul_grad_op->AddInputDesc("input2", input_desc3);
  confusion_mul_grad_op->AddOutputDesc("output0", output_desc1);
  confusion_mul_grad_op->AddOutputDesc("output1", output_desc2);
  ge::NodePtr confusion_mul_grad_node = graph.AddNode(confusion_mul_grad_op);
  FUSION_PASS_CHECK(
      confusion_mul_grad_node == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "confusion_mul_grad_node is null, fusion failed."),
      return NOT_CHANGED);
  newNodes.push_back(confusion_mul_grad_node);

  ge::OutDataAnchorPtr new_in_anchor_ptr0 = mul_node->GetInDataAnchor(0)->GetPeerOutAnchor();
  ge::OutDataAnchorPtr new_in_anchor_ptr1 = mul_node->GetInDataAnchor(1)->GetPeerOutAnchor();
  ge::OutDataAnchorPtr new_in_anchor_ptr2 = mul1_node->GetInDataAnchor(0)->GetPeerOutAnchor();

  ge::GraphUtils::AddEdge(new_in_anchor_ptr0, confusion_mul_grad_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(new_in_anchor_ptr1, confusion_mul_grad_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(new_in_anchor_ptr2, confusion_mul_grad_node->GetInDataAnchor(2));

  for (auto in_data_anchor : mul_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(mul_node->GetOutDataAnchor(0), in_data_anchor),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mul node out data edge failed."),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(confusion_mul_grad_node->GetOutDataAnchor(0), in_data_anchor),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add mul node out data edge failed."),
                      return FAILED);
  }

  for (auto in_data_anchor : sum_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(sum_node->GetOutDataAnchor(0), in_data_anchor),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove sum node out data edge failed."),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(confusion_mul_grad_node->GetOutDataAnchor(1), in_data_anchor),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add sum node out data edge failed."),
                      return FAILED);
  }

  for (auto in_data_anchor : mul1_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(mul1_node->GetOutDataAnchor(0), in_data_anchor),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mul1 node out data edge failed."),
                      return FAILED);
  }

  // set attrs num_classes and dtype of confusion_matrix node
  std::vector<int64_t> axis;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(sum_node->GetOpDesc(), AXIS, axis),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get axis attr failed"), return FAILED);

  bool keep_dims;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetBool(sum_node->GetOpDesc(), KEEPDIMS, keep_dims),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get keep_dims attr failed."), return FAILED);

  ge::OpDescPtr confusion_mul_grad_desc = confusion_mul_grad_node->GetOpDesc();
  FUSION_PASS_CHECK(!ge::AttrUtils::SetBool(confusion_mul_grad_desc, KEEPDIMS, keep_dims),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Set keep_dims attr failed"), return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(confusion_mul_grad_desc, AXIS, axis),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Set axis attr failed"), return FAILED);

  if (mul_node->GetInControlAnchor()) {
    for (auto out_control_anchor : mul_node->GetInControlAnchor()->GetPeerOutControlAnchors()) {
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(out_control_anchor, confusion_mul_grad_node->GetInControlAnchor()) != SUCCESS,
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add mul node input control edge failed."),
          return FAILED);
      FUSION_PASS_CHECK(
          ge::GraphUtils::RemoveEdge(out_control_anchor, mul_node->GetInControlAnchor()) != SUCCESS,
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mul node input control edge failed."),
          return FAILED);
    }
  }

  if (mul1_node->GetInControlAnchor()) {
    for (auto out_control_anchor : mul1_node->GetInControlAnchor()->GetPeerOutControlAnchors()) {
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(out_control_anchor, confusion_mul_grad_node->GetInControlAnchor()) != SUCCESS,
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add mul1 node control edge failed."), return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(out_control_anchor, mul1_node->GetInControlAnchor()) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mul1 node control edge failed."),
                        return FAILED);
    }
  }

  if (sum_node->GetOutControlAnchor()) {
    for (auto in_control_anchor : sum_node->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(confusion_mul_grad_node->GetOutControlAnchor(), in_control_anchor) != SUCCESS,
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add sum node out control edge failed."),
          return FAILED);
      FUSION_PASS_CHECK(
          ge::GraphUtils::RemoveEdge(sum_node->GetOutControlAnchor(), in_control_anchor) != SUCCESS,
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove sum node out control edge failed."),
          return FAILED);
    }
  }

  // remove input node and const node in subgraph
  FUSION_PASS_CHECK(graph.RemoveNode(mul_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mul node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(sum_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove sum node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mul1_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mul1 node failed."), return FAILED);

  return SUCCESS;
}

REGISTER_PASS("MulGradFusionPass", BUILT_IN_GRAPH_PASS, MulGradFusionPass);
}  // namespace fe
