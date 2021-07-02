/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "layernorm_training_fusion_pass.h"
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <memory>

#include "op_log.h"
#include "fp16_t.hpp"
#include "cce/dnn_base.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

namespace fe {
static const string ADD = "Add";
static const string MUL = "Mul";
static const string SUB = "Sub";
static const string RSQRT = "Rsqrt";
static const string MEAN = "ReduceMeanD";
static const string SQUAREDDIFFERENCE = "SquaredDifference";
static const string ADD_PATTERN_1 = "add1";
static const string ADD_PATTERN_2 = "add2";
static const string MUL_PATTERN_1 = "mul1";
static const string MUL_PATTERN_2 = "mul2";
static const string MUL_PATTERN_3 = "mul3";
static const string SUB_PATTERN = "sub";
static const string RSQRT_PATTERN = "rsqrt";
static const string REDUCEMEAND_PATTERN_1 = "reduce_mean_d_1";
static const string REDUCEMEAND_PATTERN_2 = "reduce_mean_d_2";
static const string SQUAREDDIFFERENCE_PATTERN = "squared_difference";
static const string LAYERNORM = "LayerNorm";

vector<FusionPattern*> LayerNormTrainingFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("LayerNormTrainingFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  pattern->AddOpDesc(REDUCEMEAND_PATTERN_1, {MEAN})
      .AddOpDesc(SQUAREDDIFFERENCE_PATTERN, {SQUAREDDIFFERENCE})
      .AddOpDesc(REDUCEMEAND_PATTERN_2, {MEAN})
      .AddOpDesc(ADD_PATTERN_1, {ADD})
      .AddOpDesc(RSQRT_PATTERN, {RSQRT})
      .AddOpDesc(MUL_PATTERN_1, {MUL})
      .AddOpDesc(MUL_PATTERN_2, {MUL})
      .AddOpDesc(MUL_PATTERN_3, {MUL})
      .AddOpDesc(SUB_PATTERN, {SUB})
      .AddOpDesc(ADD_PATTERN_2, {ADD})
      .SetInputs(SQUAREDDIFFERENCE_PATTERN, {REDUCEMEAND_PATTERN_1})
      .SetInputs(REDUCEMEAND_PATTERN_2, {SQUAREDDIFFERENCE_PATTERN})
      .SetInputs(ADD_PATTERN_1, {REDUCEMEAND_PATTERN_2})
      .SetInputs(RSQRT_PATTERN, {ADD_PATTERN_1})
      .SetInputs(MUL_PATTERN_1, {RSQRT_PATTERN})
      .SetInputs(MUL_PATTERN_2, {REDUCEMEAND_PATTERN_1, MUL_PATTERN_1})
      .SetInputs(SUB_PATTERN, {MUL_PATTERN_2})
      .SetInputs(MUL_PATTERN_3, {MUL_PATTERN_1})
      .SetInputs(ADD_PATTERN_2, {MUL_PATTERN_3, SUB_PATTERN})
      .SetOutput(ADD_PATTERN_2);
  patterns.push_back(pattern);
  return patterns;
}

Status LayerNormTrainingFusionPass::RemoveSmalleNodes(ge::ComputeGraph& graph, const ge::NodePtr& mean_node1,
                                                      const ge::NodePtr& square_difference_node,
                                                      const ge::NodePtr& mean_node2,
                                                      const ge::NodePtr& add_node1, const ge::NodePtr& rsqrt_node,
                                                      const ge::NodePtr& mul_node1, const ge::NodePtr& mul_node2,
                                                      const ge::NodePtr& mul_node3, const ge::NodePtr& sub_node,
                                                      const ge::NodePtr& add_node2) {
  FUSION_PASS_CHECK(graph.RemoveNode(mean_node1) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove node %s failed.", mean_node1->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(square_difference_node) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove node %s failed.",
                    square_difference_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mean_node2) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove node %s failed.", mean_node2->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(add_node1) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove node %s failed.", add_node1->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(rsqrt_node) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove node %s failed.", rsqrt_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mul_node1) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove node %s failed.", rsqrt_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mul_node2) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove node %s failed.", rsqrt_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mul_node3) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove node %s failed.", rsqrt_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(sub_node) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove node %s failed.", rsqrt_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(add_node2) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove node %s failed.", rsqrt_node->GetName().c_str()),
                    return FAILED);
  return SUCCESS;
}


Status LayerNormTrainingFusionPass::AddTensorDescForLn(const ge::OpDescPtr& ln_opdesc,
                                                       const ge::GeTensorDesc& x_tensor,
                                                       const ge::GeTensorDesc& gamma_tensor,
                                                       const ge::GeTensorDesc& beta_tensor,
                                                       const ge::GeTensorDesc& ln_out_tensor,
                                                       const ge::GeTensorDesc& mean_out_tensor,
                                                       const ge::GeTensorDesc& varience_out_tensor) {
  FUSION_PASS_CHECK(ln_opdesc->AddInputDesc(x_tensor) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add input0 of ln failed."), return FAILED);
  FUSION_PASS_CHECK(ln_opdesc->AddInputDesc("gamma", gamma_tensor) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add input1 of ln failed."), return FAILED);
  FUSION_PASS_CHECK(ln_opdesc->AddInputDesc("beta", beta_tensor) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add input2 of ln failed."), return FAILED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Size of input of LayerNorm is %u.", ln_opdesc->GetInputsSize());

  /* Output */
  FUSION_PASS_CHECK(ln_opdesc->AddOutputDesc("y", ln_out_tensor) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add output0 of bn failed."), return FAILED);
  FUSION_PASS_CHECK(ln_opdesc->AddOutputDesc("mean", mean_out_tensor) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add output1 of bn failed."), return FAILED);
  FUSION_PASS_CHECK(ln_opdesc->AddOutputDesc("variance", varience_out_tensor) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add output2 of bn failed."), return FAILED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Size of output of LayerNorm is %u.", ln_opdesc->GetOutputsSize());
  return SUCCESS;
}


Status LayerNormTrainingFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter graph fusion LayerNormTrainingFusionPass!");
  ge::NodePtr mean_node1 = GetNodeFromMapping(REDUCEMEAND_PATTERN_1, mapping);
  ge::NodePtr square_difference_node = GetNodeFromMapping(SQUAREDDIFFERENCE_PATTERN, mapping);
  ge::NodePtr mean_node2 = GetNodeFromMapping(REDUCEMEAND_PATTERN_2, mapping);
  ge::NodePtr add_node1 = GetNodeFromMapping(ADD_PATTERN_1, mapping);
  ge::NodePtr rsqrt_node = GetNodeFromMapping(RSQRT_PATTERN, mapping);
  ge::NodePtr mul_node1 = GetNodeFromMapping(MUL_PATTERN_1, mapping);
  ge::NodePtr mul_node2 = GetNodeFromMapping(MUL_PATTERN_2, mapping);
  ge::NodePtr mul_node3 = GetNodeFromMapping(MUL_PATTERN_3, mapping);
  ge::NodePtr sub_node = GetNodeFromMapping(SUB_PATTERN, mapping);
  ge::NodePtr add_node2 = GetNodeFromMapping(ADD_PATTERN_2, mapping);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Check node null or not");
  FUSION_PASS_CHECK(mean_node1 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mean_node1 is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(square_difference_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(),
                    "square_difference_node is null, fusion failed."), return PARAM_INVALID);
  FUSION_PASS_CHECK(mean_node2 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mean_node2 is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(add_node1 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "add_node1 is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(rsqrt_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "rsqrt_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mul_node1 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mul_node1 is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mul_node2 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mul_node2 is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mul_node3 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mul_node3 is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(sub_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "sub_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(add_node2 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "add_node2 is null, fusion failed."),
                    return PARAM_INVALID);
  // check and set attr
  ge::Operator op_mean0 = ge::OpDescUtils::CreateOperatorFromNode(mean_node1);
  std::vector<int64_t> axes0 = {-1};
  if (GRAPH_SUCCESS != op_mean0.GetAttr("axes", axes0)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "get attr axes failed.");
    return GRAPH_FAILED;
  }
  bool keep_dims0 = true;
  if (GRAPH_SUCCESS != op_mean0.GetAttr("keep_dims", keep_dims0)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "get attr keep_dims failed.");
    return GRAPH_FAILED;
  }
  Operator op_mean1 = ge::OpDescUtils::CreateOperatorFromNode(mean_node2);
  std::vector<int64_t> axes1 = {-1};
  if (GRAPH_SUCCESS != op_mean1.GetAttr("axes", axes1)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "get attr axes failed.");
    return GRAPH_FAILED;
  }
  bool keep_dims1 = true;
  if (GRAPH_SUCCESS != op_mean1.GetAttr("keep_dims", keep_dims1)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "get attr keep_dims failed.");
    return GRAPH_FAILED;
  }
  if ((axes0.size() != 1) || (axes1.size() != 1) || (axes0[0] != axes1[0])) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the axes of mean are not same, not change");
    return NOT_CHANGED;
  }
  if (!keep_dims0 || !keep_dims1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the attr keep_dims of mean is not true, not change");
    return NOT_CHANGED;
  }
  ge::GeTensorDesc input_desc = mean_node1->GetOpDesc()->GetInputDesc(0);
  std::vector<int64_t> input_dims = input_desc.GetShape().GetDims();
  size_t dims_size = input_dims.size();
  if (dims_size < 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "input shape must be greater to one, not change");
    return NOT_CHANGED;
  }
  if ((axes0[0] != -1) && (axes0[0] != (int64_t)(input_dims.size() - 1))) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the axes of mean is not the last dim of input, not change");
    return NOT_CHANGED;
  }

  /* 1. check nodes and get input tensor x, gamma, beta */
  /* check node squareddifference get input tensor (x) */
  FUSION_PASS_CHECK(square_difference_node->GetOpDesc()->GetInputsSize() < 2,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "%s input size is < 2", square_difference_node->GetName().c_str()),
                    return PARAM_INVALID);
  ge::OutDataAnchorPtr out_anchorof_x = nullptr;
  ge::GeTensorDesc x_tensor;
  ge::GeTensorDesc mean_out_tensor;
  if (square_difference_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode() != mean_node1) {
    x_tensor = square_difference_node->GetOpDesc()->GetInputDesc(0);
    out_anchorof_x = square_difference_node->GetInDataAnchor(0)->GetPeerOutAnchor();
    // output1
    mean_out_tensor = square_difference_node->GetOpDesc()->GetInputDesc(1);
    FUSION_PASS_CHECK(out_anchorof_x == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "out_anchorof_x must not be null"),
                      return PARAM_INVALID);
  } else {
    x_tensor = square_difference_node->GetOpDesc()->GetInputDesc(1);
    out_anchorof_x = square_difference_node->GetInDataAnchor(1)->GetPeerOutAnchor();
    // output1
    mean_out_tensor = square_difference_node->GetOpDesc()->GetInputDesc(0);
    FUSION_PASS_CHECK(out_anchorof_x == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "out_anchorof_x must not be null"),
                      return PARAM_INVALID);
  }
  /* check add node */
  FUSION_PASS_CHECK(add_node1->GetOpDesc()->GetInputsSize() < 2,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "%s input size is < 2", add_node1->GetName().c_str()),
                    return PARAM_INVALID);
  ge::OutDataAnchorPtr in_anchor0 = add_node1->GetInDataAnchor(0)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(in_anchor0 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "in_anchor0 must not be null"),
                    return PARAM_INVALID);
  ge::NodePtr input_node0 = in_anchor0->GetOwnerNode();
  FUSION_PASS_CHECK(input_node0 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "input_node0 must not be null"),
                    return PARAM_INVALID);
  /* output 2 */
  ge::GeTensorDesc varience_out_tensor = mean_node2->GetOpDesc()->GetOutputDesc(0);
  /* check mul_node1 and get gamma tensor */
  FUSION_PASS_CHECK(mul_node1->GetOpDesc()->GetInputsSize() < 2,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "%s input size is < 2", mul_node1->GetName().c_str()),
                    return PARAM_INVALID);
  ge::OutDataAnchorPtr in_anchorof_mul = nullptr;
  ge::GeTensorDesc gamma_tensor;
  if (mul_node1->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode() != rsqrt_node) {
    gamma_tensor = mul_node1->GetOpDesc()->GetInputDesc(1);
    in_anchorof_mul = mul_node1->GetInDataAnchor(1)->GetPeerOutAnchor();
    FUSION_PASS_CHECK(in_anchorof_mul == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "in_anchorof_mul must not be null"),
                      return PARAM_INVALID);
  } else {
    gamma_tensor = mul_node1->GetOpDesc()->GetInputDesc(0);
    in_anchorof_mul = mul_node1->GetInDataAnchor(0)->GetPeerOutAnchor();
    FUSION_PASS_CHECK(in_anchorof_mul == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "in_anchorof_mul must not be null"),
                      return PARAM_INVALID);
  }

  /* check sub node and get beta tensor */
  FUSION_PASS_CHECK(sub_node->GetOpDesc()->GetInputsSize() < 2,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "%s input size is < 2", mul_node1->GetName().c_str()),
                    return PARAM_INVALID);
  ge::OutDataAnchorPtr in_anchorof_sub = sub_node->GetInDataAnchor(0)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(in_anchorof_sub == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "in_anchorof_sub must not be null"),
                    return PARAM_INVALID);
  ge::NodePtr beta_node = in_anchorof_sub->GetOwnerNode();
  FUSION_PASS_CHECK(beta_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "beta_node must not be null"),
                    return PARAM_INVALID);
  ge::GeTensorDesc beta_tensor = sub_node->GetOpDesc()->GetInputDesc(0);

  /* 2. Get output node and edge */
  ge::OutDataAnchorPtr out_anchor_ln = add_node2->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(out_anchor_ln == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "out_anchor_ln must not be null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(out_anchor_ln->GetPeerInDataAnchors().empty(),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "size of peer in anchor of ln is 0"), return FAILED);
  ge::OutDataAnchorPtr out_anchor_mean = mean_node1->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(out_anchor_mean == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "out_anchor_mean must not be null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(out_anchor_mean->GetPeerInDataAnchors().empty(),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "size of peer in anchor of mean is 0"), return FAILED);
  ge::OutDataAnchorPtr out_anchor_rsqrt = rsqrt_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(out_anchor_rsqrt == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "out_anchor_rsqrt must not be null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(out_anchor_rsqrt->GetPeerInDataAnchors().empty(),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "size of peer in anchor of out_anchor_rsqrt is 0"), return FAILED);
  ge::OutDataAnchorPtr out_anchor_mul1 = mul_node1->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(out_anchor_mul1 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "out_anchor_mul1 must not be null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(out_anchor_mul1->GetPeerInDataAnchors().empty(),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "size of peer in anchor of out_anchor_mul1 is 0"), return FAILED);

  ge::GeTensorDesc ln_out_tensor = add_node2->GetOpDesc()->GetOutputDesc(0);

  /* 3. Add new Opdesc of LayerNorm*/
  std::shared_ptr<ge::OpDesc> ln_opdesc = nullptr;
  std::string ln_node_name = add_node2->GetName() + "_" + "layernorm";
  ln_opdesc = std::make_shared<ge::OpDesc>(ln_node_name, LAYERNORM);

  FUSION_PASS_CHECK(ln_opdesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "ln_opdesc is null, fusion failed."),
                    return PARAM_INVALID);

  /* 4. Add TensorDesc into new node. Ln must have three inputs and three
   * outputs. If not, we will return false. */
  (void)AddTensorDescForLn(ln_opdesc, x_tensor, gamma_tensor, beta_tensor, ln_out_tensor,
                           mean_out_tensor, varience_out_tensor);
  /* 5. Add Node into graph */
  ge::NodePtr ln_node = graph.AddNode(ln_opdesc);
  newNodes.push_back(ln_node);

  /* 6. remove old edges and add new edges */
  if (square_difference_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode() != mean_node1) {
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(out_anchorof_x, ln_node->GetInDataAnchor(0)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(out_anchorof_x,
                      square_difference_node->GetInDataAnchor(0)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
  } else {
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(out_anchorof_x, ln_node->GetInDataAnchor(0)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(out_anchorof_x,
                      square_difference_node->GetInDataAnchor(1)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
  }
  /* remove scale (gamma) from mul node 1 and Add it to layernorm */
  if (mul_node1->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode() != rsqrt_node) {
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(in_anchorof_mul, ln_node->GetInDataAnchor(1)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add scale(gamma) edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(in_anchorof_mul, mul_node1->GetInDataAnchor(1)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove scale(gamma) edge failed."), return FAILED);
  } else {
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(in_anchorof_mul, ln_node->GetInDataAnchor(1)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add scale(gamma) edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(in_anchorof_mul, mul_node1->GetInDataAnchor(0)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove scale(gamma) edge failed."), return FAILED);
  }
  /* remove offset (beta) from sub and Add it to layernorm */
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(in_anchorof_sub, ln_node->GetInDataAnchor(2)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add offset(beta) edge failed."), return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(in_anchorof_sub, sub_node->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove offset(beta) edge failed."), return FAILED);

  /* remove add_node2 output and add it to layernorm */
  size_t index = 0;
  for (auto& peerInAnchor : out_anchor_ln->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(peerInAnchor == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "peerInAnchor must not be null"),
                      return PARAM_INVALID);
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(out_anchor_ln, peerInAnchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove edge %u failed.", index), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(ln_node->GetOutDataAnchor(0), peerInAnchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge %u failed.", index), return FAILED);
    index++;
  }
  // remove mean's output and add it to layernorm
  size_t index_1 = 0;
  for (auto& peerInAnchor : out_anchor_mean->GetPeerInDataAnchors()) {
    if (peerInAnchor->GetOwnerNode()->GetName() != square_difference_node->GetName()) {
      FUSION_PASS_CHECK(peerInAnchor == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "peerInAnchor must not be null"),
                        return PARAM_INVALID);
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(out_anchor_mean, peerInAnchor) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove edge %u failed.", index_1), return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(ln_node->GetOutDataAnchor(1), peerInAnchor) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge %u failed.", index_1), return FAILED);
    }
    index_1++;
  }
  // remove rsqrt's output and add it to layernorm
  size_t index_2 = 0;
  for (auto& peerInAnchor : out_anchor_rsqrt->GetPeerInDataAnchors()) {
    if (peerInAnchor->GetOwnerNode()->GetName() != mul_node1->GetName()) {
      FUSION_PASS_CHECK(peerInAnchor == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "peerInAnchor must not be null"),
                        return PARAM_INVALID);
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(out_anchor_rsqrt, peerInAnchor) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove edge %u failed.", index_2), return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(ln_node->GetOutDataAnchor(2), peerInAnchor) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge %u failed.", index_2), return FAILED);
    }
    index_2++;
  }
  /* remove all small nodes */
  Status ret = RemoveSmalleNodes(graph, mean_node1, square_difference_node, mean_node2, add_node1,
                                 rsqrt_node, mul_node1, mul_node2, mul_node3, sub_node, add_node2);
  if (ret != SUCCESS) {
    return ret;
  }

  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(ln_opdesc, "begin_norm_axis", axes0[0]),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "node %s set begin_norm_axis attr not success.",
                    ln_opdesc->GetName().c_str()), return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(ln_opdesc, "begin_params_axis", -1),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "node %s set begin_params_axis attr not success.",
                    ln_opdesc->GetName().c_str()), return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetFloat(ln_opdesc, "epsilon", 1e-7),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "node %s set epsilon attr not success.",
                    ln_opdesc->GetName().c_str()), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "LayerNorm graph fusion success for node %s!", add_node2->GetName().c_str());
  return SUCCESS;
}
REGISTER_PASS("LayerNormTrainingFusionPass", BUILT_IN_GRAPH_PASS, LayerNormTrainingFusionPass);
}  // namespace fe
