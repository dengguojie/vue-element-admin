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

/*!
 * \file reduce_mean_variance_fusion_pass.cpp
 * \brief reduce_mean_variance fusion pass
 */
#include "reduce_mean_variance_fusion_pass.h"

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
#include "fp16_t.hpp"

using namespace ge;
namespace fe {
static const std::string MEAN = "ReduceMeanD";
static const std::string SQUAREDDIFFERENCE = "SquaredDifference";
static const std::string PATTERN_MEAN0 = "FusedNodeMean0";
static const std::string PATTERN_SQUAREDDIFFERENCE0 = "FusedNodeSquaredDifference0";
static const std::string PATTERN_MEAN1 = "FusedNodeMean1";

static const std::string REDUCEMEANVARIANCE = "ReduceMeanVariance";

vector<FusionPattern*> ReduceMeanVarianceFusionPass::DefinePatterns() {
  /*
               input0                                    input0
              /       \                                    |
           mean_d     mean_d                      reduce_mean_variance
             |          |                             |            |
             |   squared_difference    ----->         | -- mul     |
             |          |                             |        \   |
             |          |                             |           add
             |          |                             |            |
         output_0    output_1                      output_0     output_1
  */
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ReduceMeanVarianceFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, 
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_MEAN0, {MEAN})
      .AddOpDesc(PATTERN_SQUAREDDIFFERENCE0, {SQUAREDDIFFERENCE})
      .AddOpDesc(PATTERN_MEAN1, {MEAN})
      .SetInputs(PATTERN_SQUAREDDIFFERENCE0, {PATTERN_MEAN0})
      .SetInputs(PATTERN_MEAN1, {PATTERN_SQUAREDDIFFERENCE0})
      .SetOutput(PATTERN_MEAN1);

  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define ReduceMeanVarianceFusionPass pattern end");
  return patterns;
}

Status ReduceMeanVarianceFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                            vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter reduceMeanVarianceFusionPass graph fusion process");
  // get all nodes
  ge::NodePtr mean0_node = GetNodeFromMapping(PATTERN_MEAN0, mapping);
  ge::NodePtr squared0_node = GetNodeFromMapping(PATTERN_SQUAREDDIFFERENCE0, mapping);
  ge::NodePtr mean1_node = GetNodeFromMapping(PATTERN_MEAN1, mapping);

  FUSION_PASS_CHECK(mean0_node == nullptr, 
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mean0_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(squared0_node == nullptr, 
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "squared0_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mean1_node == nullptr, 
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mean1_node is null, fusion failed."),
                    return PARAM_INVALID);

  // check input link
  std::string mean0_input_name = mean0_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName();
  std::string squared0_input0_name = squared0_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName();
  std::string squared0_input1_name = squared0_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName();
  
  if ((strcmp(mean0_input_name.c_str(), squared0_input0_name.c_str()) != 0) &&
      (strcmp(mean0_input_name.c_str(), squared0_input1_name.c_str()) != 0)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "mean0_input and squared0_input are not same, not change");
    return NOT_CHANGED;
  }

  // check output link
  FUSION_PASS_CHECK(squared0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "squared0_node output size is [%d], which not equal to 1.",
                            squared0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
    return NOT_CHANGED);

  // check input and attr
  ge::Operator op_mean0 = ge::OpDescUtils::CreateOperatorFromNode(mean0_node);
  std::vector<int64_t> axes0 = {0};
  if (GRAPH_SUCCESS != op_mean0.GetAttr("axes", axes0)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get attr axes failed.");
    return GRAPH_FAILED;
  }
  bool keep_dims0 = false;
  if (GRAPH_SUCCESS != op_mean0.GetAttr("keep_dims", keep_dims0)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get attr keep_dims failed.");
    return GRAPH_FAILED;
  }
  Operator op_mean1 = ge::OpDescUtils::CreateOperatorFromNode(mean1_node);
  std::vector<int64_t> axes1 = {0};
  if (GRAPH_SUCCESS != op_mean1.GetAttr("axes", axes1)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get attr axes failed.");
    return GRAPH_FAILED;
  }
  bool keep_dims1 = false;
  if (GRAPH_SUCCESS != op_mean0.GetAttr("keep_dims", keep_dims1)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get attr keep_dims failed.");
    return GRAPH_FAILED;
  }
  if (axes0.size() != axes1.size()) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the axes of mean are not same, not change");
    return NOT_CHANGED;
  }
  if (!keep_dims0 || !keep_dims1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the attr keep_dims of mean is not true, not change");
    return NOT_CHANGED;
  }

  // copy Opdesc
  std::shared_ptr<ge::OpDesc> mean_var_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((mean_var_desc = std::make_shared<ge::OpDesc>(mean0_node->GetName() +
                          "/" + REDUCEMEANVARIANCE, REDUCEMEANVARIANCE)),
                          return FAILED);
  FUSION_PASS_CHECK(mean_var_desc == nullptr, 
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mean_var_desc is null, fusion failed."),
                    return PARAM_INVALID);

  // copy Opdesc
  std::shared_ptr<ge::OpDesc> mul_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((mul_desc = std::make_shared<ge::OpDesc>(mean0_node->GetName() + "/" + "Mul", "Mul")),
                          return FAILED);
  FUSION_PASS_CHECK(mul_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mul_desc is null, fusion failed."),
                    return PARAM_INVALID);

  // copy Opdesc
  std::shared_ptr<ge::OpDesc> sub_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((sub_desc = std::make_shared<ge::OpDesc>(mean0_node->GetName() + "/" + "Sub", "Sub")),
                          return FAILED);
  FUSION_PASS_CHECK(sub_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "sub_desc is null, fusion failed."),
                    return PARAM_INVALID);

  // add input for mean_var_desc
  ge::GeTensorDesc input_desc = mean0_node->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(mean_var_desc->AddInputDesc(0, input_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input1_desc failed."),
                    return FAILED);

  // add output for mean_var_desc
  ge::GeTensorDesc output1_desc = mean0_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(mean_var_desc->AddOutputDesc("mean", output1_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output1_desc failed."),
                    return FAILED);
  ge::GeTensorDesc output2_desc = mean1_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(mean_var_desc->AddOutputDesc("variance", output2_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output2_desc failed."),
                    return FAILED);

  // add input for mul_desc
  ge::GeTensorDesc mul_input1_desc = mean0_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(mul_desc->AddInputDesc(0, mul_input1_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add mul_input1_desc failed."),
                    return FAILED);
  FUSION_PASS_CHECK(mul_desc->AddInputDesc(1, mul_input1_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add mul_input2_desc failed."),
                    return FAILED);

  // add output for mul_desc
  ge::GeTensorDesc mul_output1_desc = mean0_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(mul_desc->AddOutputDesc("y", mul_output1_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add mul_output1_desc failed."),
                    return FAILED);

  // add input for sub_desc
  ge::GeTensorDesc sub_input1_desc = mean0_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(sub_desc->AddInputDesc(0, sub_input1_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add sub_input1_desc failed."),
                    return FAILED);

  FUSION_PASS_CHECK(sub_desc->AddInputDesc(1, sub_input1_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add sub_input2_desc failed."),
                    return FAILED);

  // add output for sub_desc
  ge::GeTensorDesc sub_output1_desc = mean0_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(sub_desc->AddOutputDesc("y", sub_output1_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add sub_output1_desc failed."),
                    return FAILED);

  // add mean_var_norm node
  ge::NodePtr mean_var_node = graph.AddNode(mean_var_desc);
  fusionNodes.push_back(mean_var_node);
  ge::NodePtr mul_node = graph.AddNode(mul_desc);
  fusionNodes.push_back(mul_node);
  ge::NodePtr sub_node = graph.AddNode(sub_desc);
  fusionNodes.push_back(sub_node);

  // add attr
  Operator op_mean_var = ge::OpDescUtils::CreateOperatorFromNode(mean_var_node);
  op_mean_var.SetAttr("axes", axes0);
  op_mean_var.SetAttr("keep_dims", keep_dims0);// set node type

  mean_var_node->GetOpDesc()->SetType(REDUCEMEANVARIANCE);
  mul_node->GetOpDesc()->SetType("Mul");
  sub_node->GetOpDesc()->SetType("Sub");

  FUSION_PASS_CHECK(!CheckOpSupported(mean_var_desc), OP_LOGI(FUSED_OP_TYPE.c_str(), "fusion op is not supported"),
    return NOT_CHANGED);

  // connect input edge
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mean0_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            mean_var_node->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Add edge between node %s. and node %s failed.",
                                                   mean0_node->GetInDataAnchor(0)->GetPeerOutAnchor()->
                                                   GetOwnerNode()->GetName().c_str(),
                                                   mean_var_node->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mean_var_node->GetOutDataAnchor(0),
                                            mul_node->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Add edge between node %s. and node %s failed.",
                                                   mean_var_node->GetName().c_str(),
                                                   mul_node->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mean_var_node->GetOutDataAnchor(0),
                                            mul_node->GetInDataAnchor(1)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Add edge between node %s. and node %s failed.",
                                                   mean_var_node->GetName().c_str(),
                                                   mul_node->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mean_var_node->GetOutDataAnchor(1),
                                            sub_node->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Add edge between node %s. and node %s failed.",
                                                   mean_var_node->GetName().c_str(),
                                                   sub_node->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mul_node->GetOutDataAnchor(0),
                                            sub_node->GetInDataAnchor(1)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Add edge between node %s. and node %s failed.",
                                                   mul_node->GetName().c_str(),
                                                   sub_node->GetName().c_str()),
                    return FAILED);

  // connect output edge
  for (auto &inDataAnchor : mean0_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(mean0_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge1 failed."),
                      return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mean_var_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge1 failed."),
                      return FAILED);
  }

  for (auto &inDataAnchor : mean1_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(mean1_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge2 failed."),
                      return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(sub_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge2 failed."),
                      return FAILED);
  }



  // delete fused nodes
  FUSION_PASS_CHECK(graph.RemoveNode(mean0_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mean0_node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(squared0_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove squared0_node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mean1_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mean1_node failed."),
                    return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "ReduceMeanVarianceFusionPass graph fusion success!");
  return SUCCESS;
}
REGISTER_PASS("ZReduceMeanVarianceFusionPass", BUILT_IN_GRAPH_PASS, ReduceMeanVarianceFusionPass);
}  // namespace fe
