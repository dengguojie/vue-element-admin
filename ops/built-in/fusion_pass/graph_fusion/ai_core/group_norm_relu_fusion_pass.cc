/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file group_norm_relu_fusion_pass.h
 * \brief reshape + instance_norm & const + reshape + mul + add + relu fusion pass
 */
#include "group_norm_relu_fusion_pass.h"

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
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {
static int64_t SIZE = 2;
static const char *RESHAPE = "Reshape";
static const char *INSTANCENORM = "InstanceNorm";
static const char *MUL = "Mul";
static const char *ADD = "Add";
static const char *RELU = "Relu";
static const std::string PATTERN_RESHAPE_1 = "Reshape_1";
static const std::string PATTERN_RESHAPE_2 = "Reshape_2";
static const std::string PATTERN_INSTANCENORM = "InstanceNorm";
static const std::string PATTERN_MUL = "Mul";
static const std::string PATTERN_ADD = "Add";
static const std::string PATTERN_RELU = "Relu";
static const std::string GROUP_NORM_RELU = "GroupNormRelu";
/*                        
        x                         |              
     /     \                      |           
 Reshape    \                     |         
    |        |                    |               
    |        |            x       | 
    |        |            |       |
InstanceNorm Const   ==> GNR      |               
    \        /            |       |     
     \      /             y       |
      \    /                      |                                   
     Reshape                      |              
        |                         |             
       Mul                        |              
        |                         |
       Add                        |
        |                         |
      Relu                        |
        |                         |
        y                         |
*/
/*input:first add, second remove;output:first remove, second add*/
vector<FusionPattern*> GroupNormReluFusionPass::DefinePatterns() {
  vector < FusionPattern * > patterns;
  FusionPattern *pattern =
      new (std::nothrow) FusionPattern("GroupNormReluFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), 
                    "new a pattern object failed."), return patterns);
  pattern->AddOpDesc(PATTERN_RESHAPE_1, {RESHAPE})
      .AddOpDesc(PATTERN_INSTANCENORM, {INSTANCENORM})
      .AddOpDesc(PATTERN_RESHAPE_2, {RESHAPE})
      .AddOpDesc(PATTERN_MUL, {MUL})
      .AddOpDesc(PATTERN_ADD, {ADD})
      .AddOpDesc(PATTERN_RELU, {RELU})
      .SetInputs(PATTERN_INSTANCENORM, {PATTERN_RESHAPE_1})
      .SetInputs(PATTERN_RESHAPE_2, {PATTERN_INSTANCENORM})
      .SetInputs(PATTERN_MUL, {PATTERN_RESHAPE_2})
      .SetInputs(PATTERN_ADD, {PATTERN_MUL})
      .SetInputs(PATTERN_RELU, {PATTERN_ADD})
      .SetOutput(PATTERN_RELU);

  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define GroupNormReluFusionPass pattern end");
  return patterns;
}

Status GroupNormReluFusionPass::Fusion(ge::ComputeGraph& graph,
                                        Mapping& mapping,
                                        vector<ge::NodePtr> &fusionNodes) {
  // get all nodes
  ge::NodePtr reshape_node = GetNodeFromMapping(PATTERN_RESHAPE_1, mapping);
  ge::NodePtr instance_norm_node = GetNodeFromMapping(PATTERN_INSTANCENORM, mapping);
  ge::NodePtr second_reshape_node = GetNodeFromMapping(PATTERN_RESHAPE_2, mapping);
  ge::NodePtr mul_node = GetNodeFromMapping(PATTERN_MUL, mapping);
  ge::NodePtr add_node = GetNodeFromMapping(PATTERN_ADD, mapping);
  ge::NodePtr relu_node = GetNodeFromMapping(PATTERN_RELU, mapping);
  FUSION_PASS_CHECK(reshape_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), 
                    "reshape_node is null, fusion failed."), return NOT_CHANGED);
  FUSION_PASS_CHECK(instance_norm_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), 
                    "instance_norm_node is null, fusion failed."), return NOT_CHANGED);
  FUSION_PASS_CHECK(mul_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), 
                    "mul_node is null, fusion failed."), return NOT_CHANGED);
  FUSION_PASS_CHECK(add_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), 
                    "add_node is null, fusion failed."), return NOT_CHANGED);

  // check output link
  FUSION_PASS_CHECK(reshape_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape_node output size is [%d], which not equal to 1.",
                            reshape_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(instance_norm_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "instance_norm_node output size is [%d], which not equal to 1.",
                            instance_norm_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(mul_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "mul_node output size is [%d], which not equal to 1.",
                            mul_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(add_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add_node output size is [%d], which not equal to 1.",
                            add_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(second_reshape_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add_node output size is [%d], which not equal to 1.",
                            second_reshape_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);

  // get all node's desc
  ge::OpDescPtr reshape_desc = reshape_node->GetOpDesc();
  ge::OpDescPtr instance_norm_desc = instance_norm_node->GetOpDesc();
  ge::OpDescPtr mul_desc = mul_node->GetOpDesc();
  ge::OpDescPtr add_desc = add_node->GetOpDesc(); 
  FUSION_PASS_CHECK(reshape_desc == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), 
                    "reshape_node's OpDesc is null, fusion failed."), return NOT_CHANGED);
  FUSION_PASS_CHECK(instance_norm_desc == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), 
                    "instance_norm_node's OpDesc is null, fusion failed."), return NOT_CHANGED);
  FUSION_PASS_CHECK(mul_desc == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), 
                    "mul_node's OpDesc is null, fusion failed."), return NOT_CHANGED);
  FUSION_PASS_CHECK(add_desc == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), 
                    "add_node's OpDesc is null, fusion failed."), return NOT_CHANGED);

  // set new op group_norm_relu
  std::shared_ptr<ge::OpDesc> group_norm_relu_desc = nullptr;
  group_norm_relu_desc = std::make_shared<ge::OpDesc>(mul_node->GetName() + "/" + GROUP_NORM_RELU, 
                                                      GROUP_NORM_RELU);
  FUSION_PASS_CHECK(group_norm_relu_desc == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), 
                    "group_norm_relu_desc is null, fusion failed."),
                    return NOT_CHANGED);

  // add input
  ge::GeTensorDesc input_desc_0 = reshape_desc->GetInputDesc(0);
  FUSION_PASS_CHECK(group_norm_relu_desc->AddInputDesc(input_desc_0) != SUCCESS, 
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input_0 failed."),
                    return NOT_CHANGED);
  ge::GeTensorDesc input_desc_1 = mul_desc->GetInputDesc(1);
  FUSION_PASS_CHECK(group_norm_relu_desc->AddInputDesc(input_desc_1) != SUCCESS, 
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input_1 failed."),
                    return NOT_CHANGED);
  ge::GeTensorDesc input_desc_2 = add_desc->GetInputDesc(1);
  FUSION_PASS_CHECK(group_norm_relu_desc->AddInputDesc(input_desc_2) != SUCCESS, 
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input_2 failed."),
                    return NOT_CHANGED);

  // add output
  ge::GeTensorDesc output_desc = relu_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(group_norm_relu_desc->AddOutputDesc(output_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), 
                    "add output failed."), return NOT_CHANGED);

  // add group_norm_relu node
  ge::NodePtr group_norm_relu_node = graph.AddNode(group_norm_relu_desc);
  fusionNodes.push_back(group_norm_relu_node);

  // add attr
  ge::GeTensorDesc reshape_output_desc = reshape_desc->GetOutputDesc(0);
  auto dims = reshape_output_desc.GetShape().GetDims();
  FUSION_PASS_CHECK(dims.size() < SIZE,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), 
                    "the reshape dims are below 2."), return NOT_CHANGED);
  auto num_groups = dims[1];
  Operator op_group_norm_relu = ge::OpDescUtils::CreateOperatorFromNode(group_norm_relu_node);
  op_group_norm_relu.SetAttr("num_groups", num_groups);
  Operator op_instance_norm = ge::OpDescUtils::CreateOperatorFromNode(instance_norm_node);
  float epsilon = 0;
  op_instance_norm.GetAttr("epsilon", epsilon);
  op_group_norm_relu.SetAttr("eps", epsilon);

  // check whether op is supported
  FUSION_PASS_CHECK(!CheckOpSupported(group_norm_relu_desc), 
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Op Not Supported."),
                    return NOT_CHANGED);

  // add input edge
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
      reshape_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
      group_norm_relu_node->GetInDataAnchor(0)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
          reshape_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
          group_norm_relu_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
      mul_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
      group_norm_relu_node->GetInDataAnchor(1)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
          mul_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
          group_norm_relu_node->GetName().c_str()),
      return FAILED); 
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
      add_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
      group_norm_relu_node->GetInDataAnchor(2)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
          add_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
          group_norm_relu_node->GetName().c_str()),
      return FAILED); 

  // add output edge
  for (auto &inDataAnchor : relu_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(relu_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), 
                      "Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(group_norm_relu_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), 
                      "Add out data edge failed."), return FAILED);
  }

  // delete fused nodes
  FUSION_PASS_CHECK(graph.RemoveNode(reshape_node) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(),
                    "Remove reshape_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(instance_norm_node) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(),
                    "Remove instance_norm_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(second_reshape_node) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(),
                    "Remove second_reshape_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mul_node) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(),
                    "Remove mul_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(add_node) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(),
                    "Remove add_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(relu_node) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(),
                    "Remove relu_node failed."), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "GroupNormReluFusionPass graph fusion success!");
  return SUCCESS;
}

REGISTER_PASS("GroupNormReluFusionPass", BUILT_IN_GRAPH_PASS, GroupNormReluFusionPass);
}
