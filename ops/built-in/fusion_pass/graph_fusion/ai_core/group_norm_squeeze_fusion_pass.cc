/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
 * \file group_norm_squeeze_fusion_pass.cc
 * \brief squeeze + instance_norm & const + mul + add fusion pass
 */
#include "group_norm_squeeze_fusion_pass.h"

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
static const uint64_t SIZE = 2;
static const char *SQUEEZE = "Squeeze";
static const char *INSTANCENORM = "InstanceNorm";
static const char *MUL = "Mul";
static const char *ADD = "Add";
static const std::string PATTERN_SQUEEZE = "Squeeze";
static const std::string PATTERN_INSTANCENORM = "InstanceNorm";
static const std::string PATTERN_MUL = "Mul";
static const std::string PATTERN_ADD = "Add";
static const std::string GROUP_NORM = "GroupNorm";

vector<FusionPattern*> GroupNormSqueezeFusionPass::DefinePatterns() {
  vector < FusionPattern *> patterns;
  FusionPattern *pattern =
      new (std::nothrow) FusionPattern("GroupNormSqueezeFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(),
                    "new a pattern object failed."), return patterns);
  pattern->AddOpDesc(PATTERN_SQUEEZE, {SQUEEZE})
      .AddOpDesc(PATTERN_INSTANCENORM, {INSTANCENORM})
      .AddOpDesc(PATTERN_MUL, {MUL})
      .AddOpDesc(PATTERN_ADD, {ADD})
      .SetInputs(PATTERN_INSTANCENORM, {PATTERN_SQUEEZE})
      .SetInputs(PATTERN_MUL, {PATTERN_INSTANCENORM})
      .SetInputs(PATTERN_ADD, {PATTERN_MUL})
      .SetOutput(PATTERN_ADD);

  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define GroupNormSqueezeFusionPass pattern end");
  return patterns;
}

Status GroupNormSqueezeFusionPass::CheckNode(const ge::NodePtr &squeeze_node,
                                             const ge::NodePtr &instance_norm_node,
                                             const ge::NodePtr &mul_node,
                                             const ge::NodePtr &add_node) {
  FUSION_PASS_CHECK(squeeze_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(),
                    "squeeze_node is null, fusion failed."), return NOT_CHANGED);
  FUSION_PASS_CHECK(instance_norm_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(),
                    "instance_norm_node is null, fusion failed."), return NOT_CHANGED);
  FUSION_PASS_CHECK(mul_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(),
                    "mul_node is null, fusion failed."), return NOT_CHANGED);
  FUSION_PASS_CHECK(add_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(),
                    "add_node is null, fusion failed."), return NOT_CHANGED);

  FUSION_PASS_CHECK(instance_norm_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "instance_norm_node output size is [%d], which not equal to 1.",
                            instance_norm_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(mul_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "mul_node output size is [%d], which not equal to 1.",
                            mul_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  return SUCCESS;
}

Status GroupNormSqueezeFusionPass::GenGroupNorm(const ge::NodePtr &squeeze_node,
                                                const ge::NodePtr &instance_norm_node,
                                                const ge::NodePtr &mul_node,
                                                const ge::NodePtr &add_node,
                                                ge::OpDescPtr &group_norm_desc) {
  ge::OpDescPtr squeeze_desc = squeeze_node->GetOpDesc();
  ge::OpDescPtr instance_norm_desc = instance_norm_node->GetOpDesc();
  ge::OpDescPtr mul_desc = mul_node->GetOpDesc();
  ge::OpDescPtr add_desc = add_node->GetOpDesc();
  FUSION_PASS_CHECK(squeeze_desc == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(),
                    "squeeze_node's OpDesc is null, fusion failed."), return NOT_CHANGED);
  FUSION_PASS_CHECK(instance_norm_desc == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(),
                    "instance_norm_node's OpDesc is null, fusion failed."), return NOT_CHANGED);
  FUSION_PASS_CHECK(mul_desc == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(),
                    "mul_node's OpDesc is null, fusion failed."), return NOT_CHANGED);
  FUSION_PASS_CHECK(add_desc == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(),
                    "add_node's OpDesc is null, fusion failed."), return NOT_CHANGED);

  // add input
  ge::GeTensorDesc input_desc_0 = squeeze_desc->GetInputDesc(0);
  FUSION_PASS_CHECK(group_norm_desc->AddInputDesc(input_desc_0) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input_0 failed."),
                    return NOT_CHANGED);
  ge::GeTensorDesc mul_desc_1 = mul_desc->GetInputDesc(1);
  FUSION_PASS_CHECK(group_norm_desc->AddInputDesc(mul_desc_1) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input_1 failed."),
                    return NOT_CHANGED);
  ge::GeTensorDesc add_desc_1 = add_desc->GetInputDesc(1);
  FUSION_PASS_CHECK(group_norm_desc->AddInputDesc(add_desc_1) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add input_2 failed."),
                    return NOT_CHANGED);
  // add output
  ge::GeTensorDesc squeeze_output_desc = squeeze_desc->GetOutputDesc(0);
  auto dims = squeeze_output_desc.GetShape().GetDims();
  FUSION_PASS_CHECK(dims.size() < SIZE,
                    OP_LOGI(FUSED_OP_TYPE.c_str(),
                    "the squeeze dims are below 2."), return NOT_CHANGED);
  auto num_groups = dims[1];
  auto batch_n = dims[0];

  ge::GeTensorDesc mean_desc;
  vector<int64_t> mean_shape;
  mean_shape.push_back(batch_n * num_groups);
  mean_desc.SetShape(ge::GeShape(mean_shape));
  mean_desc.SetFormat(ge::FORMAT_ND);
  mean_desc.SetDataType(squeeze_output_desc.GetDataType());

  ge::GeTensorDesc output_desc = add_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(group_norm_desc->AddOutputDesc("y", output_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(),
                    "add output failed."), return NOT_CHANGED);
  FUSION_PASS_CHECK(group_norm_desc->AddOutputDesc("mean", mean_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(),
                    "add output failed."), return NOT_CHANGED);
  FUSION_PASS_CHECK(group_norm_desc->AddOutputDesc("var", mean_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(),
                    "add output failed."), return NOT_CHANGED);
  return SUCCESS;
}

Status GroupNormSqueezeFusionPass::AddNodeEdge(ge::NodePtr &squeeze_node,
                                               ge::NodePtr &instance_norm_node,
                                               ge::NodePtr &mul_node,
                                               ge::NodePtr &add_node,
                                               ge::NodePtr &group_norm_node) {
  // add input edge
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
      squeeze_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
      group_norm_node->GetInDataAnchor(0)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
          squeeze_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
          group_norm_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
      mul_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
      group_norm_node->GetInDataAnchor(1)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
          mul_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
          group_norm_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
      add_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
      group_norm_node->GetInDataAnchor(2)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
          add_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
          group_norm_node->GetName().c_str()),
      return FAILED);

  // add output edge
  for (auto &inDataAnchor : add_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(add_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(group_norm_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "Add out data edge failed."), return FAILED);
  }
  return SUCCESS;
}

Status GroupNormSqueezeFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr> &fusionNodes) {
  // get all nodes
  ge::NodePtr squeeze_node = GetNodeFromMapping(PATTERN_SQUEEZE, mapping);
  ge::NodePtr instance_norm_node = GetNodeFromMapping(PATTERN_INSTANCENORM, mapping);
  ge::NodePtr mul_node = GetNodeFromMapping(PATTERN_MUL, mapping);
  ge::NodePtr add_node = GetNodeFromMapping(PATTERN_ADD, mapping);

  Status result = CheckNode(squeeze_node, instance_norm_node, mul_node, add_node);
  FUSION_PASS_CHECK(result != SUCCESS, OP_LOGI(FUSED_OP_TYPE.c_str(),
                   "check failed, fusion failed."), return result);

  // set new op group_norm
  std::shared_ptr<ge::OpDesc> group_norm_desc = nullptr;
  group_norm_desc = std::make_shared<ge::OpDesc>(mul_node->GetName() + "/" + GROUP_NORM, GROUP_NORM);
  FUSION_PASS_CHECK(group_norm_desc == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(),
                   "group_norm_desc is null, fusion failed."), return NOT_CHANGED);
  Status result_desc = GenGroupNorm(squeeze_node, instance_norm_node, mul_node, add_node, group_norm_desc);
  FUSION_PASS_CHECK(result_desc != SUCCESS, OP_LOGI(FUSED_OP_TYPE.c_str(),
                   "create groupnorm failed, fusion failed."), return result_desc);

  // add group_norm node
  ge::NodePtr group_norm_node = graph.AddNode(group_norm_desc);
  fusionNodes.push_back(group_norm_node);

  // add attr
  ge::OpDescPtr squeeze_desc = squeeze_node->GetOpDesc();
  ge::GeTensorDesc squeeze_output_desc = squeeze_desc->GetOutputDesc(0);
  auto dims = squeeze_output_desc.GetShape().GetDims();
  auto num_groups = dims[1];
  Operator op_group_norm = ge::OpDescUtils::CreateOperatorFromNode(group_norm_node);
  op_group_norm.SetAttr("num_groups", num_groups);
  Operator op_instance_norm = ge::OpDescUtils::CreateOperatorFromNode(instance_norm_node);
  float epsilon = 0;
  int fusion_axis = 2;
  op_instance_norm.GetAttr("epsilon", epsilon);
  op_group_norm.SetAttr("eps", epsilon);
  Operator op_squeeze = ge::OpDescUtils::CreateOperatorFromNode(squeeze_node);
  std::vector<int> axis;
  op_squeeze.GetAttr("axis", axis);
  FUSION_PASS_CHECK(axis[0] != fusion_axis, OP_LOGI(FUSED_OP_TYPE.c_str(),
                   "axis of squeeze is not equal to 2, fusion failed."), return NOT_CHANGED);

  // check whether op is supported
  FUSION_PASS_CHECK(!CheckOpSupported(group_norm_desc),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Op Not Supported."),
                    return NOT_CHANGED);

  Status result_edge = AddNodeEdge(squeeze_node, instance_norm_node, mul_node, add_node, group_norm_node);
  FUSION_PASS_CHECK(result_edge != SUCCESS, OP_LOGI(FUSED_OP_TYPE.c_str(),
                   "add edge failed, fusion failed."), return result_edge);

  // delete fused nodes
  FUSION_PASS_CHECK(graph.RemoveNode(squeeze_node) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(),
                    "Remove squeeze_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(instance_norm_node) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(),
                    "Remove instance_norm_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mul_node) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(),
                    "Remove mul_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(add_node) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(),
                    "Remove add_node failed."), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "GroupNormSqueezeFusionPass graph fusion success!");
  return SUCCESS;
}

REGISTER_PASS("GroupNormSqueezeFusionPass", BUILT_IN_GRAPH_PASS, GroupNormSqueezeFusionPass);
}
