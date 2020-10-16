/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @brief layer norm fusion pass
 *
 */

#include "layer_norm_fusion_pass.h"

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
static const char *MEAN = "ReduceMeanD";
static const char *SQUAREDDIFFERENCE = "SquaredDifference";
static const char *SUB = "Sub";
static const char *MUL = "Mul";
static const char *ADD = "Add";
static const char *RSQRT = "Rsqrt";
static const std::string PATTERN_MEAN0 = "FusedNodeMean0";
static const std::string PATTERN_SQUAREDDIFFERENCE0 = "FusedNodeSquaredDifference0";
static const std::string PATTERN_SUB0 = "FusedNodeSub0";
static const std::string PATTERN_MEAN1 = "FusedNodeMean1";
static const std::string PATTERN_ADD0 = "FusedNodeAdd0";
static const std::string PATTERN_RSQRT0 = "FusedNodeRsqrt0";
static const std::string PATTERN_MUL0 = "FusedNodeMul0";
static const std::string PATTERN_MUL1 = "FusedNodeMul1";
static const std::string PATTERN_ADD1 = "FusedNodeAdd1";
static const std::string LAYERNORM = "LayerNorm";

vector<FusionPattern*> LayerNormFusionPass::DefinePatterns() {
  vector < FusionPattern * > patterns;

  FusionPattern *pattern =
      new (std::nothrow) FusionPattern("LayerNormFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
           return patterns);
  pattern->AddOpDesc(PATTERN_MEAN0, {MEAN})
      .AddOpDesc(PATTERN_SQUAREDDIFFERENCE0, {SQUAREDDIFFERENCE})
      .AddOpDesc(PATTERN_SUB0, {SUB})
      .AddOpDesc(PATTERN_MEAN1, {MEAN})
      .AddOpDesc(PATTERN_ADD0, {ADD})
      .AddOpDesc(PATTERN_RSQRT0, {RSQRT})
      .AddOpDesc(PATTERN_MUL0, {MUL})
      .AddOpDesc(PATTERN_MUL1, {MUL})
      .AddOpDesc(PATTERN_ADD1, {{ADD}})
      .SetInputs(PATTERN_SQUAREDDIFFERENCE0, {PATTERN_MEAN0})
      .SetInputs(PATTERN_SUB0, {PATTERN_MEAN0})
      .SetInputs(PATTERN_MEAN1, {PATTERN_SQUAREDDIFFERENCE0})
      .SetInputs(PATTERN_ADD0, {PATTERN_MEAN1})
      .SetInputs(PATTERN_RSQRT0, {PATTERN_ADD0})
      .SetInputs(PATTERN_MUL0, {PATTERN_RSQRT0})
      .SetInputs(PATTERN_MUL1, {PATTERN_MUL0, PATTERN_SUB0})
      .SetInputs(PATTERN_ADD1, {PATTERN_MUL1})
      .SetOutput(PATTERN_ADD1);

  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define LayerNormFusionPass pattern end");
  return patterns;
}

Status LayerNormFusionPass::Fusion(ge::ComputeGraph& graph,
                                     Mapping& mapping,
                                     vector<ge::NodePtr> &fusionNodes) {
  // get all nodes
  ge::NodePtr mean0_node = GetNodeFromMapping(PATTERN_MEAN0, mapping);
  ge::NodePtr squared0_node = GetNodeFromMapping(PATTERN_SQUAREDDIFFERENCE0, mapping);
  ge::NodePtr sub0_node = GetNodeFromMapping(PATTERN_SUB0, mapping);
  ge::NodePtr mean1_node = GetNodeFromMapping(PATTERN_MEAN1, mapping);
  ge::NodePtr add0_node = GetNodeFromMapping(PATTERN_ADD0, mapping);
  ge::NodePtr rsqrt0_node = GetNodeFromMapping(PATTERN_RSQRT0, mapping);
  ge::NodePtr mul0_node = GetNodeFromMapping(PATTERN_MUL0, mapping);
  ge::NodePtr mul1_node = GetNodeFromMapping(PATTERN_MUL1, mapping);
  ge::NodePtr add1_node = GetNodeFromMapping(PATTERN_ADD1, mapping);
  FUSION_PASS_CHECK(mean0_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mean0_node is null, fusion failed."), return PARAM_INVALID);
  FUSION_PASS_CHECK(squared0_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "squared0_node is null, fusion failed."), return PARAM_INVALID);
  FUSION_PASS_CHECK(sub0_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "sub0_node is null, fusion failed."), return PARAM_INVALID);
  FUSION_PASS_CHECK(mean1_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mean1_node is null, fusion failed."), return PARAM_INVALID);
  FUSION_PASS_CHECK(add0_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "add0_node is null, fusion failed."), return PARAM_INVALID);
  FUSION_PASS_CHECK(rsqrt0_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "rsqrt0_node is null, fusion failed."), return PARAM_INVALID);
  FUSION_PASS_CHECK(mul0_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mul0_node is null, fusion failed."), return PARAM_INVALID);
  FUSION_PASS_CHECK(mul1_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mul1_node is null, fusion failed."), return PARAM_INVALID);
  FUSION_PASS_CHECK(add1_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "add1_node is null, fusion failed."), return PARAM_INVALID);

  //get const data
  Operator op_mean0 = ge::OpDescUtils::CreateOperatorFromNode(mean0_node);
  std::vector<int32_t> const_data0;
  if (GRAPH_SUCCESS != op_mean0.GetAttr("axes", const_data0)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "get attr axes failed.");
    return GRAPH_FAILED;
  }

  Operator op_mean1 = ge::OpDescUtils::CreateOperatorFromNode(mean1_node);
  std::vector<int32_t> const_data1;
  if (GRAPH_SUCCESS != op_mean1.GetAttr("axes", const_data1)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "get attr axes failed.");
    return GRAPH_FAILED;
  }

  Operator op_add0 = ge::OpDescUtils::CreateOperatorFromNode(add0_node);
  Tensor data2;
  if (GRAPH_SUCCESS != op_add0.GetInputConstData("x1", data2)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "get const data of add0 node failed for x1, need to check x2");
    if (GRAPH_SUCCESS != op_add0.GetInputConstData("x2", data2)) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "get const data of add0 node failed for x2, not change");
      return NOT_CHANGED;
    }
  }
  std::vector<float> const_data2;
  float* const_data_ptr2 = (float*) data2.GetData();
  FUSION_PASS_CHECK(const_data_ptr2 == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "const_data_ptr2 is null, fusion failed."), return PARAM_INVALID);
  const_data2.push_back((float) (*(const_data_ptr2)));

  // copy Opdesc
  std::shared_ptr<ge::OpDesc> layer_desc = nullptr;
  layer_desc =
      std::make_shared<ge::OpDesc>(add1_node->GetName() + "/" + LAYERNORM, LAYERNORM);
  FUSION_PASS_CHECK(layer_desc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "layer_desc is null, fusion failed."),
           return PARAM_INVALID);

  // add input
  ge::GeTensorDesc input1_desc = mean0_node->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(layer_desc->AddInputDesc(0, input1_desc) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "add input1_desc failed."), return FAILED);



  Tensor test_data;
  ge::GeTensorDesc input2_desc = mul0_node->GetOpDesc()->GetInputDesc(0);
  Operator op_mul0 = ge::OpDescUtils::CreateOperatorFromNode(mul0_node);
  ge::NodePtr op_mul0_input0 = mul0_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  if (GRAPH_SUCCESS != op_mul0.GetInputConstData("x1", test_data)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "get const data of mul0 node failed for x1, need to check peer node");
    if (op_mul0_input0->GetType() != "Enter") {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "get enter node of mul0 node failed for x1, not change");
      return NOT_CHANGED;
    }
  }
  FUSION_PASS_CHECK(layer_desc->AddInputDesc(1, input2_desc) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "add input2_desc failed."), return FAILED);

  ge::GeTensorDesc input3_desc = add1_node->GetOpDesc()->GetInputDesc(0);
  Operator op_add1 = ge::OpDescUtils::CreateOperatorFromNode(add1_node);
  ge::NodePtr op_add1_input0 = add1_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  ge::NodePtr op_add1_input1 = add1_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();\
  if (GRAPH_SUCCESS != op_add1.GetInputConstData("x1", test_data)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "get const data of add1 node failed for x1, need to check peer node");
    if (op_add1_input0->GetType() != "Enter") {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "get enter node of add1 node failed for x1, not change");
      return NOT_CHANGED;
    }
  }
  FUSION_PASS_CHECK(layer_desc->AddInputDesc(2, input3_desc) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "add input3_desc failed."), return FAILED);

  // add output
  ge::GeTensorDesc output1_desc = add1_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(layer_desc->AddOutputDesc("y", output1_desc) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "add output1_desc failed."), return FAILED);
  ge::GeTensorDesc output2_desc = mean0_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(layer_desc->AddOutputDesc("mean", output2_desc) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "add output2_desc failed."), return FAILED);
  ge::GeTensorDesc output3_desc = mean1_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(layer_desc->AddOutputDesc("variance", output3_desc) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "add output3_desc failed."), return FAILED);

  // add layer_norm node
  ge::NodePtr layer_node = graph.AddNode(layer_desc);
  fusionNodes.push_back(layer_node);

  // add attr
  Operator op_layer= ge::OpDescUtils::CreateOperatorFromNode(layer_node);
  if (!const_data0.empty()) {
    op_layer.SetAttr("begin_norm_axis", const_data0[0]);
  }
  op_layer.SetAttr("begin_params_axis", -1);
  op_layer.SetAttr("epsilon", const_data2[0]);

  // connect input edge
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
               mean0_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
               layer_node->GetInDataAnchor(0)) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                   mean0_node->GetInDataAnchor(0)
                       ->GetPeerOutAnchor()
                       ->GetOwnerNode()
                       ->GetName()
                       .c_str(),
                   layer_node->GetName().c_str()),
           return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
               mul0_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
               layer_node->GetInDataAnchor(1)) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                   mul0_node->GetInDataAnchor(0)
                       ->GetPeerOutAnchor()
                       ->GetOwnerNode()
                       ->GetName()
                       .c_str(),
                   layer_node->GetName().c_str()),
           return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
               add1_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
               layer_node->GetInDataAnchor(2)) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                   add1_node->GetInDataAnchor(0)
                       ->GetPeerOutAnchor()
                       ->GetOwnerNode()
                       ->GetName()
                       .c_str(),
                   layer_node->GetName().c_str()),
           return FAILED);

  // connect output edge
  for (auto inDataAnchor :
       add1_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(add1_node->GetOutDataAnchor(0),
                                        inDataAnchor) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(layer_node->GetOutDataAnchor(0),
                                     inDataAnchor) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
  }

  // set node type
  layer_node->GetOpDesc()->SetType(LAYERNORM);

  // delete fused nodes
  FUSION_PASS_CHECK(graph.RemoveNode(mean0_node) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove mean0_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(squared0_node) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove squared0_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(sub0_node) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove sub0_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mean1_node) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove mean1_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(add0_node) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove add0_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(rsqrt0_node) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove rsqrt0_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mul0_node) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove mul0_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mul1_node) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove mul1_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(add1_node) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove add1_node failed."), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "LayerNormFusionPass graph fusion success!");
  return SUCCESS;
}
REGISTER_PASS("LayerNormFusionPass", BUILT_IN_GRAPH_PASS, LayerNormFusionPass);
}
