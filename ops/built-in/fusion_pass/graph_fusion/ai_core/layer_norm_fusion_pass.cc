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
 * \file layer_norm_fusion_pass.cpp
 * \brief layer norm fusion pass
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
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "fp16_t.hpp"

using namespace ge;
namespace fe {
static const char* MEAN = "ReduceMeanD";
static const char* SQUAREDDIFFERENCE = "SquaredDifference";
static const char* SUB = "Sub";
static const char* MUL = "Mul";
static const char* ADD = "Add";
static const char* RSQRT = "Rsqrt";
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
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("LayerNormFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "new a pattern object failed."),
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

Status LayerNormFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
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
  FUSION_PASS_CHECK(mean0_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "mean0_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(squared0_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "squared0_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(sub0_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "sub0_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mean1_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "mean1_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(add0_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "add0_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(rsqrt0_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "rsqrt0_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mul0_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "mul0_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mul1_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "mul1_node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(add1_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "add1_node is null, fusion failed."),
                    return PARAM_INVALID);

  // check input link
  std::string mean0_input_name = mean0_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName();
  std::string squared0_input0_name = squared0_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName();
  std::string squared0_input1_name = squared0_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName();
  std::string sub0_input_name = sub0_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName();
  if (strcmp(mean0_input_name.c_str(), sub0_input_name.c_str()) != 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "mean0_input and sub0_input are not same, not change");
    return NOT_CHANGED;
  }
  if ((strcmp(mean0_input_name.c_str(), squared0_input0_name.c_str()) != 0) &&
      (strcmp(mean0_input_name.c_str(), squared0_input1_name.c_str()) != 0)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "mean0_input and squared0_input are not same, not change");
    return NOT_CHANGED;
  }

  // check output link
  FUSION_PASS_CHECK(mean0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "mean0_node output size is [%d], which not equal to 2.",
                            mean0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(squared0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "squared0_node output size is [%d], which not equal to 1.",
                            squared0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(sub0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "sub0_node output size is [%d], which not equal to 1.",
                            sub0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(mean1_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "mean1_node output size is [%d], which not equal to 1.",
                            mean1_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(add0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "add0_node output size is [%d], which not equal to 1.",
                            add0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(rsqrt0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "rsqrt0_node output size is [%d], which not equal to 1.",
                            rsqrt0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(mul0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "mul0_node output size is [%d], which not equal to 1.",
                            mul0_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(mul1_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "mul1_node output size is [%d], which not equal to 1.",
                            mul1_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);

  // check input and attr
  ge::Operator op_mean0 = ge::OpDescUtils::CreateOperatorFromNode(mean0_node);
  std::vector<int64_t> axes0;
  if (GRAPH_SUCCESS != op_mean0.GetAttr("axes", axes0)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get attr axes failed.");
    return GRAPH_FAILED;
  }
  bool keep_dims0;
  if (GRAPH_SUCCESS != op_mean0.GetAttr("keep_dims", keep_dims0)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get attr keep_dims failed.");
    return GRAPH_FAILED;
  }
  Operator op_mean1 = ge::OpDescUtils::CreateOperatorFromNode(mean1_node);
  std::vector<int64_t> axes1;
  if (GRAPH_SUCCESS != op_mean1.GetAttr("axes", axes1)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get attr axes failed.");
    return GRAPH_FAILED;
  }
  bool keep_dims1;
  if (GRAPH_SUCCESS != op_mean0.GetAttr("keep_dims", keep_dims1)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get attr keep_dims failed.");
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
  ge::GeTensorDesc input_desc = mean0_node->GetOpDesc()->GetInputDesc(0);
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

  // check const input
  ge::Tensor add0_const;
  ge::Operator op_add0 = ge::OpDescUtils::CreateOperatorFromNode(add0_node);
  ge::GeTensorDesc add0_desc = add0_node->GetOpDesc()->GetInputDesc(0);
  if (GRAPH_SUCCESS != op_add0.GetInputConstData("x1", add0_const)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "get const data of add0 node failed for x1, need to check x2");
    if (GRAPH_SUCCESS != op_add0.GetInputConstData("x2", add0_const)) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "get const data of add0 node failed for x2, not change");
      return NOT_CHANGED;
    } else {
      add0_desc = add0_node->GetOpDesc()->GetInputDesc(1);
    }
  }
  std::vector<int64_t> add0_dims = add0_desc.GetShape().GetDims();
  if (add0_dims.size() != 0) {
    if (PatternFusionUtil::IsUnknownShape(add0_dims[0])) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "LayerNormFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
    if (!((add0_dims.size() == 1) && (add0_dims[0] == 1))) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "the const input of add0_node must be scalar, not change");
      return NOT_CHANGED;
    }
  }
  DataType dtype = op_add0.GetInputDesc("x1").GetDataType();
  std::vector<float> add0_vec;
  const uint8_t* add0_data = add0_const.GetData();
  if (add0_data == nullptr) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "const data of add0 node is null.");
    return GRAPH_FAILED;
  }
  if (dtype == ge::DT_FLOAT) {
    float value = (*((float*)add0_data));
    add0_vec.push_back(value);
  } else if (dtype == ge::DT_FLOAT16) {
    float value = (*((fp16_t*)add0_data)).toFloat();
    add0_vec.push_back(value);
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "add0_node type is not float or fp16, not change");
    return NOT_CHANGED;
  }

  Tensor mul0_const;
  Operator op_mul0 = ge::OpDescUtils::CreateOperatorFromNode(mul0_node);
  ge::NodePtr op_mul0_input0 = mul0_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  if (GRAPH_SUCCESS != op_mul0.GetInputConstData("x1", mul0_const)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "get const data of mul0 node failed for x1, need to check peer node");
    if (op_mul0_input0->GetType() != "Enter") {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "get enter node of mul0 node failed for x1, not change");
      return NOT_CHANGED;
    }
  }
  ge::GeTensorDesc mul0_desc = mul0_node->GetOpDesc()->GetInputDesc(0);
  std::vector<int64_t> mul0_dims = mul0_desc.GetShape().GetDims();
  if (mul0_dims.size() != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the const input of mul0_node must be 1D, not change");
    return NOT_CHANGED;
  }
  if (PatternFusionUtil::IsUnknownShape(input_dims[dims_size - 1]) ||
      PatternFusionUtil::IsUnknownShape(mul0_dims[0])) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "LayerNormFusionPass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }
  if (mul0_dims[0] != input_dims[dims_size - 1]) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the size of mul0 const input must be equal to the last dim of input, not change");
    return NOT_CHANGED;
  }

  Tensor add1_const;
  Operator op_add1 = ge::OpDescUtils::CreateOperatorFromNode(add1_node);
  ge::NodePtr op_add1_input0 = add1_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  if (GRAPH_SUCCESS != op_add1.GetInputConstData("x1", add1_const)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "get const data of add1 node failed for x1, need to check peer node");
    if (op_add1_input0->GetType() != "Enter") {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "get enter node of add1 node failed for x1, not change");
      return NOT_CHANGED;
    }
  }
  ge::GeTensorDesc add1_desc = add1_node->GetOpDesc()->GetInputDesc(0);
  std::vector<int64_t> add1_dims = add1_desc.GetShape().GetDims();
  if (add1_dims.size() != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the const input of add1_node must be 1D, not change");
    return NOT_CHANGED;
  }
  if (PatternFusionUtil::IsUnknownShape(add1_dims[0])) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "LayerNormFusionPass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }
  if (add1_dims[0] != input_dims[dims_size - 1]) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the size of add1 const input must be equal to the last dim of input, not change");
    return NOT_CHANGED;
  }

  // copy Opdesc
  std::shared_ptr<ge::OpDesc> layer_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (layer_desc = std::make_shared<ge::OpDesc>(add1_node->GetName() + "/" + LAYERNORM, LAYERNORM)),
      return FAILED);
  FUSION_PASS_CHECK(layer_desc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "layer_desc is null, fusion failed."),
                    return PARAM_INVALID);

  // add input
  FUSION_PASS_CHECK(layer_desc->AddInputDesc(0, input_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input1_desc failed."), return FAILED);
  ge::GeTensorDesc input2_desc = mul0_node->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(layer_desc->AddInputDesc(1, input2_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input2_desc failed."), return FAILED);
  ge::GeTensorDesc input3_desc = add1_node->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(layer_desc->AddInputDesc(2, input3_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input3_desc failed."), return FAILED);

  // add output
  ge::GeTensorDesc output1_desc = add1_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(layer_desc->AddOutputDesc("y", output1_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output1_desc failed."), return FAILED);
  ge::GeTensorDesc output2_desc = mean0_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(layer_desc->AddOutputDesc("mean", output2_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output2_desc failed."), return FAILED);
  ge::GeTensorDesc output3_desc = mean1_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(layer_desc->AddOutputDesc("variance", output3_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output3_desc failed."), return FAILED);

  // add layer_norm node
  ge::NodePtr layer_node = graph.AddNode(layer_desc);
  fusionNodes.push_back(layer_node);

  // add attr
  Operator op_layer = ge::OpDescUtils::CreateOperatorFromNode(layer_node);
  op_layer.SetAttr("begin_norm_axis", axes0[0]);
  op_layer.SetAttr("begin_params_axis", -1);
  op_layer.SetAttr("epsilon", add0_vec[0]);

  // connect input edge
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mean0_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            layer_node->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Add edge between node %s. and node %s failed.",
                            mean0_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                            layer_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mul0_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            layer_node->GetInDataAnchor(1)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Add edge between node %s. and node %s failed.",
                            mul0_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                            layer_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(add1_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            layer_node->GetInDataAnchor(2)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Add edge between node %s. and node %s failed.",
                            add1_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                            layer_node->GetName().c_str()),
                    return FAILED);

  // connect output edge
  for (auto &inDataAnchor : add1_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(add1_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."),
                                                     return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(layer_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."),
                                                     return FAILED);
  }

  // set node type
  layer_node->GetOpDesc()->SetType(LAYERNORM);

  // delete fused nodes
  FUSION_PASS_CHECK(graph.RemoveNode(mean0_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mean0_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(squared0_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove squared0_node failed."),
                                                   return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(sub0_node) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Remove sub0_node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mean1_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mean1_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(add0_node) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Remove add0_node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(rsqrt0_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove rsqrt0_node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mul0_node) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Remove mul0_node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mul1_node) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Remove mul1_node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(add1_node) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Remove add1_node failed."),
                    return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "LayerNormFusionPass graph fusion success!");
  return SUCCESS;
}
REGISTER_PASS("LayerNormFusionPass", BUILT_IN_GRAPH_PASS, LayerNormFusionPass);
}  // namespace fe
