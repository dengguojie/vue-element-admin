/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file mul_maximum_fusion_pass.cpp
 * \brief
 */
#include "mul_maximum_fusion_pass.h"

#include <iostream>
#include <vector>
#include <map>

#include "op_log.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "fp16_t.hpp"

namespace fe {
static const string MUL = "Mul";
static const string MAXIMUM = "Maximum";
static const string PATTERN_MUL = "Mul";
static const string PATTERN_MAXIMUM = "Maximum";
static const string PATTERN_INPUT = "input";

/*  tf.nn.leaky_relu fusion by mul and maximum
    alpha as a constant input for mul
                     xxx
                     / \
                    /   \
                   |     |
                   |    mul <----const op (scale factor:alpha)
                   |     |
                    \   /
                     \ /
                  maximum(x=mul,y=xxx)
*/

vector<FusionPattern*> MulMaximumFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define MulMaximumFusionPass pattern begin");
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("MulMaximumFusion");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_MUL, {MUL})
      .AddOpDesc(PATTERN_MAXIMUM, {MAXIMUM})
      .AddOpDesc(PATTERN_INPUT)
      .SetInputs(PATTERN_MUL, {PATTERN_INPUT})
      .SetInputs(PATTERN_MAXIMUM, {PATTERN_MUL, PATTERN_INPUT})
      .SetOutput(PATTERN_MAXIMUM);

  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define MulMaximumFusionPass pattern end");
  return patterns;
}

Status MulMaximumFusionPass::CheckPeerMulInDataAnchors(const ge::OutDataAnchorPtr& outputAnchor,
                                                       const size_t& expectedNum) {
  FUSION_PASS_CHECK(outputAnchor == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "outputAnchor must not be null"),
                    return PARAM_INVALID);
  if (outputAnchor->GetPeerInDataAnchors().size() == expectedNum) {
    return SUCCESS;
  }
  return FAILED;
}

Status MulMaximumFusionPass::IsMatch(ge::NodePtr& mulNode, ge::NodePtr& maximumNode) {
  FUSION_PASS_CHECK(mulNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mulNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(maximumNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "maximumNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(CheckPeerMulInDataAnchors(mulNode->GetOutDataAnchor(0), 1) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "%s contains more than one peer input", mulNode->GetName().c_str()),
                    return NOT_CHANGED);

  size_t maximum_in_nodes_size = ge::OpDescUtils::GetNonConstInputsSize(maximumNode);
  if (maximum_in_nodes_size != 2) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "maximum node need 2 inputs,but now is %d.", (int)maximum_in_nodes_size);
    return NOT_CHANGED;
  }
  auto in_anchors = mulNode->GetAllInDataAnchors();
  std::vector<ge::GeTensorPtr> mulWeights;
  for (auto in_anchor : in_anchors) {
    auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr)
      continue;

    auto in_node = out_anchor->GetOwnerNode();
    string nodeType = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(in_node);
    if (nodeType == "Const" || nodeType == "Constant") {
      ge::OpDescPtr opDesc = in_node->GetOpDesc();
      ge::GeTensorPtr weight = nullptr;
      ge::AttrUtils::MutableTensor(opDesc, ge::ATTR_NAME_WEIGHTS, weight);
      mulWeights.push_back(weight);
    }
  }
  size_t inputs_in_nodes_size = mulNode->GetOpDesc()->GetInputsSize();
  size_t mul_in_nodes_size = inputs_in_nodes_size - mulWeights.size();
  if (mul_in_nodes_size != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "mul node need 1 inputs,but now is %d.", (int)mul_in_nodes_size);
    return NOT_CHANGED;
  }

  uint32_t const_input_index = 0;
  for (uint32_t index = 0; index < mulNode->GetAllInDataAnchors().size(); index++) {
    ge::NodePtr input_node_const = mulNode->GetInDataAnchor(index)->GetPeerOutAnchor()->GetOwnerNode();
    string nodeType = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(input_node_const);
    if (nodeType != "Const" && nodeType != "Constant") {
      const_input_index = index;
      break;
    }
  }
  ge::NodePtr mulInputNode = mulNode->GetInDataNodes().at(const_input_index);
  FUSION_PASS_CHECK(mulInputNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mulInputNode is null, fusion failed."),
                    return PARAM_INVALID);
  for (auto maximumInputNode : maximumNode->GetInDataNodes()) {
    int64_t mulInputNodeId = mulInputNode->GetOpDesc()->GetId();
    int64_t maximumInputNodeId = maximumInputNode->GetOpDesc()->GetId();
    if (mulInputNodeId == maximumInputNodeId &&
        mulInputNode->GetOpDesc()->GetType() == maximumInputNode->GetOpDesc()->GetType() &&
        mulInputNode->GetOpDesc()->GetName() == maximumInputNode->GetOpDesc()->GetName()) {
      return SUCCESS;
    }
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "leaky_relu maximum.input_node.not same to mul.input_node.can not fusion.");
  return NOT_CHANGED;
}

Status MulMaximumFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define MulMaximumFusionPass fusion begin");
  ge::NodePtr mulNode = GetNodeFromMapping(PATTERN_MUL, mapping);
  ge::NodePtr maximumNode = GetNodeFromMapping(PATTERN_MAXIMUM, mapping);
  FUSION_PASS_CHECK(mulNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "MulNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(maximumNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "MaximumNode is null, fusion failed."),
                    return PARAM_INVALID);

  if (IsMatch(mulNode, maximumNode) != SUCCESS) {
    return NOT_CHANGED;
  }

  auto in_anchors = mulNode->GetAllInDataAnchors();
  std::vector<ge::GeTensorPtr> mulWeights;
  for (auto in_anchor : in_anchors) {
    auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr)
      continue;

    auto in_node = out_anchor->GetOwnerNode();
    string nodeType = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(in_node);
    if (nodeType == "Const" || nodeType == "Constant") {
      ge::OpDescPtr opDesc = in_node->GetOpDesc();
      ge::GeTensorPtr weight = nullptr;
      ge::AttrUtils::MutableTensor(opDesc, ge::ATTR_NAME_WEIGHTS, weight);
      mulWeights.push_back(weight);
    }
  }

  if (mulWeights.size() != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "mul node need 1 weights, but now is %d.", (int)mulWeights.size());
    return NOT_CHANGED;
  }

  ge::GeTensorPtr alpha = mulWeights[0];
  FUSION_PASS_CHECK(alpha == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "alpha is null, fusion failed."),
                    return PARAM_INVALID);
  if (alpha->GetTensorDesc().GetShape().GetDimNum() != 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "mul op weight should be scalar.but now is %d-D.",
            (int)alpha->GetTensorDesc().GetShape().GetDimNum());
    return NOT_CHANGED;
  }
  if (alpha->GetTensorDesc().GetDataType() != ge::DT_FLOAT && alpha->GetTensorDesc().GetDataType() != ge::DT_FLOAT16) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "alpha type float or fp16, but now is %d.",
            (int)alpha->GetTensorDesc().GetDataType());
    return NOT_CHANGED;
  }

  ge::DataType alphaType = alpha->GetTensorDesc().GetDataType();
  FUSION_PASS_CHECK(alpha->GetData().data() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "alpha->GetData().data() is null, fusion failed."),
                    return FAILED);
  if (alphaType == ge::DT_FLOAT) {
    float negative_slope = (*((float*)alpha->GetData().data()));
    if (negative_slope >= 1.0) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "leaky_relu alpha need < 1.0,but now is %f.", negative_slope);
      return NOT_CHANGED;
    }
    FUSION_PASS_CHECK(
        !ge::AttrUtils::SetFloat(mulNode->GetOpDesc(), ge::ATTR_NAME_NEGATIVE_SLOPE, negative_slope),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Set negative slope failed, exit MulMaximumFusionPass with status: failed."),
        return FAILED);
  } else if (alphaType == ge::DT_FLOAT16) {
    float negative_slope = (*((fp16_t*)alpha->GetData().data())).toFloat();
    if (negative_slope >= 1.0) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "leaky_relu alpha need < 1.0,but now is %f.", negative_slope);
      return NOT_CHANGED;
    }
    FUSION_PASS_CHECK(
        !ge::AttrUtils::SetFloat(mulNode->GetOpDesc(), ge::ATTR_NAME_NEGATIVE_SLOPE, negative_slope),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Set negative slope failed, exit MulMaximumFusionPass with status: failed."),
        return FAILED);
  }

  for (auto inAnchor : maximumNode->GetAllInDataAnchors()) {
    if (inAnchor == nullptr || inAnchor->GetPeerOutAnchor() == nullptr ||
        inAnchor->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc()->GetType() == "Mul")
      continue;
    inAnchor->UnlinkAll();
  }
  uint32_t const_input_index = 0;
  for (uint32_t index = 0; index < mulNode->GetAllInDataAnchors().size(); index++) {
    ge::NodePtr input_node_const = mulNode->GetInDataAnchor(index)->GetPeerOutAnchor()->GetOwnerNode();
    string nodeType = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(input_node_const);
    if (nodeType == "Const" || nodeType == "Constant") {
      const_input_index = index;
      break;
    }
  }
  ge::InDataAnchorPtr mulAnchorPtr1 = mulNode->GetInDataAnchor(const_input_index);
  ge::OutDataAnchorPtr constAnchorPtr1 = mulAnchorPtr1->GetPeerOutAnchor();
  ge::NodePtr constNode1 = constAnchorPtr1->GetOwnerNode();
  ge::GraphUtils::RemoveEdge(constAnchorPtr1, mulAnchorPtr1);
  ge::NodeUtils::ClearInDataAnchor(mulNode, mulAnchorPtr1);
  ge::OpDescUtils::ClearInputDesc(mulNode->GetOpDesc(), const_input_index);
  if (PatternFusionUtil::GetOutEdgeSize(constNode1) == 0) {
    FUSION_PASS_CHECK(graph.RemoveNode(constNode1) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove Node[%s] failed", constNode1->GetName().c_str()),
                      return FAILED);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Remove const Node:[%s].", constNode1->GetName().c_str());
  }

  mulNode->GetOpDesc()->SetType("LeakyRelu");

  mulWeights.clear();
  ge::OpDescUtils::SetWeights(mulNode, mulWeights);

  if (maximumNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    for (ge::InDataAnchorPtr inAnchorPtr : maximumNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(mulNode->GetOutDataAnchor(0), inAnchorPtr),
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index to fusion node:%s's 1st index failed.",
                  mulNode->GetName().c_str(), maximumNode->GetName().c_str()),
          return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's 1st index to fusion node:%s's 1st index.",
              mulNode->GetName().c_str(), maximumNode->GetName().c_str());
    }
  }

  maximumNode->GetOutDataAnchor(0)->UnlinkAll();
  FUSION_PASS_CHECK(graph.RemoveNode(maximumNode) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove maximum node failed."), return FAILED);

  fusionNodes.push_back(mulNode);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define MulMaximumFusionPass fusion end");
  return SUCCESS;
}

REGISTER_PASS("MulMaximumFusionPass", BUILT_IN_GRAPH_PASS, MulMaximumFusionPass);
}  // namespace fe
