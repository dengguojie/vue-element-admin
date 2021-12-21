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
 * \file a_a_square_sum_maximum_rsqrt_mul_fusion_pass.cpp
 * \brief square sum maximum rsqrt mul fusion pass
 */
#include "a_a_square_sum_maximum_rsqrt_mul_fusion_pass.h"

#include <iostream>
#include <vector>
#include <map>

#include "pattern_fusion_util.h"
#include "op_log.h"
#include "error_util.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "fp16_t.hpp"

namespace fe {
static const string SQUARE = "Square";
static const string REDUCESUM = "ReduceSum";
static const string MAXIMUM = "Maximum";
static const string RSQRT = "Rsqrt";
static const string MUL = "Mul";
static const string PATTERN_SQUARE = "Square";
static const string PATTERN_SUM = "Sum";
static const string PATTERN_MAXIMUM = "Maximum";
static const string PATTERN_RSQRT = "Rsqrt";
static const string PATTERN_MUL = "Mul";
static const string PATTERN_INPUT = "input";
static const string L2_NORMALIZE_ATTR_AXIS = "axis";

/*  tf.nn.l2_normalize are fused by square, sum, maximum, rsqrt and mul
    axis as a constant input for sum, eps as a constant input for maximum
                   x
                 /   \
                |  square
                |    |
                |   sum <----const op (reduction_indices axis of l2_normalize)
                |    |
                |  maximum <----const op (parameter eps of l2_normalize)
                |    |
                |  rsqrt
                \   /
                 \ /
                 mul
*/

vector<FusionPattern*> AASquareSumMaximumRsqrtMulFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define AASquareSumMaximumRsqrtMulFusionPass pattern begin");
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("AASquareSumMaximumRsqrtMulFusion");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_SQUARE, {SQUARE})
      .AddOpDesc(PATTERN_SUM, {REDUCESUM})
      .AddOpDesc(PATTERN_MAXIMUM, {MAXIMUM})
      .AddOpDesc(PATTERN_RSQRT, {RSQRT})
      .AddOpDesc(PATTERN_MUL, {MUL})
      .AddOpDesc(PATTERN_INPUT)
      .SetInputs(PATTERN_SQUARE, {PATTERN_INPUT})
      .SetInputs(PATTERN_SUM, {PATTERN_SQUARE})
      .SetInputs(PATTERN_MAXIMUM, {PATTERN_SUM})
      .SetInputs(PATTERN_RSQRT, {PATTERN_MAXIMUM})
      .SetInputs(PATTERN_MUL, {PATTERN_RSQRT, PATTERN_INPUT})
      .SetOutput(PATTERN_MUL);

  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define AASquareSumMaximumRsqrtMulFusionPass pattern end");
  return patterns;
}

Status AASquareSumMaximumRsqrtMulFusionPass::CheckPeerAllInDataAnchors(const ge::OutDataAnchorPtr& outputAnchor,
                                                                       const size_t& expectedNum) {
  FUSION_PASS_CHECK(outputAnchor == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "outputAnchor must not be null"),
                    return PARAM_INVALID);
  if (outputAnchor->GetPeerInDataAnchors().size() == expectedNum) {
    return SUCCESS;
  }
  return FAILED;
}

Status AASquareSumMaximumRsqrtMulFusionPass::IsMatch(ge::NodePtr& squareNode, ge::NodePtr& sumNode,
                                                     ge::NodePtr& maximumNode, ge::NodePtr& rsqrtNode,
                                                     ge::NodePtr& mulNode) {
  FUSION_PASS_CHECK(squareNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "squareNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(sumNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "sumNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(maximumNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "maximumNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(rsqrtNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "rsqrtNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mulNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(
      CheckPeerAllInDataAnchors(squareNode->GetOutDataAnchor(0), 1) != SUCCESS,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "%s contains more than one peer input", squareNode->GetName().c_str()),
      return NOT_CHANGED);
  FUSION_PASS_CHECK(CheckPeerAllInDataAnchors(sumNode->GetOutDataAnchor(0), 1) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "%s contains more than one peer input", sumNode->GetName().c_str()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(
      CheckPeerAllInDataAnchors(maximumNode->GetOutDataAnchor(0), 1) != SUCCESS,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "%s contains more than one peer input", maximumNode->GetName().c_str()),
      return NOT_CHANGED);
  FUSION_PASS_CHECK(
      CheckPeerAllInDataAnchors(rsqrtNode->GetOutDataAnchor(0), 1) != SUCCESS,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "%s contains more than one peer input", rsqrtNode->GetName().c_str()),
      return NOT_CHANGED);

  size_t square_in_nodes_size = ge::OpDescUtils::GetNonConstInputsSize(squareNode);
  if (square_in_nodes_size != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "square node need 1 inputs,but now is %d.", (int)square_in_nodes_size);
    return NOT_CHANGED;
  }
  size_t rsqrt_in_nodes_size = ge::OpDescUtils::GetNonConstInputsSize(rsqrtNode);
  if (rsqrt_in_nodes_size != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "rsqrt node need 1 inputs,but now is %d.", (int)rsqrt_in_nodes_size);
    return NOT_CHANGED;
  }
  size_t mul_in_nodes_size = ge::OpDescUtils::GetNonConstInputsSize(mulNode);
  if (mul_in_nodes_size != 2) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "mul node need 2 inputs,but now is %d.", (int)mul_in_nodes_size);
    return NOT_CHANGED;
  }
  auto sum_in_anchors = sumNode->GetAllInDataAnchors();
  std::vector<ge::GeTensorPtr> sumWeights;
  for (auto &sum_in_anchor : sum_in_anchors) {
    FUSION_PASS_CHECK(sum_in_anchor == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "sum_in_anchor is null, fusion failed."),
                      return PARAM_INVALID);

    auto sum_out_anchor = sum_in_anchor->GetPeerOutAnchor();
    if (sum_out_anchor == nullptr)
      continue;

    auto sum_in_node = sum_out_anchor->GetOwnerNode();
    std::string type = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(sum_in_node);
    if (type == "Const" || type == "Constant") {
      ge::OpDescPtr sumOpDesc = sum_in_node->GetOpDesc();
      ge::GeTensorPtr weight = nullptr;
      ge::AttrUtils::MutableTensor(sumOpDesc, ge::ATTR_NAME_WEIGHTS, weight);
      sumWeights.push_back(weight);
    }
  }
  size_t sum_in_nodes_size = sumNode->GetOpDesc()->GetInputsSize();
  size_t sum_nonconst_size = sum_in_nodes_size - sumWeights.size();
  if (sum_nonconst_size != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "sum node need 1 inputs,but now is %d.", (int)sum_nonconst_size);
    return NOT_CHANGED;
  }
  auto maximum_in_anchors = maximumNode->GetAllInDataAnchors();
  std::vector<ge::GeTensorPtr> maximumWeights;
  for (auto &in_anchor : maximum_in_anchors) {
    auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr)
      continue;

    auto maximum_in_node = out_anchor->GetOwnerNode();
    std::string type = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(maximum_in_node);
    if (type == "Const" || type == "Constant") {
      ge::OpDescPtr maximumOpDesc = maximum_in_node->GetOpDesc();
      ge::GeTensorPtr weight = nullptr;
      ge::AttrUtils::MutableTensor(maximumOpDesc, ge::ATTR_NAME_WEIGHTS, weight);
      maximumWeights.push_back(weight);
    }
  }
  size_t maximum_in_nodes_size = maximumNode->GetOpDesc()->GetInputsSize();
  size_t maximum_nonconst_size = maximum_in_nodes_size - maximumWeights.size();
  if (maximum_nonconst_size != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "maximum node need 1 inputs,but now is %d.", (int)maximum_nonconst_size);
    return NOT_CHANGED;
  }

  ge::NodePtr squareInputNode = squareNode->GetInDataNodes().at(0);
  FUSION_PASS_CHECK(squareInputNode == nullptr,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "squareInputNode is null, fusion failed."), return PARAM_INVALID);

  for (auto &mulInputNode : mulNode->GetInDataNodes()) {
    int64_t mulInputNodeId = mulInputNode->GetOpDesc()->GetId();
    int64_t squareInputNodeId = squareInputNode->GetOpDesc()->GetId();
    if (mulInputNodeId == squareInputNodeId &&
        mulInputNode->GetOpDesc()->GetType() == squareInputNode->GetOpDesc()->GetType() &&
        mulInputNode->GetOpDesc()->GetName() == squareInputNode->GetOpDesc()->GetName()) {
      return SUCCESS;
    }
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "l2norm square.input_node.not same to mul.input_node.can not fusion.");
  return NOT_CHANGED;
}

Status AASquareSumMaximumRsqrtMulFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                    vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define AASquareSumMaximumRsqrtMulFusionPass fusion begin");
  ge::NodePtr squareNode = GetNodeFromMapping(PATTERN_SQUARE, mapping);
  ge::NodePtr sumNode = GetNodeFromMapping(PATTERN_SUM, mapping);
  ge::NodePtr maximumNode = GetNodeFromMapping(PATTERN_MAXIMUM, mapping);
  ge::NodePtr rsqrtNode = GetNodeFromMapping(PATTERN_RSQRT, mapping);
  ge::NodePtr mulNode = GetNodeFromMapping(PATTERN_MUL, mapping);
  FUSION_PASS_CHECK(squareNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "squareNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(sumNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "sumNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(maximumNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "maximumNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(rsqrtNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "rsqrtNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(mulNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulNode is null, fusion failed."),
                    return PARAM_INVALID);

  if (IsMatch(squareNode, sumNode, maximumNode, rsqrtNode, mulNode) != SUCCESS) {
    return NOT_CHANGED;
  }

  auto sum_in_anchors = sumNode->GetAllInDataAnchors();
  std::vector<ge::GeTensorPtr> sumWeights;
  for (auto &in_anchor : sum_in_anchors) {
    auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr)
      continue;

    auto in_node = out_anchor->GetOwnerNode();
    std::string type = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(in_node);
    if (type == "Const" || type == "Constant") {
      ge::OpDescPtr opDesc = in_node->GetOpDesc();
      ge::GeTensorPtr weight = nullptr;
      ge::AttrUtils::MutableTensor(opDesc, ge::ATTR_NAME_WEIGHTS, weight);
      sumWeights.push_back(weight);
    }
  }

  if (sumWeights.size() != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "sum node need 1 weights, but now is %d.", (int)sumWeights.size());
    return NOT_CHANGED;
  }

  ge::GeTensorPtr axis_w = sumWeights[0];
  FUSION_PASS_CHECK(axis_w == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "axis_w is null, fusion failed."),
                    return NOT_CHANGED);

  if (axis_w->GetTensorDesc().GetDataType() != ge::DT_INT32 && axis_w->GetTensorDesc().GetDataType() != ge::DT_INT64) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "axis_w type int32 or int64,but now is %d.",
            (int)axis_w->GetTensorDesc().GetDataType());
    return NOT_CHANGED;
  }

  size_t size = 0;
  size_t sizeTotal = axis_w->GetData().GetSize();
  ge::DataType axisType = axis_w->GetTensorDesc().GetDataType();
  if (axisType == ge::DT_INT32) {
    const int32_t* const_data_ptr = (int32_t*)(axis_w->GetData().GetData());
    FUSION_PASS_CHECK(const_data_ptr == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "const_data_ptr is null, fusion failed."), return PARAM_INVALID);
    size = sizeTotal / sizeof(int32_t);

    std::vector<int32_t> const_data;
    for (size_t i = 0; i < size; ++i) {
      int32_t tmp = (int32_t)((*(const_data_ptr + i)));
      const_data.push_back(tmp);

      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node:%s const data int32 proto %d", mulNode->GetOpDesc()->GetName().c_str(), tmp);
    }
    ge::AttrUtils::SetListInt(mulNode->GetOpDesc(), ge::L2_NORMALIZE_ATTR_AXIS, const_data);
  } else if (axisType == ge::DT_INT64) {
    const int64_t* const_data_ptr = (int64_t*)(axis_w->GetData().GetData());
    FUSION_PASS_CHECK(const_data_ptr == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "const_data_ptr is null, fusion failed."), return PARAM_INVALID);
    size = sizeTotal / sizeof(int64_t);

    std::vector<int64_t> const_data;
    for (size_t i = 0; i < size; ++i) {
      int64_t tmp = (int64_t)((*(const_data_ptr + i)));
      const_data.push_back(tmp);
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node:%s const data int64 proto %d", mulNode->GetOpDesc()->GetName().c_str(), tmp);
    }
    ge::AttrUtils::SetListInt(mulNode->GetOpDesc(), ge::L2_NORMALIZE_ATTR_AXIS, const_data);
  }

  auto maximum_in_anchors = maximumNode->GetAllInDataAnchors();
  std::vector<ge::GeTensorPtr> maximumWeights;
  for (auto &in_anchor : maximum_in_anchors) {
    auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr)
      continue;

    auto in_node = out_anchor->GetOwnerNode();
    std::string type = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(in_node);
    if (type == "Const" || type == "Constant") {
      ge::OpDescPtr opDesc = in_node->GetOpDesc();
      ge::GeTensorPtr weight = nullptr;
      ge::AttrUtils::MutableTensor(opDesc, ge::ATTR_NAME_WEIGHTS, weight);
      maximumWeights.push_back(weight);
    }
  }

  if (maximumWeights.size() != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "mul node need 1 weights, but now is %d.", (int)maximumWeights.size());
    return NOT_CHANGED;
  }

  ge::GeTensorPtr eps_w = maximumWeights[0];
  FUSION_PASS_CHECK(eps_w == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "eps_w is null, fusion failed."),
                    return PARAM_INVALID);

  if (eps_w->GetTensorDesc().GetShape().GetDimNum() != 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "maximum op weight should be scalar.but now is %d-D.",
            (int)eps_w->GetTensorDesc().GetShape().GetDimNum());
    return NOT_CHANGED;
  }

  if (eps_w->GetTensorDesc().GetDataType() != ge::DT_FLOAT && eps_w->GetTensorDesc().GetDataType() != ge::DT_FLOAT16) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "eps_w type float or fp16, but now is %d.",
            (int)eps_w->GetTensorDesc().GetDataType());
    return NOT_CHANGED;
  }

  ge::DataType epsType = eps_w->GetTensorDesc().GetDataType();
  FUSION_PASS_CHECK(eps_w->GetData().data() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "eps_w->GetData().data() is null."),
                    return FAILED);
  if (epsType == ge::DT_FLOAT) {
    float eps = (*((float*)eps_w->GetData().data()));
    OP_LOGI(FUSED_OP_TYPE.c_str(), "L2_norm eps = %f.", eps);
    FUSION_PASS_CHECK(!ge::AttrUtils::SetFloat(mulNode->GetOpDesc(), ge::L2_NORMALIZE_ATTR_EPS, eps),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "set L2 normalize eps failed."), return FAILED);
  } else if (epsType == ge::DT_FLOAT16) {
    float eps = (*((fp16_t*)eps_w->GetData().data())).toFloat();
    OP_LOGI(FUSED_OP_TYPE.c_str(), "L2_norm eps = %f.", eps);
    FUSION_PASS_CHECK(!ge::AttrUtils::SetFloat(mulNode->GetOpDesc(), ge::L2_NORMALIZE_ATTR_EPS, eps),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "set L2 normalize eps failed."), return FAILED);
  }

  uint32_t rsqrt_output_index = 0;
  for (uint32_t index = 0; index < mulNode->GetAllInDataAnchors().size(); index++) {
    ge::NodePtr input_node_rsqrt = mulNode->GetInDataAnchor(index)->GetPeerOutAnchor()->GetOwnerNode();
    if (input_node_rsqrt->GetType() == "Rsqrt") {
      rsqrt_output_index = index;
      break;
    }
  }
  FUSION_PASS_CHECK(rsqrtNode->GetOutDataAnchor(0) == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "rsqrtNode get output failed."),
                    return PARAM_INVALID);

  if (rsqrtNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    for (ge::InDataAnchorPtr &inAnchorPtr : rsqrtNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::RemoveEdge(rsqrtNode->GetOutDataAnchor(0), inAnchorPtr),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove edge from fused node:%s's index to fusion node:%s's 1st index failed.",
                  rsqrtNode->GetName().c_str(), mulNode->GetName().c_str()),
          return FAILED);
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Remove edge from fused node:%s's 1st index to fusion node:%s's 1st index.",
              rsqrtNode->GetName().c_str(), mulNode->GetName().c_str());
    }
    ge::InDataAnchorPtr mul_anchor_ptr1 = mulNode->GetInDataAnchor(rsqrt_output_index);
    ge::NodeUtils::ClearInDataAnchor(mulNode, mul_anchor_ptr1);
    ge::OpDescUtils::ClearInputDesc(mulNode->GetOpDesc(), rsqrt_output_index);
  }

  mulNode->GetOpDesc()->SetType("L2Normalize");

  FUSION_PASS_CHECK(graph.RemoveNode(rsqrtNode) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove rsqrt node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(maximumNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove maximum node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(sumNode) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove sum node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(squareNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove square node failed."), return FAILED);

  fusionNodes.push_back(mulNode);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define AASquareSumMaximumRsqrtMulFusionPass fusion end");
  return SUCCESS;
}

REGISTER_PASS("AASquareSumMaximumRsqrtMulFusionPass", BUILT_IN_GRAPH_PASS, AASquareSumMaximumRsqrtMulFusionPass);
}  // namespace fe
