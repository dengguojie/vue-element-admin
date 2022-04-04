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
 * \file softmax_cross_entropy_loss_fusion_pass.cpp
 * \brief softmaxcrossentropyloss fusion pass
 */
#include "softmax_cross_entropy_loss_fusion_pass.h"
#include "tbe_ops_pass_util.h"
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "op_log.h"
#include "error_util.h"
#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "external/graph/operator_factory.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

namespace fe {
static const string PATTERN_FUSEDNODE = "SoftmaxCrossEntropyLoss";
static const string FUSED_NODE = "SoftmaxCrossEntropyLoss";

vector<FusionPattern*> SoftmaxCrossEntropyLossFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("SoftmaxCrossEntropyLossFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

Status SoftmaxCrossEntropyLossFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                 vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SoftmaxCrossEntropyLossFusionPass fusion begin.");
  ge::NodePtr originNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(originNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "originNode is null, fusion failed."),
                    return PARAM_INVALID);
  auto originDesc = originNode->GetOpDesc();
  FUSION_PASS_CHECK(originDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "originNode get output failed."),
                    return PARAM_INVALID);

  FUSION_PASS_CHECK(originDesc->GetInputsSize() < 2,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "originNode input size small than 2"),
                    return PARAM_INVALID);

  Operator op = ge::OpDescUtils::CreateOperatorFromNode(originNode);
  string reduction = "reduction";
  if (GRAPH_SUCCESS != op.GetAttr("reduction", reduction)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Get reduction failed.");
  }

  std::string op_name("GatherElements");
  auto gather_elements_op = ge::OperatorFactory::CreateOperator(op_name.c_str(), "GatherElements");
  FUSION_PASS_CHECK(gather_elements_op.IsEmpty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "Create gather_elements_op error"),
                    return FAILED);
  auto gather_elements_desc = ge::OpDescUtils::GetOpDescFromOperator(gather_elements_op);
  gather_elements_op.BreakConnect();

  auto input_scores_desc = originDesc->GetInputDesc("scores").Clone();
  gather_elements_desc->UpdateInputDesc(0, input_scores_desc);
  originDesc->UpdateOutputDesc(0, input_scores_desc);

  auto input_labels_desc = originDesc->GetInputDesc("labels").Clone();
  std::vector<int64_t> data;
  std::vector<int64_t> labels_shape = input_labels_desc.GetShape().GetDims();
  for (size_t i = 0; i < labels_shape.size(); ++i) {
    data.push_back(i);
  }
  labels_shape.insert(labels_shape.begin() + 1, 1);
  ge::GeShape shape_2(labels_shape);
  input_labels_desc.SetShape(shape_2);
  input_labels_desc.SetOriginShape(shape_2);
  gather_elements_desc->UpdateInputDesc(1, input_labels_desc);

  auto output_desc = originDesc->GetInputDesc("labels").Clone();
  DataType dataType = input_scores_desc.GetDataType();
  output_desc.SetDataType(dataType);
  output_desc.SetOriginDataType(dataType);
  gather_elements_desc->UpdateOutputDesc(0, output_desc);

  ge::NodePtr new_node = graph.AddNode(gather_elements_desc);
  FUSION_PASS_CHECK(new_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to create GatherElements node"),
                    return FAILED);

  // get op
  Operator op_pool = ge::OpDescUtils::CreateOperatorFromNode(new_node);
  op_pool.SetAttr("dim", 1);
  OP_LOGE(FUSED_OP_TYPE.c_str(), "get attr dim failed.");
  FUSION_PASS_CHECK(
      SUCCESS !=
          ge::GraphUtils::AddEdge(originNode->GetInDataAnchor(1)->GetPeerOutAnchor(), new_node->GetInDataAnchor(1)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add labels input edge to gather_elements failed."),
      return FAILED);
  if (reduction == "none") {
    for (auto in_anchor : originNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      in_anchor->UnlinkAll();
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(new_node->GetOutDataAnchor(0), in_anchor),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add  gather_elements  edge failed."),
                        return FAILED);
    }
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(originNode->GetOutDataAnchor(0), new_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add originNode edge to new_node failed."),
                      return FAILED);
  } else if (reduction == "sum") {
    std::string op_name("ReduceSumD");
    auto reduce_sum_op = ge::OperatorFactory::CreateOperator(op_name.c_str(), "ReduceSumD");
    FUSION_PASS_CHECK(reduce_sum_op.IsEmpty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "Create reduce_sum_op error"),
                      return FAILED);
    auto reduce_sum_desc = ge::OpDescUtils::GetOpDescFromOperator(reduce_sum_op);
    reduce_sum_op.BreakConnect();

    auto input_x_desc = originDesc->GetInputDesc("labels").Clone();
    input_x_desc.SetDataType(dataType);
    input_x_desc.SetOriginDataType(dataType);
    reduce_sum_desc->UpdateInputDesc(0, input_x_desc);

    auto input_desc = originDesc->GetInputDesc("scores").Clone();
    std::vector<int64_t> out = {1};
    ge::GeShape output_shape(out);
    input_desc.SetShape(output_shape);
    reduce_sum_desc->UpdateOutputDesc(0, input_desc);

    ge::NodePtr reduce_sum_node = graph.AddNode(reduce_sum_desc);
    FUSION_PASS_CHECK(reduce_sum_node == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to create reduce_sum node node"),
                      return FAILED);

    // get op
    Operator op_sum = ge::OpDescUtils::CreateOperatorFromNode(reduce_sum_node);
    op_sum.SetAttr("axes", data);
    OP_LOGE(FUSED_OP_TYPE.c_str(), "get attr axes failed.");
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(new_node->GetOutDataAnchor(0), reduce_sum_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add new_node edge to reduce_sum_node failed."),
        return FAILED);
    for (auto in_anchor : originNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      in_anchor->UnlinkAll();
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(reduce_sum_node->GetOutDataAnchor(0), in_anchor),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add  reduce_sum  edge failed."),
                        return FAILED);
    }
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(originNode->GetOutDataAnchor(0), new_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add originNode edge to new_node failed."),
                      return FAILED);
  } else if (reduction == "mean") {
    std::string op_name("ReduceMeanD");
    auto reduce_mean_op = ge::OperatorFactory::CreateOperator(op_name.c_str(), "ReduceMeanD");
    FUSION_PASS_CHECK(reduce_mean_op.IsEmpty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "Create reduce_mean_op error"),
                      return FAILED);
    auto reduce_mean_desc = ge::OpDescUtils::GetOpDescFromOperator(reduce_mean_op);
    reduce_mean_op.BreakConnect();

    auto input_x_desc = originDesc->GetInputDesc("labels").Clone();
    input_x_desc.SetDataType(dataType);
    input_x_desc.SetOriginDataType(dataType);
    reduce_mean_desc->UpdateInputDesc(0, input_x_desc);

    auto input_desc = originDesc->GetInputDesc("scores").Clone();
    std::vector<int64_t> out = {1};
    ge::GeShape output_shape(out);
    input_desc.SetShape(output_shape);
    reduce_mean_desc->UpdateOutputDesc(0, input_desc);

    ge::NodePtr reduce_mean_node = graph.AddNode(reduce_mean_desc);
    FUSION_PASS_CHECK(reduce_mean_node == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to create reduce_sum node node"),
                      return FAILED);

    // get op
    Operator op_mean = ge::OpDescUtils::CreateOperatorFromNode(reduce_mean_node);
    op_mean.SetAttr("axes", data);
    OP_LOGE(FUSED_OP_TYPE.c_str(), "get attr axes failed.");
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(new_node->GetOutDataAnchor(0), reduce_mean_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add new_node edge to reduce_sum_node failed."),
        return FAILED);
    for (auto in_anchor : originNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      in_anchor->UnlinkAll();
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(reduce_mean_node->GetOutDataAnchor(0), in_anchor),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add  reduce_mean  edge failed."),
                        return FAILED);
    }
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(originNode->GetOutDataAnchor(0), new_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add originNode edge to new_node failed."),
                      return FAILED);
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define SoftmaxCrossEntropyLossFusionPass fusion end");

  return SUCCESS;
}

REGISTER_PASS("SoftmaxCrossEntropyLossFusionPass", BUILT_IN_GRAPH_PASS, SoftmaxCrossEntropyLossFusionPass);
}  // namespace fe
