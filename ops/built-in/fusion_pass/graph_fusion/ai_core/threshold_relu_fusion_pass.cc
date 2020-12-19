/* Copyright (c) Huawei Technologies Co., Ltd. 2012-2020. All rights reserved.
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
 */

#include "threshold_relu_fusion_pass.h"

#include <memory>
#include <vector>

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "tbe_fusion_pass_util.h"
namespace fe {
const std::string ThresholdReluPass::PATTERN_FUSEDNODE = "ThresholdedRelu";
vector<FusionPattern *> ThresholdReluPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern =
      (new (std::nothrow) FusionPattern("ThresholdReluPass"));
  FUSION_PASS_CHECK(pattern == nullptr,
                    OP_LOGE("ThresholdReluPass", "new pattern error"),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {"ThresholdedRelu"})
      .SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

Status ThresholdReluPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                                 vector<ge::NodePtr> &fusion_nodes) {
  ge::NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fused_node == nullptr,
                    OP_LOGE("ThresholdReluPass", "Fusion GetNode Error"),
                    return PARAM_INVALID);
  ge::NodePtr threshold_node = nullptr;
  ge::NodePtr mul_node = nullptr;
  auto ret = CreateThresholdNode(graph, fused_node, threshold_node);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE("ThresholdReluPass", "Fusion fail"),
                    return FAILED);
  ret = CreateMulNode(graph, fused_node, mul_node);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE("ThresholdReluPass", "Fusion CreateMulNode fail"),
                    return FAILED);
  fusion_nodes.push_back(threshold_node);
  fusion_nodes.push_back(mul_node);
  ret = FusionNode(fused_node, threshold_node, mul_node);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE("ThresholdReluPass", "FusionNode fail"),
                    return FAILED);
  ret = RemoveFusedNode(graph, fused_node);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE("ThresholdReluPass", "RemoveFusedNode fail"),
                    return FAILED);
  return SUCCESS;
}

Status ThresholdReluPass::FusionNode(ge::NodePtr &fused_node,
                                     ge::NodePtr &threshold_node,
                                     ge::NodePtr &mul_node) const {
  auto in_data_anchor = fused_node->GetInDataAnchor(0);
  auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  auto ret = ge::GraphUtils::AddEdge(peer_out_anchor,
                                     threshold_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(
      ret != SUCCESS,
      OP_LOGE("ThresholdReluPass", "AddEdg to thresholdNode fail"),
      return FAILED);
  ret = ge::GraphUtils::AddEdge(peer_out_anchor, mul_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE("ThresholdReluPass", "AddEdge to mulNode fail"),
                    return FAILED);
  ret = ge::GraphUtils::AddEdge(threshold_node->GetOutDataAnchor(0),
                                mul_node->GetInDataAnchor(1));
  FUSION_PASS_CHECK(
      ret != SUCCESS,
      OP_LOGE("ThresholdReluPass", "AddEdge to mulNode from threshold fail"),
      return FAILED);
  ret = AddEdgeToMulForOut(fused_node, mul_node);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE("ThresholdReluPass", "AddEdgeToMulForOut fail"),
                    return FAILED);
  return SUCCESS;
}

Status ThresholdReluPass::CreateThresholdNode(ge::ComputeGraph &graph,
                                              ge::NodePtr &fused_node,
                                              ge::NodePtr &new_node) const {
  ge::OpDescPtr new_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((new_desc = std::make_shared<ge::OpDesc>(
                               "Threshold_ThresholdRelu", "Threshold")),
                          return INTERNAL_ERROR);
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fused_node);
  auto input_desc = op.GetInputDesc(0);
  float alpha = 0.0;
  auto ret = op.GetAttr("alpha", alpha);
  FUSION_PASS_CHECK(
      ret != SUCCESS,
      OP_LOGE("ThresholdReluPass", "CreateThresholdNode GetAttr alpha fail"),
      return FAILED);
  ge::GeShape input_shape(input_desc.GetShape().GetDims());
  ge::Format data_format = input_desc.GetFormat();
  ge::DataType data_type = input_desc.GetDataType();
  ret =
      new_desc->AddInputDesc(GeTensorDesc(input_shape, data_format, data_type));
  FUSION_PASS_CHECK(
      ret != SUCCESS,
      OP_LOGE("ThresholdReluPass", "CreateThresholdNode AddInputDesc fail"),
      return FAILED);
  ret = new_desc->AddOutputDesc(
      GeTensorDesc(input_shape, data_format, data_type));
  FUSION_PASS_CHECK(
      ret != SUCCESS,
      OP_LOGE("ThresholdReluPass", "CreateThresholdNode AddOutputDesc fail"),
      return FAILED);
  new_node = graph.AddNode(new_desc);
  Operator new_op = ge::OpDescUtils::CreateOperatorFromNode(new_node);
  new_op.SetAttr("threshold", alpha);
  return SUCCESS;
}

Status ThresholdReluPass::CreateMulNode(ge::ComputeGraph &graph,
                                        ge::NodePtr &fused_node,
                                        ge::NodePtr &new_node) const {
  ge::OpDescPtr new_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (new_desc = std::make_shared<ge::OpDesc>("Mul_For_ThresholdRelu", "Mul")),
      return INTERNAL_ERROR);
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fused_node);
  auto input_desc = op.GetInputDesc(0);
  ge::GeShape input_shape(input_desc.GetShape().GetDims());
  ge::Format data_format = input_desc.GetFormat();
  ge::DataType data_type = input_desc.GetDataType();
  auto ret =
      new_desc->AddInputDesc(GeTensorDesc(input_shape, data_format, data_type));
  FUSION_PASS_CHECK(
      ret != SUCCESS,
      OP_LOGE("ThresholdReluPass", "CreateMulNode AddInputDesc one fail."),
      return FAILED);
  ret =
      new_desc->AddInputDesc(GeTensorDesc(input_shape, data_format, data_type));
  FUSION_PASS_CHECK(
      ret != SUCCESS,
      OP_LOGE("ThresholdReluPass", "CreateMulNode AddinputDesc two fail."),
      return FAILED);
  ret = new_desc->AddOutputDesc(
      GeTensorDesc(input_shape, data_format, data_type));
  FUSION_PASS_CHECK(
      ret != SUCCESS,
      OP_LOGE("ThresholdReluPass", "CreateMulNode AddoutputDesc fail."),
      return FAILED);
  new_node = graph.AddNode(new_desc);
  return SUCCESS;
}

Status ThresholdReluPass::AddEdgeToMulForOut(ge::NodePtr &fused_node,
                                             ge::NodePtr &mul_node) const {
  Status ret = SUCCESS;
  auto out_data_anchor = fused_node->GetOutDataAnchor(0);
  auto peer_indata_anchors = out_data_anchor->GetPeerInDataAnchors();
  for (auto in_data_anchor : peer_indata_anchors) {
    ret = ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(0),
                                     in_data_anchor);
    FUSION_PASS_CHECK(
        ret != SUCCESS,
        OP_LOGE("ThresholdReluPass", "AddEdgeToMulForOut removeEdge fail"),
        return FAILED);
    ret =
        ge::GraphUtils::AddEdge(mul_node->GetOutDataAnchor(0), in_data_anchor);
    FUSION_PASS_CHECK(
        ret != SUCCESS,
        OP_LOGE("ThresholdReluPass", "AddEdgeToMulForOut addEdge fail"),
        return FAILED);
  }
  return SUCCESS;
}

Status ThresholdReluPass::RemoveFusedNode(ge::ComputeGraph &graph,
                                          ge::NodePtr &fused_node) const {
  for (auto in_anchor : fused_node->GetAllInDataAnchors()) {
    if (in_anchor != nullptr) {
      in_anchor->UnlinkAll();
    }
  }

  for (auto out_anchor : fused_node->GetAllOutDataAnchors()) {
    if (out_anchor != nullptr) {
      out_anchor->UnlinkAll();
    }
  }

  FUSION_PASS_CHECK(graph.RemoveNode(fused_node) != SUCCESS,
                    OP_LOGE("ThresholdReluPass", "RemoveFusedNode error"),
                    return FAILED);
  return SUCCESS;
}
REGISTER_PASS("ThresholdReluPass", BUILT_IN_GRAPH_PASS, ThresholdReluPass);
}  //  namespace fe
