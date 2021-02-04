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

#include "max_fusion_pass.h"

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
const std::string MaxToMaximumPass::PATTERN_FUSEDNODE = "MaxN";
const int MaxToMaximumPass::MIN_INPUT_SIZE = 2;
vector<FusionPattern*> MaxToMaximumPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern =
      (new (std::nothrow) FusionPattern("MaxToMaximumPass"));
  FUSION_PASS_CHECK(pattern == nullptr,
                    OP_LOGE("MaxToMaximumPass", "new pattern error"),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {"MaxN"}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

Status MaxToMaximumPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                vector<ge::NodePtr>& fusion_nodes) {
  ge::NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fused_node == nullptr, OP_LOGE("MaxToMaximumPass", "Fusion GetNode Error"), return PARAM_INVALID);
  ge::OpDescPtr op_desc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(op_desc == nullptr, OP_LOGE("MaxToMaximumPass", "Fusion GetOpdesc Error"), return PARAM_INVALID);

  int input_size = op_desc->GetInputsSize();
  FUSION_PASS_CHECK(input_size < MIN_INPUT_SIZE, OP_LOGW("MaxToMaximumPass", "input_size less %d", MIN_INPUT_SIZE),
                    return PARAM_INVALID);
  GetDataFormatAndType(fused_node);
  auto ret = FusionNode(graph, fused_node, fusion_nodes, input_size);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE("MaxToMaximumPass", "FusionNode FAIL"), return FAILED);
  ret = RemoveFusedNode(graph, fused_node);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE("MaxToMaximumPass", "AddEdgeForList FAIL"), return FAILED);
  return SUCCESS;
}

Status MaxToMaximumPass::FusionNode(ge::ComputeGraph& graph,
                                    ge::NodePtr& fused_node,
                                    vector<ge::NodePtr>& fusion_nodes,
                                    const int input_size) const {
  ge::NodePtr root_node = nullptr;
  ge::NodePtr curr_node = nullptr;
  ge::NodePtr new_node = nullptr;
  CreateNode(graph, root_node, 0);
  UpdateDescForRoot(fused_node, root_node);
  FUSION_PASS_CHECK(root_node == nullptr, OP_LOGE("MaxToMaximumPass", "Fusion Create root node Error"),
                    return PARAM_INVALID);
  fusion_nodes.push_back(root_node);
  auto ret = AddEdgeForRoot(fused_node, root_node);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE("MaxToMaximumPass", "AddEdgeForRoot FAIL"), return PARAM_INVALID);
  curr_node = root_node;
  for (int i = 2; i < input_size; ++i) {
    CreateNode(graph, new_node, i);
    FUSION_PASS_CHECK(new_node == nullptr, OP_LOGE("MaxToMaximumPass", "Fusion Create new node Error"),
                      return PARAM_INVALID);
    fusion_nodes.push_back(new_node);
    ret = UpdateDescForList(fused_node, new_node, curr_node, i);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE("MaxToMaximumPass", "UpdateDescForList FAIL"), return PARAM_INVALID);
    ret = AddEdgeForList(fused_node, new_node, curr_node, i);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE("MaxToMaximumPass", "AddEdgeForList FAIL"), return PARAM_INVALID);
    curr_node = new_node;
  }

  ret = AddEdgeForEnd(fused_node, curr_node);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE("MaxToMaximumPass", "AddEdgeForList FAIL"), return PARAM_INVALID);
  return SUCCESS;
}

Status MaxToMaximumPass::CreateNode(ge::ComputeGraph& graph,
                                    ge::NodePtr& new_node,
                                    const int index) const {
  ge::OpDescPtr new_desc = nullptr;
  std::string name = "Maximum_Fuss_" + to_string(index);
  FUSION_PASS_MAKE_SHARED((new_desc = std::make_shared<ge::OpDesc>(name, "Maximum")), return INTERNAL_ERROR);
  auto ret = new_desc->AddInputDesc(GeTensorDesc(GeShape(), data_format, data_type));
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE("MaxToMaximumPass", "AddInputDesc first FAIL"), return FAILED);
  ret = new_desc->AddInputDesc(GeTensorDesc(GeShape(), data_format, data_type));
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE("MaxToMaximumPass", "AddInputDesc second FAIL"), return FAILED);

  ret = new_desc->AddOutputDesc(GeTensorDesc(GeShape(), data_format, data_type));
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE("MaxToMaximumPass", "AddOutputDesc FAIL"), return FAILED);
  new_node = graph.AddNode(new_desc);
  return SUCCESS;
}

Status MaxToMaximumPass::AddEdgeForRoot(ge::NodePtr& fused_node,
                                        ge::NodePtr& new_node) const {
  auto in_data_anchor = fused_node->GetInDataAnchor(0);
  auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  auto in_data_anchor_n = new_node->GetInDataAnchor(0);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(peer_out_anchor, in_data_anchor_n) != SUCCESS,
                    OP_LOGE("MaxToMaximumPass", "AddEdgeForRoot first error"), return FAILED);
  auto in_data_anchor1 = fused_node->GetInDataAnchor(1);
  auto peer_out_anchor1 = in_data_anchor1->GetPeerOutAnchor();
  auto in_data_anchor1n = new_node->GetInDataAnchor(1);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(peer_out_anchor1, in_data_anchor1n) != SUCCESS,
                    OP_LOGE("MaxToMaximumPass", "AddEdgeForRoot second error"), return FAILED);
  return SUCCESS;
}

Status MaxToMaximumPass::AddEdgeForList(ge::NodePtr& fused_node,
                                        ge::NodePtr& new_node,
                                        ge::NodePtr& curr_node,
                                        const int index) const {
  FUSION_PASS_CHECK(index < 0, OP_LOGE("MaxToMaximumPass", "AddEdgeForList index less 0"), return FAILED);
  auto in_data_anchor = fused_node->GetInDataAnchor(index);
  auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  auto in_data_anchorn = new_node->GetInDataAnchor(0);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(peer_out_anchor, in_data_anchorn) != SUCCESS,
                    OP_LOGE("MaxToMaximumPass", "AddEdgeForList first error"), return FAILED);
  auto curr_outdata_anchor = curr_node->GetOutDataAnchor(0);
  auto new_indata_anchor1 = new_node->GetInDataAnchor(1);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(curr_outdata_anchor, new_indata_anchor1) != SUCCESS,
                    OP_LOGE("MaxToMaximumPass", "AddEdgeForList second error"), return FAILED);
  return SUCCESS;
}

Status MaxToMaximumPass::AddEdgeForEnd(ge::NodePtr& fused_node,
                                       ge::NodePtr& endNode) const {
  auto out_data_anchor = fused_node->GetOutDataAnchor(0);
  auto out_data_anchor_end = endNode->GetOutDataAnchor(0);
  for (auto in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    if (in_data_anchor != nullptr) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(out_data_anchor, in_data_anchor) != SUCCESS,
                        OP_LOGE("MaxToMaximumPass", "RemoveEdge fail"), return FAILED);
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(out_data_anchor_end, in_data_anchor) != SUCCESS,
          OP_LOGE("MaxToMaximumPass", "AddEdgeForEnd second error"), return FAILED);
    }
  }
  return SUCCESS;
}

Status MaxToMaximumPass::RemoveFusedNode(ge::ComputeGraph& graph,
                                         ge::NodePtr& fused_node) const {
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

  FUSION_PASS_CHECK(graph.RemoveNode(fused_node) != SUCCESS, OP_LOGE("MaxToMaximumPass", "RemoveFusedNode error"),
                    return FAILED);
  return SUCCESS;
}

void MaxToMaximumPass::GetDataFormatAndType(ge::NodePtr& fused_node) {
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fused_node);
  auto input_desc = op.GetDynamicInputDesc("x", 0);
  data_format = input_desc.GetFormat();
  data_type = input_desc.GetDataType();
}

Status MaxToMaximumPass::UpdateDescForRoot(ge::NodePtr& fused_node,
                                           ge::NodePtr& new_node) const {
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fused_node);
  auto input_desc = op.GetDynamicInputDesc("x", 0);
  auto input_desc1 = op.GetDynamicInputDesc("x", 1);
  ge::GeShape input_shape(input_desc.GetShape().GetDims());
  ge::GeShape input_shape1(input_desc1.GetShape().GetDims());
  ge::OpDescPtr op_desc = new_node->GetOpDesc();
  FUSION_PASS_CHECK(op_desc == nullptr, OP_LOGE("MaxToMaximumPass", "UpdateDescForRoot GetOpDesc error"), return FAILED);
  ge::GeTensorDesc new_input_desc = op_desc->GetInputDesc(0);
  ge::GeTensorDesc new_input_desc1 = op_desc->GetInputDesc(1);
  ge::GeTensorDesc new_output_desc = op_desc->GetOutputDesc(0);
  new_input_desc.SetShape(input_shape);
  new_input_desc1.SetShape(input_shape1);
  op_desc->UpdateInputDesc(0, new_input_desc);
  op_desc->UpdateInputDesc(1, new_input_desc1);
  std::vector<int64_t> dims(1, 0);
  GetOutputShape(dims, input_shape);
  GetOutputShape(dims, input_shape1);
  ge::GeShape infer_shape(dims);
  new_output_desc.SetShape(infer_shape);
  op_desc->UpdateOutputDesc(0, new_output_desc);
  return SUCCESS;
}

Status MaxToMaximumPass::UpdateDescForList(ge::NodePtr& fused_node,
                                           ge::NodePtr& new_node,
                                           ge::NodePtr& curr_node,
                                           const int index) const {
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fused_node);
  auto input_desc = op.GetDynamicInputDesc("x", index);
  ge::GeShape input_shape(input_desc.GetShape().GetDims());
  ge::OpDescPtr op_desc = new_node->GetOpDesc();
  FUSION_PASS_CHECK(op_desc == nullptr, OP_LOGE("MaxToMaximumPass", "UpdateDescForList GetOpDesc error"), return FAILED);
  ge::GeTensorDesc new_input_desc = op_desc->GetInputDesc(0);
  ge::GeTensorDesc new_input_desc1 = op_desc->GetInputDesc(1);
  ge::GeTensorDesc new_output_desc = op_desc->GetOutputDesc(0);
  ge::OpDescPtr curr_op_desc = curr_node->GetOpDesc();
  FUSION_PASS_CHECK(curr_op_desc == nullptr, OP_LOGE("MaxToMaximumPass", "UpdateDescForList GetOpDesc error"),
                    return FAILED);
  ge::GeShape input_shape1 = curr_op_desc->GetOutputDesc(0).GetShape();
  new_input_desc.SetShape(input_shape);
  new_input_desc1.SetShape(input_shape1);
  op_desc->UpdateInputDesc(0, new_input_desc);
  op_desc->UpdateInputDesc(1, new_input_desc1);
  std::vector<int64_t> dims(1, 0);
  GetOutputShape(dims, input_shape);
  GetOutputShape(dims, input_shape1);
  ge::GeShape infer_shape(dims);
  new_output_desc.SetShape(infer_shape);
  op_desc->UpdateOutputDesc(0, new_output_desc);
  return SUCCESS;
}

void MaxToMaximumPass::GetOutputShape(std::vector<int64_t>& dims,
                                      const ge::GeShape input_shape) const {
  int32_t dims_size = dims.size();
  std::vector<int64_t> input_dims = input_shape.GetDims();
  int32_t input_dims_size = input_dims.size();
  if (input_dims_size > dims_size) {
    for (int i = 0; i < input_dims_size - dims_size; ++i) {
      dims.insert(dims.begin(), 0);
    }
    dims_size = dims.size();
  }
  int32_t i = dims_size - input_dims_size;
  int32_t j = 0;
  while (i < dims_size && j < input_dims_size) {
    if (dims[i] < input_dims[j]) {
      dims[i] = input_dims[j];
    }
    i++;
    j++;
  }
}

REGISTER_PASS("MaxToMaximumPass", BUILT_IN_GRAPH_PASS, MaxToMaximumPass);
}  //  namespace fe
