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
 * \file unpack_fusion_pass.cpp
 * \brief
 */
#include "unpack_fusion_pass.h"

#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {
static const std::string kPatternFusedNode = "FusedNodeUnpack";
constexpr int64_t kMiniOut{63};
/*
1:
Unapck ---> SplitVD + Unapck

2:
            input                                     input
              |                                         |
           Unpack          ---  -fusion-  --->       SplitVD
      /       |       \                       /         |         \
     /        |        \                     /          |          \
output_1 ... output_m .. output_n        Unpack_1 ... Unpack_M ... Unpack_N
                                       /   |   \     /   |   \    /   |   \
                              output_1  output_2 ... output_m  ...      output_n
*/
Status UnpackFusionPass::AddUnpackOps(OpDescPtr fused_desc, ComputeGraph& graph, vector<NodePtr>& new_nodes,
                                      std::vector<GeTensorDesc> output_desc, const NodePtr fused_node,
                                      const NodePtr splitvd_base_node, const int64_t num, const int64_t axis,
                                      const int64_t i, const int64_t j, const int64_t mini_out) {
  OpDescPtr unpack_desc = AttrUtils::CopyOpDesc(fused_desc);
  unpack_desc->SetName(unpack_desc->GetName() + "/Unpack" + to_string(j));
  unpack_desc->SetType("Unpack");
  for (int64_t c = num - 1; c >= mini_out; c--) {
    OpDescUtils::ClearOutputDesc(unpack_desc, c);
  }
  NodePtr unpack_node = graph.AddNode(unpack_desc);
  new_nodes.push_back(unpack_node);

  AttrUtils::SetInt(unpack_node->GetOpDesc(), "axis", axis);
  AttrUtils::SetInt(unpack_node->GetOpDesc(), "num", mini_out);

  if (static_cast<int64_t>(output_desc.size()) <= j) {
    return FAILED;
  }
  unpack_desc->UpdateInputDesc(0, output_desc[j]);
  for (int64_t h = 0; h < mini_out; h++) {
    GeTensorDesc unpack_output_tensor = unpack_desc->GetOutputDesc(h);
    GeShape unpack_output_shape = unpack_output_tensor.GetShape();
    unpack_output_tensor.SetShape(unpack_output_shape);
    unpack_desc->UpdateOutputDesc(h, unpack_output_tensor);
  }

  FUSION_PASS_CHECK(
      unpack_node == nullptr,
      OP_LOGE(kFusedOpType.c_str(), "The unpack_node:%s is null, fusion failed.", unpack_node->GetName().c_str()),
      return PARAM_INVALID);
  FUSION_PASS_CHECK(
      GraphUtils::AddEdge(splitvd_base_node->GetOutDataAnchor(i), unpack_node->GetInDataAnchor(0)) != SUCCESS,
      OP_LOGE(kFusedOpType.c_str(), "Add edge from fused node:%s's index[%lld] to fusion node:%s's index[%lld] failed.",
              splitvd_base_node->GetName().c_str(), i, unpack_node->GetName().c_str(), i),
      return FAILED);

  for (int64_t m = 0; m < mini_out; m++) {
    FUSION_PASS_CHECK(
        fused_node->GetOutDataAnchor(kMiniOut * i + m) == nullptr,
        OP_LOGE(kFusedOpType.c_str(), "The OutDataAnchor(kMiniOut * i + m) of fused_node is null, fusion failed."),
        return PARAM_INVALID);
    for (InDataAnchorPtr in_anchor_ptr : fused_node->GetOutDataAnchor(kMiniOut * i + m)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(
          GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(kMiniOut * i + m), in_anchor_ptr) != SUCCESS,
          OP_LOGE(kFusedOpType.c_str(), "Remove out data edge failed."), return FAILED);
      FUSION_PASS_CHECK(GraphUtils::AddEdge(unpack_node->GetOutDataAnchor(m), in_anchor_ptr) != SUCCESS,
                        OP_LOGE(kFusedOpType.c_str(), "Add out data edge failed."), return FAILED);
    }
  }
  return SUCCESS;
}

vector<FusionPattern*> UnpackFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("UnpackFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(kFusedOpType.c_str(), "New a pattern object failed."), return patterns);
  pattern->AddOpDesc(kPatternFusedNode, {"Unpack"}).SetOutput(kPatternFusedNode);
  patterns.push_back(pattern);

  return patterns;
}

Status UnpackFusionPass::Fusion(ComputeGraph& graph, Mapping& mapping, vector<NodePtr>& new_nodes) {
  NodePtr fused_node = GetNodeFromMapping(kPatternFusedNode, mapping);
  OpDescPtr fused_desc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(fused_desc == nullptr,
                    OP_LOGE(kFusedOpType.c_str(), "The fused_node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  GeTensorDesc input_desc = fused_desc->GetInputDesc(0);
  vector<int64_t> input_shape = input_desc.GetShape().GetDims();
  for (int idx = 0; idx < input_shape.size(); idx++) {
    if (PatternFusionUtil::IsUnknownShape(input_shape[idx])) {
      OP_LOGE("UnpackFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
  }
  // A maximum of 63 tensors are supported in mini mode.
  int64_t num;
  AttrUtils::GetInt(fused_desc, "num", num);
  if (PatternFusionUtil::IsUnknownShape(num)) {
    OP_LOGE("UnpackFusionPass cannot be applied for num unknown shape.");
    return NOT_CHANGED;
  }
  FUSION_PASS_CHECK(num <= kMiniOut,
                    OP_LOGD(kFusedOpType.c_str(), "The amount of num of Unapck node is less than %lld.", kMiniOut),
                    return SUCCESS);
  FUSION_PASS_CHECK(num > kMiniOut * kMiniOut,
                    OP_LOGD(kFusedOpType.c_str(), "The amount of num of Unapck node is greater than %lld.",
                            kMiniOut * kMiniOut),
                    return SUCCESS);
  if (num > kMiniOut) {
    int64_t nodes_num = (num + kMiniOut - 1) / kMiniOut;
    int64_t last_node_num_unpack = num - (kMiniOut * (nodes_num - 1));
    vector<int64_t> size_splits_new(nodes_num - 1, kMiniOut);
    size_splits_new.push_back(last_node_num_unpack);

    int64_t axis;
    AttrUtils::GetInt(fused_desc, "axis", axis);

    OpDescPtr splitvd_base_desc = AttrUtils::CopyOpDesc(fused_desc);
    splitvd_base_desc->SetName(splitvd_base_desc->GetName() + "/SplitVD" + "Base_node");
    splitvd_base_desc->SetType("SplitVD");
    std::vector<GeTensorDesc> output_desc;
    for (int64_t c = num - 1; c >= nodes_num; c--) {
      OpDescUtils::ClearOutputDesc(splitvd_base_desc, c);
    }
    NodePtr splitvd_base_node = graph.AddNode(splitvd_base_desc);
    new_nodes.push_back(splitvd_base_node);
    AttrUtils::SetListInt(splitvd_base_node->GetOpDesc(), "size_splits", size_splits_new);
    AttrUtils::SetInt(splitvd_base_node->GetOpDesc(), "split_dim", axis);
    AttrUtils::SetInt(splitvd_base_node->GetOpDesc(), "num_split", nodes_num);

    GeTensorDesc splitvd_input_tensor = splitvd_base_desc->GetInputDesc(0);
    GeShape splitvd_input_shape = splitvd_input_tensor.GetShape();
    int64_t dimnum = splitvd_input_shape.GetDimNum();
    int64_t split_dim = axis < 0 ? axis + dimnum : axis;
    for (int64_t h = 0; h < nodes_num; h++) {
      GeTensorDesc splitvd_out_tensor = splitvd_base_desc->GetOutputDesc(h);
      GeShape splitvd_out_shape = splitvd_input_shape;
      splitvd_out_shape.SetDim(split_dim, size_splits_new[h]);
      splitvd_out_tensor.SetShape(splitvd_out_shape);
      splitvd_base_desc->UpdateOutputDesc(h, splitvd_out_tensor);
      output_desc.push_back(splitvd_out_tensor);
    }
    FUSION_PASS_CHECK(splitvd_base_node == nullptr,
                      OP_LOGE(kFusedOpType.c_str(), "The splitvd_base_node:%s is null, fusion failed.",
                              splitvd_base_node->GetName().c_str()),
                      return PARAM_INVALID);
    FUSION_PASS_CHECK(fused_node->GetInDataAnchor(0) == nullptr,
                      OP_LOGE(kFusedOpType.c_str(), "The InDataAnchor(0) of fused_node is null, fusion failed."),
                      return PARAM_INVALID);
    FUSION_PASS_CHECK(
        GraphUtils::AddEdge(fused_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                            splitvd_base_node->GetInDataAnchor(0)) != SUCCESS,
        OP_LOGE(kFusedOpType.c_str(), "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                fused_node->GetName().c_str(), 0, splitvd_base_node->GetName().c_str(), 0),
        return FAILED);

    for (int64_t i = 0; i < nodes_num; i++) {
      if (i < nodes_num - 1) {
        AddUnpackOps(fused_desc, graph, new_nodes, output_desc, fused_node, splitvd_base_node, num, axis, i, i,
                     kMiniOut);
      } else if (i == nodes_num - 1 && last_node_num_unpack != 1) {
        AddUnpackOps(fused_desc, graph, new_nodes, output_desc, fused_node, splitvd_base_node, num, axis, i,
                     nodes_num - 1, last_node_num_unpack);
      } else {
        FUSION_PASS_CHECK(
            fused_node->GetOutDataAnchor(kMiniOut * i) == nullptr,
            OP_LOGE(kFusedOpType.c_str(), "The OutDataAnchor(kMiniOut * i) of fused_node is null, fusion failed."),
            return PARAM_INVALID);
        for (InDataAnchorPtr in_anchor_ptr : fused_node->GetOutDataAnchor(kMiniOut * i)->GetPeerInDataAnchors()) {
          FUSION_PASS_CHECK(
              GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(kMiniOut * i), in_anchor_ptr) != SUCCESS,
              OP_LOGE(kFusedOpType.c_str(), "Remove out data edge failed."), return FAILED);
          FUSION_PASS_CHECK(GraphUtils::AddEdge(splitvd_base_node->GetOutDataAnchor(i), in_anchor_ptr) != SUCCESS,
                            OP_LOGE(kFusedOpType.c_str(), "Add out data edge failed."), return FAILED);
        }
      }
    }
  }

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

  FUSION_PASS_CHECK(GRAPH_SUCCESS != graph.RemoveNode(fused_node),
                    OP_LOGE(kFusedOpType.c_str(), "Remove Node [%s] failed", fused_node->GetName().c_str()),
                    return FAILED);

  OP_LOGI(kFusedOpType.c_str(), "Unpack --> Unpack fusion SUCCEED.");
  return SUCCESS;
}
REGISTER_PASS("UnpackFusionPass", BUILT_IN_GRAPH_PASS, UnpackFusionPass);
}  // namespace fe
