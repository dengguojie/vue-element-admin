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
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>

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
static const char* FUSED_NODE = "Unpack";
static const std::string PATTERN_FUSEDNODE = "FusedNodeUnpack";
static const int64_t MINI_OUT = 63;
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
Status UnpackFusionPass::AddUnpackOps(ge::OpDescPtr fused_desc, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                      std::vector<ge::GeTensorDesc> output_desc, ge::NodePtr fused_node,
                                      ge::NodePtr splitvd_base_node, int64_t num, int64_t axis, int64_t i, int64_t j,
                                      int64_t mini_out) {
  ge::OpDescPtr unpack_desc = AttrUtils::CopyOpDesc(fused_desc);
  unpack_desc->SetName(unpack_desc->GetName() + "/Unpack" + to_string(j));
  unpack_desc->SetType("Unpack");
  for (int64_t c = num - 1; c >= mini_out; c--) {
    OpDescUtils::ClearOutputDesc(unpack_desc, c);
  }
  ge::NodePtr unpack_node = graph.AddNode(unpack_desc);
  newNodes.push_back(unpack_node);

  ge::AttrUtils::SetInt(unpack_node->GetOpDesc(), "axis", axis);
  ge::AttrUtils::SetInt(unpack_node->GetOpDesc(), "num", mini_out);

  if ((int64_t)output_desc.size() <= j) {
    return FAILED;
  }
  unpack_desc->UpdateInputDesc(0, output_desc[j]);
  for (int64_t h = 0; h < mini_out; h++) {
    ge::GeTensorDesc unpack_output_tensor = unpack_desc->GetOutputDesc(h);
    ge::GeShape unpack_output_shape = unpack_output_tensor.GetShape();
    unpack_output_tensor.SetShape(unpack_output_shape);
    unpack_desc->UpdateOutputDesc(h, unpack_output_tensor);
  }

  FUSION_PASS_CHECK(
      unpack_node == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "unpack_node:%s is null, fusion failed.", unpack_node->GetName().c_str()),
      return PARAM_INVALID);
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(splitvd_base_node->GetOutDataAnchor(i), unpack_node->GetInDataAnchor(0)),
      OP_LOGE(FUSED_OP_TYPE.c_str(),
              "Add edge from fused node:%s's index[%d]"
              " to fusion node:%s's index[%d] failed.",
              splitvd_base_node->GetName().c_str(), i, unpack_node->GetName().c_str(), i),
      return FAILED);

  for (int64_t m = 0; m < mini_out; m++) {
    for (InDataAnchorPtr inAnchorPtr : fused_node->GetOutDataAnchor(MINI_OUT * i + m)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(MINI_OUT * i + m), inAnchorPtr),
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(unpack_node->GetOutDataAnchor(m), inAnchorPtr),
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
    }
  }
  return SUCCESS;
}

vector<FusionPattern*> UnpackFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("UnpackFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);

  return patterns;
}

Status UnpackFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  ge::OpDescPtr fused_desc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(fused_desc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "fused_node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  // A maximum of 63 tensors are supported in mini mode.
  int64_t num;
  ge::AttrUtils::GetInt(fused_desc, "num", num);
  FUSION_PASS_CHECK(num <= MINI_OUT,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "The amount of num of Unapck node is less than 63."),
                    return SUCCESS);
  if (num > MINI_OUT) {
    int64_t nodes_num = (num + MINI_OUT - 1) / MINI_OUT;
    int64_t last_node_num_unpack = num - (MINI_OUT * (nodes_num - 1));
    vector<int64_t> size_splits_new(nodes_num - 1, MINI_OUT);
    size_splits_new.push_back(last_node_num_unpack);

    int64_t axis;
    ge::AttrUtils::GetInt(fused_desc, "axis", axis);

    ge::OpDescPtr splitvd_base_desc = AttrUtils::CopyOpDesc(fused_desc);
    splitvd_base_desc->SetName(splitvd_base_desc->GetName() + "/SplitVD" + "Base_node");
    splitvd_base_desc->SetType("SplitVD");
    std::vector<ge::GeTensorDesc> output_desc;
    for (int64_t c = num - 1; c >= nodes_num; c--) {
      OpDescUtils::ClearOutputDesc(splitvd_base_desc, c);
    }
    ge::NodePtr splitvd_base_node = graph.AddNode(splitvd_base_desc);
    newNodes.push_back(splitvd_base_node);
    ge::AttrUtils::SetListInt(splitvd_base_node->GetOpDesc(), "size_splits", size_splits_new);
    ge::AttrUtils::SetInt(splitvd_base_node->GetOpDesc(), "split_dim", axis);
    ge::AttrUtils::SetInt(splitvd_base_node->GetOpDesc(), "num_split", nodes_num);

    ge::GeTensorDesc splitvd_input_tensor = splitvd_base_desc->GetInputDesc(0);
    ge::GeShape splitvd_input_shape = splitvd_input_tensor.GetShape();
    int64_t dimnum = splitvd_input_shape.GetDimNum();
    int64_t split_dim = axis < 0 ? axis + dimnum : axis;
    for (int64_t h = 0; h < nodes_num; h++) {
      ge::GeTensorDesc splitvd_out_tensor = splitvd_base_desc->GetOutputDesc(h);
      ge::GeShape splitvd_out_shape = splitvd_input_shape;
      splitvd_out_shape.SetDim(split_dim, size_splits_new[h]);
      splitvd_out_tensor.SetShape(splitvd_out_shape);
      splitvd_base_desc->UpdateOutputDesc(h, splitvd_out_tensor);
      output_desc.push_back(splitvd_out_tensor);
    }
    FUSION_PASS_CHECK(splitvd_base_node == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "splitvd_base_node:%s is null, fusion failed.",
                              splitvd_base_node->GetName().c_str()),
                      return PARAM_INVALID);

    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                         splitvd_base_node->GetInDataAnchor(0)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(),
                              "Add edge from fused node:%s's index[%d]"
                              "to fusion node:%s's index[%d] failed.",
                              fused_node->GetName().c_str(), (0), splitvd_base_node->GetName().c_str(), 0),
                      return FAILED);

    for (int64_t i = 0; i < nodes_num; i++) {
      if (i < nodes_num - 1) {
        AddUnpackOps(fused_desc, graph, newNodes, output_desc, fused_node, splitvd_base_node, num, axis, i, i,
                     MINI_OUT);
      } else if (i == nodes_num - 1 && last_node_num_unpack != 1) {
        AddUnpackOps(fused_desc, graph, newNodes, output_desc, fused_node, splitvd_base_node, num, axis, i,
                     nodes_num - 1, last_node_num_unpack);
      } else {
        for (InDataAnchorPtr inAnchorPtr : fused_node->GetOutDataAnchor(MINI_OUT * i)->GetPeerInDataAnchors()) {
          FUSION_PASS_CHECK(
              SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(MINI_OUT * i), inAnchorPtr),
              OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
          FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(splitvd_base_node->GetOutDataAnchor(i), inAnchorPtr),
                            OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
        }
      }
    }
  }

  for (auto inAnchor : fused_node->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  for (auto outAnchor : fused_node->GetAllOutDataAnchors()) {
    if (outAnchor != nullptr) {
      outAnchor->UnlinkAll();
    }
  }

  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(fused_node),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove Node [%s] failed", fused_node->GetName().c_str()),
                    return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Unpack --> Unpack fusion SUCCESSS!!!!!");
  return SUCCESS;
}

std::string UnpackPassName = "UnpackFusionPass";
REGISTER_PASS(UnpackPassName, BUILT_IN_GRAPH_PASS, UnpackFusionPass);
}  // namespace fe
