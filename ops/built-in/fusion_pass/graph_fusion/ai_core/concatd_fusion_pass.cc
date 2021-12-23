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
 * \file concatd_fusion_pass.cpp
 * \brief ConcatD fusion pass(ConcatD --> ConcatD)
 */
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>

#include "concatd_fusion_pass.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"

using namespace ge;
namespace fe {
static const char* FUSED_NODE = "ConcatD";
static const std::string PATTERN_FUSEDNODE = "FusedNodeConcat";
vector<FusionPattern*> ConcatDFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ConcatDFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);

  return patterns;
}

void ConcatDFusionPass::UpdateInputName(ge::OpDescPtr& input_desc_ptr) {
  auto input_count = input_desc_ptr->GetAllInputsSize();
  map<string, uint32_t> name_index_map;
  string name = "x";
  for (size_t i = 0; i < input_count; ++i) {
    name_index_map.insert({name + std::to_string(i), i});
  }
  input_desc_ptr->UpdateInputName(name_index_map);
}

// vector<ge::NodePtr> &fusionNodes: Store fusion nodes,
// including newly added nodes and fused but not deleted nodes
Status ConcatDFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  ge::OpDescPtr fusedDesc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fused_node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  // A maximum of 63 tensors are supported in mini mode.
  int64_t inputs_num = fusedDesc->GetInputsSize();
  int64_t NeedTangent = 63;
  if (HasUnKnowShape(fused_node)) {
    // Maximum of 48 tensors are supported in mini mode for dynamic shape of concatv2
    NeedTangent = 48;
  }
  const int64_t max_inputs = NeedTangent;
  FUSION_PASS_CHECK(inputs_num <= max_inputs,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "The amount of input of ConcatD node is less than %lld.",
                            max_inputs),
                    return NOT_CHANGED);

  if (inputs_num > max_inputs) {
    int64_t nodes_num = (inputs_num + max_inputs - 1) / max_inputs;
    int64_t last_node_inputs_num = inputs_num - (max_inputs * (nodes_num - 1));
    ge::OpDescPtr ConcatdDesc_orig = AttrUtils::CopyOpDesc(fusedDesc);
    ge::OpDescPtr ConcatdBaseDesc = AttrUtils::CopyOpDesc(fusedDesc);
    ConcatdBaseDesc->SetName(ConcatdBaseDesc->GetName() + "/ConcatD" + "Base_node");
    ConcatdBaseDesc->SetType("ConcatD");
    int64_t concat_dim;
    ge::AttrUtils::GetInt(fusedDesc, "concat_dim", concat_dim);
    ge::AttrUtils::SetInt(ConcatdBaseDesc, "concat_dim", concat_dim);

    for (int64_t c = inputs_num - 1; c >= nodes_num; c--) {
      OpDescUtils::ClearInputDesc(ConcatdBaseDesc, c);
    }

    ge::NodePtr concatd_base_node = graph.AddNode(ConcatdBaseDesc);
    FUSION_PASS_CHECK(concatd_base_node == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "concatd_base_node:%s is null, fusion failed.",
                              concatd_base_node->GetName().c_str()),
                      return PARAM_INVALID);
    fusionNodes.push_back(concatd_base_node);
    ge::AttrUtils::SetInt(concatd_base_node->GetOpDesc(), "N", nodes_num);
    auto out_data_anchor = fused_node->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(out_data_anchor == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "out_data_anchor is null, fusion failed."),
                      return PARAM_INVALID);
    for (InDataAnchorPtr inAnchorPtr : fused_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(0), inAnchorPtr),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(concatd_base_node->GetOutDataAnchor(0), inAnchorPtr),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
    }

    for (int64_t i = 0; i < nodes_num; i++) {
      if (i < nodes_num - 1) {
        int64_t inputs_num1 = fusedDesc->GetInputsSize();
        ge::OpDescPtr ConcatdDesc = AttrUtils::CopyOpDesc(fusedDesc);
        ConcatdDesc->SetName(ConcatdDesc->GetName() + "/ConcatD" + to_string(i));
        ConcatdDesc->SetType("ConcatD");
        ge::AttrUtils::GetInt(fusedDesc, "concat_dim", concat_dim);
        ge::AttrUtils::SetInt(ConcatdDesc, "concat_dim", concat_dim);

        if (i == 0) {
          for (int64_t a = inputs_num1 - 1; a >= max_inputs; a--) {
            OpDescUtils::ClearInputDesc(ConcatdDesc, a);
          }
        } else {
          for (int64_t a = i * max_inputs - 1; a >= 0; a--) {
            OpDescUtils::ClearInputDesc(ConcatdDesc, a);
          }
          for (int64_t a = inputs_num1 - (max_inputs * i + 1); a >= max_inputs; a--) {
            OpDescUtils::ClearInputDesc(ConcatdDesc, a);
          }
        }
        ge::NodePtr concatd_node = graph.AddNode(ConcatdDesc);
        FUSION_PASS_CHECK(concatd_node == nullptr,
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "concatd_node is null, fusion failed."),
                          return PARAM_INVALID);
        fusionNodes.push_back(concatd_node);
        ge::AttrUtils::SetInt(concatd_node->GetOpDesc(), "N", max_inputs);
        // infershape begin
	UpdateInputName(ConcatdDesc);
	Operator op = ge::OpDescUtils::CreateOperatorFromNode(concatd_node);
	auto infer_shape_ret = op.InferShapeAndType();
	OP_LOGE_IF(infer_shape_ret != GRAPH_SUCCESS, NOT_CHANGED, FUSED_OP_TYPE, "InferShapeAndType failed.");
        ConcatdBaseDesc->UpdateInputDesc(i, ConcatdDesc->GetOutputDesc(0));
        // infershape end
        FUSION_PASS_CHECK(
            concatd_node == nullptr,
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "concatd_node is null, fusion failed."),
            return PARAM_INVALID);

        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(concatd_node->GetOutDataAnchor(0),
                                                             concatd_base_node->GetInDataAnchor(i)),
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                  "Add edge from fused node:%s's index[%ld] to fusion node:%s's index[%ld] failed.",
                                  concatd_base_node->GetName().c_str(), i, concatd_node->GetName().c_str(), i),
                          return FAILED);

        for (int64_t m = 0; m < max_inputs; m++) {
          FUSION_PASS_CHECK(
              SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(m + i * max_inputs)->GetPeerOutAnchor(),
                                                 concatd_node->GetInDataAnchor(m)),
              VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "Add edge from fused node:%s's index[%ld] to fusion node:%s's index[%ld] failed.",
                      fused_node->GetName().c_str(), (m + i * max_inputs), concatd_node->GetName().c_str(), m),
              return FAILED);
        }
      } else {
        int64_t inputs_num2 = fusedDesc->GetInputsSize();
        ge::OpDescPtr LastConcatDDesc = AttrUtils::CopyOpDesc(fusedDesc);
        LastConcatDDesc->SetName(LastConcatDDesc->GetName() + "/ConcatD" + to_string(nodes_num - 1));
        LastConcatDDesc->SetType("ConcatD");

        for (int64_t b = inputs_num2 - last_node_inputs_num - 1; b >= 0; b--) {
          OpDescUtils::ClearInputDesc(LastConcatDDesc, b);
        }
        ge::NodePtr last_concatd_node = graph.AddNode(LastConcatDDesc);
        FUSION_PASS_CHECK(last_concatd_node == nullptr,
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode last_concatd_node is null, fusion failed."),
                          return PARAM_INVALID);
        fusionNodes.push_back(last_concatd_node);
        ge::AttrUtils::SetInt(last_concatd_node->GetOpDesc(), "N", last_node_inputs_num);
        // the last_node infershape begin
        UpdateInputName(LastConcatDDesc);
	Operator op = ge::OpDescUtils::CreateOperatorFromNode(last_concatd_node);
	auto infer_shape_ret = op.InferShapeAndType();
	OP_LOGE_IF(infer_shape_ret != GRAPH_SUCCESS, NOT_CHANGED, FUSED_OP_TYPE, "InferShapeAndType failed.");
        ConcatdBaseDesc->UpdateInputDesc(i, LastConcatDDesc->GetOutputDesc(0));
        // the last_node infershape end
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(last_concatd_node->GetOutDataAnchor(0),
                                                             concatd_base_node->GetInDataAnchor(i)),
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                  "Add edge from fused node:%s's index[%ld] to fusion node:%s's index[%ld] failed.",
                                  concatd_base_node->GetName().c_str(), i, last_concatd_node->GetName().c_str(), i),
                          return FAILED);

        for (int64_t n = 0; n < last_node_inputs_num; n++) {
          FUSION_PASS_CHECK(
              SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(n + i * max_inputs)->GetPeerOutAnchor(),
                                                 last_concatd_node->GetInDataAnchor(n)),
              VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "Add edge from fused node:%s's index[%ld] to fusion node:%s's index[%ld] failed.",
                      fused_node->GetName().c_str(), (n + i * max_inputs), last_concatd_node->GetName().c_str(), n),
              return FAILED);
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
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove Node [%s] failed", fused_node->GetName().c_str()),
                    return FAILED);

  return SUCCESS;
}

REGISTER_PASS("ZConcatDFusionPass", BUILT_IN_GRAPH_PASS, ConcatDFusionPass);
}  // namespace fe
