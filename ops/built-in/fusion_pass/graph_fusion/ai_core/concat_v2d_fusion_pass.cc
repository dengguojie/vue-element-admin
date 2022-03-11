/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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
 * \file concat_v2d_fusion_pass.cpp
 * \brief Concatv2d fusion pass(multi Concatv2d --> single Concatv2d)
 */
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
#include "common/util/platform_info.h"
#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"
#include "concat_v2d_fusion_pass.h"

using namespace ge;
namespace fe {
static const string CONCATV2D = "ConcatV2D";
static const std::string PATTERN_FUSEDNODE = "FusedNodeConcat";
static const char ATTR_CONCAT_DIM[] = "concat_dim";
vector<FusionPattern*> Concatv2dFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("Concatv2dFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(
                    FUSED_OP_TYPE, "new a pattern object failed."), return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {CONCATV2D}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);

  return patterns;
}

bool Concatv2dFusionPass::CheckConcatValid(const ge::NodePtr& node, const ge::Format format, const ge::GeShape shape,
                                           const int32_t dim_num) {
  int32_t concat_dim = 0;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetInt(node->GetOpDesc(), ATTR_CONCAT_DIM, concat_dim),
                    OP_LOGI(FUSED_OP_TYPE, "There is no concat_dim attr."), return false);

  FUSION_PASS_CHECK(dim_num != concat_dim,
                    OP_LOGD(FUSED_OP_TYPE, "%s 's concat_dim is %ld, target is %ld, can't fuss.",
                            node->GetName().c_str(), concat_dim, dim_num),
                    return false);

  OP_LOGD(FUSED_OP_TYPE, "%s 's concat_dim is %ld, target is %ld.", node->GetName().c_str(),
          concat_dim, dim_num);

  ge::Format input_format = node->GetOpDesc()->GetInputDesc(0).GetFormat();
  FUSION_PASS_CHECK(input_format != format,
                    OP_LOGD(FUSED_OP_TYPE, "%s 's input format is %s, target is %s, can't fuss.",
                            node->GetName().c_str(), ge::TypeUtils::FormatToSerialString(input_format).c_str(),
                            ge::TypeUtils::FormatToSerialString(format).c_str()),
                    return false);

  OP_LOGD(FUSED_OP_TYPE, "%s 's input format is %s, target is %s.", node->GetName().c_str(),
          ge::TypeUtils::FormatToSerialString(input_format).c_str(),
          ge::TypeUtils::FormatToSerialString(format).c_str());

  const ge::GeShape input_shape = node->GetOpDesc()->GetInputDesc(0).GetShape();
  int32_t index = 0;
  for (auto dim : input_shape.GetDims()) {
    if (index == dim_num) {
      continue;
    }
    FUSION_PASS_CHECK(dim != shape.GetDim(index),
                      OP_LOGD(FUSED_OP_TYPE, "%s 's input %ld dim is %ld, target is %ld, can't fuss.",
                              node->GetName().c_str(), index, dim, shape.GetDim(index)),
                      OP_LOGD(FUSED_OP_TYPE, "%s 's input %ld dim is %ld, target is %ld.",
                              node->GetName().c_str(), index, dim, shape.GetDim(index));
                      return false);
    index++;
  }
  const uint32_t out_nodes_size = 1;
  FUSION_PASS_CHECK(node->GetOutAllNodes().size() != out_nodes_size,
                    OP_LOGD(FUSED_OP_TYPE, "%s 's output should be 1, can't fuss.", node->GetName().c_str()),
                    return false);
  return true;
}

bool Concatv2dFusionPass::HasUnKnowInputShape(const std::vector<ge::NodePtr> &input_nodes) {
  bool res = false;
  res = std::any_of(std::begin(input_nodes), std::end(input_nodes), \
                    [](const ge::NodePtr& item){return HasUnKnowShape(item);});
  return res;
}

Status Concatv2dFusionPass::PatternParse(ge::NodePtr concat_v2d_node, vector<ge::NodePtr>& fused_input_nodes,
                                         vector<ge::NodePtr>& concat_nodes) {
  int32_t direct_out_node_num = concat_v2d_node->GetOutAllNodes().size();
  FUSION_PASS_CHECK(direct_out_node_num <= 0,
                    OP_LOGI(FUSED_OP_TYPE, "There is no need to fusion, out node num %d.", direct_out_node_num),
                    return FAILED);
  int32_t concat_dim = 0;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetInt(concat_v2d_node->GetOpDesc(), ATTR_CONCAT_DIM, concat_dim),
                    OP_LOGI(FUSED_OP_TYPE, "There is no need to fusion."), return FAILED);

  ge::Format format = concat_v2d_node->GetOpDesc()->GetInputDesc(0).GetFormat();
  ge::GeShape shape = concat_v2d_node->GetOpDesc()->GetInputDesc(0).GetShape();
  for (auto node_ptr : concat_v2d_node->GetInAllNodes()) {
    if (node_ptr->GetType() == CONCATV2D) {
      FUSION_PASS_CHECK(!CheckConcatValid(node_ptr, format, shape, concat_dim),
                        OP_LOGI(FUSED_OP_TYPE, "There is no need to fusion."), return FAILED);
      concat_nodes.push_back(node_ptr);
      for (auto pre_node : node_ptr->GetInAllNodes()) {
        FUSION_PASS_CHECK(
            pre_node->GetOutAllNodes().size() != 1,
            OP_LOGD(FUSED_OP_TYPE, "%s 's output should be 1, can't fuss.", pre_node->GetName().c_str()),
            return FAILED);
        fused_input_nodes.push_back(pre_node);
      }
    } else {
      FUSION_PASS_CHECK(
          node_ptr->GetOutAllNodes().size() != 1,
          OP_LOGD(FUSED_OP_TYPE, "%s 's output should be 1, can't fuss.", node_ptr->GetName().c_str()),
          return FAILED);
      fused_input_nodes.push_back(node_ptr);
    }
  }
  FUSION_PASS_CHECK(!concat_nodes.size(), OP_LOGD(FUSED_OP_TYPE, "Singel concat, no need fusion."),
                    return FAILED);
  return SUCCESS;
}

Status Concatv2dFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusion_nodes) {
  ge::NodePtr concat_v2d_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);

  FUSION_PASS_CHECK(concat_v2d_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(
                    FUSED_OP_TYPE, "new a pattern object failed"), return PARAM_INVALID);
  vector<ge::NodePtr> fused_input_nodes;
  vector<ge::NodePtr> concat_nodes;

  if (SUCCESS != PatternParse(concat_v2d_node, fused_input_nodes, concat_nodes)) {
    OP_LOGD(FUSED_OP_TYPE, "do not need do concatv2d fusion here, concatv2d name %s",
            concat_v2d_node->GetName().c_str());
    fused_input_nodes.clear();
    concat_nodes.clear();
    return NOT_CHANGED;
  }
  size_t max_count = 63;
  if (HasUnKnowInputShape(fused_input_nodes)) {
    max_count = 48;

    PlatformInfo platform_info;
    OptionalInfo optional_info;
    if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != SUCCESS) {
      OP_LOGW(FUSED_OP_TYPE, "Fail to get platform info.");
      optional_info.soc_version == "";
    }
    OP_LOGD(FUSED_OP_TYPE, "Get soc_version is: [%s].", optional_info.soc_version.c_str());
    if (optional_info.soc_version == "Ascend310") {
      max_count = 21;
    }
  }

  OP_LOGI(concat_v2d_node->GetName(), "concat input number %zu, max number %zu", fused_input_nodes.size(), max_count);
  if (fused_input_nodes.size() > max_count) {
    OP_LOGD(concat_v2d_node->GetName(), "concat input number %zu more than %zu", fused_input_nodes.size(), max_count);
    fused_input_nodes.clear();
    concat_nodes.clear();
    return NOT_CHANGED;
  }

  ge::OpDescPtr fused_concat_v2d_op_desc = AttrUtils::CloneOpDesc(concat_v2d_node->GetOpDesc());
  FUSION_PASS_CHECK(
      fused_concat_v2d_op_desc == nullptr,
      OP_LOGI(FUSED_OP_TYPE, "Node:%s's OpDesc is null, fusion failed.", concat_v2d_node->GetName().c_str()),
      return PARAM_INVALID);
  OP_LOGD(FUSED_OP_TYPE, "fused_concat_v2d_op_desc %s, optye %s, input %ld, output %ld",
          fused_concat_v2d_op_desc->GetName().c_str(), fused_concat_v2d_op_desc->GetType().c_str(),
          fused_concat_v2d_op_desc->GetAllInputsDesc().size(), fused_concat_v2d_op_desc->GetAllOutputsDesc().size());
  fused_concat_v2d_op_desc->SetName(concat_v2d_node->GetName());
  fused_concat_v2d_op_desc->SetType(CONCATV2D);

  for (auto input_desc : fused_concat_v2d_op_desc->GetAllInputsDesc()) {
    FUSION_PASS_CHECK(!ge::OpDescUtils::ClearInputDesc(fused_concat_v2d_op_desc, 0),
                      OP_LOGI(FUSED_OP_TYPE, "Node:%s's clear %d th input failed.",
                              fused_concat_v2d_op_desc->GetName().c_str(), 0),
                      return PARAM_INVALID);
  }

  size_t input_idx = 0;
  for (auto input_node : fused_input_nodes) {
    uint32_t get_index = input_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetIdx();
    ge::GeTensorDesc desc = input_node->GetOutAllNodes().at(0)->GetOpDesc()->GetInputDesc(get_index);
    string name = "x" + std::to_string(input_idx);
    fused_concat_v2d_op_desc->AddInputDesc(name, desc);
    input_idx++;
  }

  int64_t num_N_new = fused_input_nodes.size();
  OP_LOGD(concat_v2d_node->GetName(), "Node:%s's has %ld inputs.", fused_concat_v2d_op_desc->GetName().c_str(),
          num_N_new);
  ge::AttrUtils::SetInt(fused_concat_v2d_op_desc, "N", num_N_new);

  ge::NodePtr fused_concat_v2d_node = graph.AddNode(fused_concat_v2d_op_desc);
  std::map<string, uint32_t> output_name_id = {{"y", 0}};
  fused_concat_v2d_node->GetOpDesc()->UpdateOutputName(output_name_id);

  uint32_t index = 0;
  for (auto input_node : fused_input_nodes) {
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(input_node->GetOutDataAnchor(0),
                                              fused_concat_v2d_node->GetInDataAnchor(index)) != ge::GRAPH_SUCCESS,
                      OP_LOGI(FUSED_OP_TYPE, "add input edge between %s and %s %d th failed.",
                              input_node->GetName().c_str(), fused_concat_v2d_node->GetName().c_str(), index),
                      return NOT_CHANGED);
    index++;
  }

  for (auto out_anchor : concat_v2d_node->GetAllOutDataAnchors()) {
    for (InDataAnchorPtr in_anchor_ptr : out_anchor->GetPeerInDataAnchors()) {
      in_anchor_ptr->UnlinkAll();
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fused_concat_v2d_node->GetOutDataAnchor(0), in_anchor_ptr),
                        VECTOR_FUSION_INNER_ERR_REPORT(
                        FUSED_OP_TYPE, "Add edge from %s to fusion node %s's %d th failed.",
                        concat_v2d_node->GetName().c_str(), fused_concat_v2d_node->GetName().c_str(), 0),
                        return FAILED);
      OP_LOGD(FUSED_OP_TYPE, "Add edge from %s to fusion node %s's %d index success.",
              concat_v2d_node->GetName().c_str(), fused_concat_v2d_node->GetName().c_str(), 0);
    }
  }

  concat_nodes.push_back(concat_v2d_node);
  for (auto concat_node : concat_nodes) {
    for (auto in_anchor : concat_node->GetAllInDataAnchors()) {
      if (in_anchor) {
        in_anchor->UnlinkAll();
      }
    }
  }

  for (auto concat_node : concat_nodes) {
    FUSION_PASS_CHECK(graph.RemoveNode(concat_node) == ge::GRAPH_FAILED,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "remove node %s failed.",
                                                     concat_node->GetName().c_str()),
                      return FAILED);
  }
  return SUCCESS;
}

REGISTER_PASS("ZConcatv2dFusionPass", BUILT_IN_GRAPH_PASS, Concatv2dFusionPass);
}  // namespace fe
