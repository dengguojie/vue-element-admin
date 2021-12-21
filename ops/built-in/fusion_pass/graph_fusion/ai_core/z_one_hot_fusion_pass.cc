/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
 * \file z_one_hot_fusion_pass.cc
 * \brief z_one_hot_fusion_pass
 */
#include "z_one_hot_fusion_pass.h"

#include <iostream>
#include <map>
#include <vector>

#include "op_log.h"
#include "error_util.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "securec.h"

using namespace std;
using namespace ge;

namespace fe {
vector<FusionPattern*> ZOneHotFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ZOneHotFusionPassPattern");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc("OneHotD", {"OneHotD"}).SetOutput("OneHotD");
  patterns.push_back(pattern);
  return patterns;
}

// to change one_hot_d node to one_hot node when check supported
// partly realize the static2dynamic process of one_hot
Status ZOneHotFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "ZOneHotFusionPass is running.");
  ge::NodePtr one_hot_node = GetNodeFromMapping("OneHotD", mapping);
  FUSION_PASS_CHECK(one_hot_node == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "onehotd not exist"), return NOT_CHANGED);
  ge::OpDescPtr one_hot_op_desc = one_hot_node->GetOpDesc();
  FUSION_PASS_CHECK(one_hot_op_desc == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "Failed to get op desc"),
                    return NOT_CHANGED);

  int32_t input_anchors = one_hot_node->GetAllInDataAnchors().size();
  OP_LOGD(FUSED_OP_TYPE.c_str(), "input anchors is %d", input_anchors);
  // one_hot_d ir inputs is 3
  int32_t onehotd_indata_anchors = 3;
  FUSION_PASS_CHECK(input_anchors < onehotd_indata_anchors, OP_LOGI(FUSED_OP_TYPE.c_str(), "input anchors less then 3"),
                    return NOT_CHANGED);
  auto indata_anchor = one_hot_node->GetInDataAnchor(1);
  auto indata_anchor2 = one_hot_node->GetInDataAnchor(2);
  FUSION_PASS_CHECK(indata_anchor == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "indata anchor 1 get failed"), return FAILED);
  FUSION_PASS_CHECK(indata_anchor2 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "indata anchor 2 get failed"), return FAILED);
  auto pre_out_dataanchor = indata_anchor->GetPeerOutAnchor();
  auto pre_out_dataanchor2 = indata_anchor2->GetPeerOutAnchor();
  FUSION_PASS_CHECK(pre_out_dataanchor == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "pre_out_dataanchor 1 get failed"),
                    return FAILED);
  FUSION_PASS_CHECK(pre_out_dataanchor2 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "pre_out_dataanchor 2 get failed"),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(pre_out_dataanchor, indata_anchor) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove edge input 1 edge failed"),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(pre_out_dataanchor2, indata_anchor2) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove edge input 2 edge failed"),
                    return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "remove edge input1 and input2 is success");

  int32_t depth;
  ge::AttrUtils::GetInt(one_hot_op_desc, "depth", depth);
  one_hot_op_desc->DelAttr("depth");
  std::vector<int64_t> depth_shape = {1};
  ge::GeShape const_depth = ge::GeShape(depth_shape);
  auto depth_input_desc = ge::GeTensorDesc(const_depth, ge::FORMAT_ND, ge::DT_INT32);
  ge::GeTensorPtr out_tensor = nullptr;
  FUSION_PASS_MAKE_SHARED((out_tensor = std::make_shared<ge::GeTensor>(depth_input_desc)), out_tensor = nullptr;
                          return PARAM_INVALID);
  vector<int32_t> perm_b32 = {static_cast<int32_t>(depth)};

  out_tensor->SetData(reinterpret_cast<uint8_t*>(perm_b32.data()), perm_b32.size() * sizeof(int32_t));
  ge::OpDescPtr out_op_desc = ge::OpDescUtils::CreateConstOp(out_tensor);
  auto const_node = graph.AddNode(out_op_desc);
  FUSION_PASS_CHECK(one_hot_node->AddLinkFrom(1, const_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to AddEdge const"), return FAILED);
  ge::NodePtr input_data_anchor1 = pre_out_dataanchor->GetOwnerNode();
  FUSION_PASS_CHECK(input_data_anchor1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_data_anchor1 is null"), return FAILED);
  ge::NodePtr input_data_anchor2 = pre_out_dataanchor2->GetOwnerNode();
  FUSION_PASS_CHECK(input_data_anchor2 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_data_anchor2 is null"), return FAILED);
  FUSION_PASS_CHECK(one_hot_node->AddLinkFrom(2, input_data_anchor1) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge input2 failed"), return FAILED);
  FUSION_PASS_CHECK(one_hot_node->AddLinkFrom(3, input_data_anchor2) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge input3 failed"), return FAILED);
  one_hot_op_desc->SetType("OneHot");
  std::map<string, uint32_t> input_name_id = {{"x", 0}, {"depth", 1}, {"on_value", 2}, {"off_value", 3}};
  one_hot_node->GetOpDesc()->UpdateInputName(input_name_id);

  bool is_supported = CheckOpSupported(one_hot_op_desc);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "is_supported=%d.", is_supported);
  if (!is_supported) {
    ge::InDataAnchorPtr anchor = one_hot_node->GetInDataAnchor(1);
    anchor->UnlinkAll();
    ge::NodeUtils::ClearInDataAnchor(one_hot_node, anchor);
    ge::OpDescUtils::ClearInputDesc(one_hot_node->GetOpDesc(), 1);
    FUSION_PASS_CHECK(graph.RemoveNode(const_node) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove const node"),
                      return FAILED);
    ge::AttrUtils::SetInt(one_hot_op_desc, "depth", depth);
    one_hot_op_desc->SetType("OneHotD");
    std::map<string, uint32_t> onehotd_name_id = {{"x", 0}, {"on_value", 1}, {"off_value", 2}};
    one_hot_node->GetOpDesc()->UpdateInputName(onehotd_name_id);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "set type OneHotD success");
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "ZOneHotFusionPass run success.");
  return SUCCESS;
}

REGISTER_PASS("ZOneHotFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, ZOneHotFusionPass);
}  // namespace fe
