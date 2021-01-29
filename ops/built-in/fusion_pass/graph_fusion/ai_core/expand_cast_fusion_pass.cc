/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file expand_cast_fusion_pass.cc
 * \brief expand cast fusion (Cast--Expand)
 */
#include "expand_cast_fusion_pass.h"
#include <memory>
#include <string>
#include <vector>

#include "graph/utils/op_desc_utils.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"

namespace fe {
static const string PATTERN_EXPAND = "expand";
static const string PATTERN_CAST = "cast";

static const char* EXPAND = "ExpandD";

/*
    fusion pattern
            node
                \
                Cast
                 \
                Expand---
                /
               /
            node
*/
vector<FusionPattern*> ExpandCastFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("expandCastFusion");
  if (pattern == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "pattern is nullptr,Create pattern not success!");
    return patterns;
  }
  pattern->AddOpDesc(PATTERN_EXPAND, {FUSED_OP_TYPE})
      .SetOutput(PATTERN_EXPAND);
  patterns.push_back(pattern);
  return patterns;
}

Status ExpandCastFusionPass::CreateCastNode(ge::ComputeGraph &graph,
                                            ge::NodePtr &fused_node,
                                            ge::NodePtr &new_node) const{
  ge::OpDescPtr new_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (new_desc = std::make_shared<ge::OpDesc>("Cast_For_Expand", "Cast")),
      return INTERNAL_ERROR);
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fused_node);
  auto input_desc = op.GetInputDesc(0);
  ge::GeShape input_shape(input_desc.GetShape().GetDims());
  ge::Format data_format = input_desc.GetFormat();

  auto ret = new_desc->AddInputDesc(GeTensorDesc(input_shape, data_format, DT_INT64));
  FUSION_PASS_CHECK(
      ret != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Create cast node failed due to add inputDesc fail."),
      return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(new_desc, "dst_type", DT_INT32),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Fail to set attr to int32 from cast."),
                    return FAILED);
  ret = new_desc->AddOutputDesc(GeTensorDesc(input_shape, data_format, DT_INT32));
  FUSION_PASS_CHECK(
      ret != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Create cast node failed due to add outputDesc fail."),
      return FAILED);
  new_node = graph.AddNode(new_desc);
  return SUCCESS;
}

bool ExpandCastFusionPass::DeleteInput(ge::ComputeGraph &graph, ge::NodePtr &node,
                        ge::OpDescPtr &desc, uint32_t index) {
  ge::InDataAnchorPtr anchor = node->GetInDataAnchor(index);
  ge::OutDataAnchorPtr const_anchor = anchor->GetPeerOutAnchor();
  ge::NodeUtils::ClearInDataAnchor(node, anchor);
  if (const_anchor != nullptr) {
    ge::GraphUtils::RemoveEdge(const_anchor, anchor);
    ge::NodePtr const_node = const_anchor->GetOwnerNode();
    if (PatternFusionUtil::GetOutEdgeSize(const_node) == 0) {
      FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(const_node),
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove Node[%s] failed",
                                const_node->GetName().c_str()),
                        return false);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Remove Node:[%s].", const_node->GetName().c_str());
    } else {
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:[%s] have output link to other node.",
              const_node->GetName().c_str());
    }
  }
  if (!ge::OpDescUtils::ClearInputDesc(desc, index)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to clear input desc[%d]", index);
  }
  return true;
}

bool ExpandCastFusionPass::GetConstValue(
    const Operator &op, const Tensor &const_tensor, const DataType &dtype,
    std::vector<int64_t> &const_data) {
  size_t size = 0;
  if (dtype == ge::DT_INT32) {
    int32_t *const_data_ptr = (int32_t *)const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back((int32_t)((*(const_data_ptr + i))));
      OP_LOGD(op.GetName().c_str(), "const data int32 in fusion pass is %d",
              (int32_t)(*(const_data_ptr + i)));
    }
  } else if (dtype == ge::DT_INT64) {
    int64_t *const_data_ptr = (int64_t *)const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back(((int64_t)(*(const_data_ptr + i))));
      OP_LOGD(op.GetName().c_str(), "const data int64 in fusion pass is %d",
              (int64_t)(*(const_data_ptr + i)));
    }
  } else if (dtype == ge::DT_INT16) {
    int16_t *const_data_ptr = (int16_t *)const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int16_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back(((int16_t)(*(const_data_ptr + i))));
      OP_LOGD(op.GetName().c_str(), "const data int16 in fusion pass is %d",
              (int16_t)(*(const_data_ptr + i)));
    } 
  } else {
    OP_LOGE(op.GetName().c_str(), "not support this type");
    return false;
  }
  return true;
}

Status ExpandCastFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusion_nodes) {
  ge::NodePtr expand_node = GetNodeFromMapping(PATTERN_EXPAND, mapping);
  FUSION_PASS_CHECK(expand_node == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "expand node is null"), return FAILED);

  ge::OpDescPtr expand_desc = expand_node->GetOpDesc();
  FUSION_PASS_CHECK(expand_desc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Expand[%s] is not supported by FE, fusion abort.",
                            expand_desc->GetName().c_str()),
                    return PARAM_INVALID);
  
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(expand_node);
  Tensor shape_tensor;
  if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get const value failed of [shape]");
    return GRAPH_FAILED;
  }
  vector<int64_t> shape_list;
  ge::DataType shape_data_type = expand_desc->GetInputDesc(1).GetDataType();
  FUSION_PASS_CHECK(!GetConstValue(op, shape_tensor, shape_data_type, shape_list),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Fail to get const value from operate."),
                    return FAILED);
  ge::AttrUtils::SetListInt(expand_desc, "shape", shape_list);
  FUSION_PASS_CHECK(!DeleteInput(graph, expand_node, expand_desc, 1),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Fail to delete const input from operate."),
                    return FAILED);
  ge::DataType x_data_type = expand_desc->GetInputDesc(0).GetDataType();

  expand_desc->SetType(EXPAND);
  if (x_data_type == DT_INT64) {

    ge::GeTensorDesc expand_input_desc = expand_desc->GetInputDesc(0);
    expand_input_desc.SetOriginDataType(DT_INT32);
    expand_input_desc.SetDataType(DT_INT32);
    ge::GeTensorDesc expand_output_desc = expand_desc->GetOutputDesc(0);
    expand_output_desc.SetOriginDataType(DT_INT32);
    expand_output_desc.SetDataType(DT_INT32);
    auto ret = expand_desc->UpdateInputDesc(0,expand_input_desc);
    auto ret2 = expand_desc->UpdateOutputDesc(0,expand_output_desc);
    FUSION_PASS_CHECK(ret != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Update inputDesc failed."),
                      return FAILED);
    FUSION_PASS_CHECK(ret2 != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Update outputDesc failed."),
                      return FAILED);

    ge::NodePtr cast_node = nullptr;
    ret = CreateCastNode(graph, expand_node, cast_node);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "cast node is null"), return FAILED);

    auto input_expand_anchor = expand_node->GetInDataAnchor(0);
    auto input_cast_anchor = cast_node->GetInDataAnchor(0);
    auto output_cast_anchor = cast_node->GetOutDataAnchor(0);
    auto input_expand_anchor_peer = expand_node->GetInDataAnchor(0)->GetPeerOutAnchor();
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(input_expand_anchor,input_expand_anchor_peer) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Fail to remove edge."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(input_expand_anchor_peer,input_cast_anchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Fail to add edge first time."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(output_cast_anchor,input_expand_anchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Fail to add edge second time."), return FAILED);
    
    ge::NodePtr expand_node_new = graph.AddNode(expand_desc);
    fusion_nodes.push_back(cast_node);
    fusion_nodes.push_back(expand_node_new);

  } else
  {
    ge::NodePtr expand_node_new = graph.AddNode(expand_desc);
    fusion_nodes.push_back(expand_node_new);
  }
  
  return SUCCESS;
}

REGISTER_PASS("ExpandCastFusionPass", BUILT_IN_GRAPH_PASS, ExpandCastFusionPass);
}  // namespace fe
