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
 * \file nms_with_mask_fusion_pass.cpp
 * \brief fusion pass(Add a pad op before NMSWithMask for input of box_scores)
 */
#include "nms_with_mask_fusion_pass.h"

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
#include "error_util.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {
static const std::string kPatternNMSWithMask = "NMSWithMask";

vector<FusionPattern*> NMSWithMaskFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("NMSWithMaskFusion");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "New a pattern object failed."), return patterns);
  pattern->AddOpDesc(kPatternNMSWithMask, {"NMSWithMask"}).SetOutput(kPatternNMSWithMask);
  patterns.push_back(pattern);
  return patterns;
}

Status NMSWithMaskFusionPass::Fusion(ComputeGraph& graph, Mapping& mapping, vector<NodePtr>& fusion_nodes) {
  NodePtr nms_node_ptr = GetNodeFromMapping(kPatternNMSWithMask, mapping);
  FUSION_PASS_CHECK(nms_node_ptr == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "NMSWithMask Node is null, fusion failed."),
                    return PARAM_INVALID);
  OpDescPtr nms_desc_ptr = nms_node_ptr->GetOpDesc();
  FUSION_PASS_CHECK(nms_node_ptr == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "NMSWithMask desc is null, FE fusion failed."), return PARAM_INVALID);

  // clone pad node desc from nms
  OpDescPtr pad_desc_ptr = AttrUtils::CloneOpDesc(nms_desc_ptr);
  FUSION_PASS_CHECK(pad_desc_ptr == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Create PadD OpDesc failed, fusion failed."),
                    return PARAM_INVALID);
  pad_desc_ptr->SetType("PadD");
  pad_desc_ptr->SetName(pad_desc_ptr->GetName() + "_PadD");

  std::map<string, uint32_t> input_name_idx;
  input_name_idx["x"] = 0;
  pad_desc_ptr->UpdateInputName(input_name_idx);

  // delete output desc of pad node
  int tmp_output_size = pad_desc_ptr->GetOutputsSize();
  FUSION_PASS_CHECK(tmp_output_size < 1,
                    OP_LOGW(kFusedOpType.c_str(), "The output of %s is zero", nms_node_ptr->GetName().c_str()),
                    return NOT_CHANGED);
  while (tmp_output_size > 0) {
    tmp_output_size--;
    OpDescUtils::ClearOutputDesc(pad_desc_ptr, tmp_output_size);
  }

  // add the output edge of pad node, and update the info of pad
  int nms_input_size = nms_desc_ptr->GetInputsSize();
  FUSION_PASS_CHECK(nms_input_size < 1,
                    OP_LOGW(kFusedOpType.c_str(), "The input of %s is zero", nms_node_ptr->GetName().c_str()),
                    return NOT_CHANGED);
  GeTensorDesc pad_output_tensor_desc = nms_desc_ptr->GetInputDesc(0);

  // update pad node info
  auto nms_input_dims = pad_output_tensor_desc.GetShape().GetDims();
  FUSION_PASS_CHECK(nms_input_dims.size() >= 2 && PatternFusionUtil::IsUnknownShape(nms_input_dims[1]),
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "NMSWithMaskFusionPass cannot be applied for unknown shape."),
                    return FAILED);
  FUSION_PASS_CHECK(
      nms_input_dims.size() != 2 || nms_input_dims[1] != 5,
      OP_LOGW(kFusedOpType.c_str(), "The input dim of %s is not 2 dims or the second dimension of input is not 5",
              nms_node_ptr->GetName().c_str()),
      return NOT_CHANGED);

  // set pad output shape
  nms_input_dims[1] = 8;
  pad_output_tensor_desc.SetShape(GeShape(nms_input_dims));

  // update output origin shape of pad
  pad_output_tensor_desc.SetOriginShape(GeShape(nms_input_dims));
  FUSION_PASS_CHECK(pad_desc_ptr->AddOutputDesc("y", pad_output_tensor_desc) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "AddOutputDesc failed"), return FAILED);
  FUSION_PASS_CHECK(pad_desc_ptr->UpdateOutputDesc("y", pad_output_tensor_desc) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "UpdateOutputDesc failed"), return FAILED);
  FUSION_PASS_CHECK(nms_desc_ptr->UpdateInputDesc(0, pad_output_tensor_desc) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "UpdateInputDesc failed"), return FAILED);

  // delete attr from nms
  FUSION_PASS_CHECK(pad_desc_ptr->DelAttr("iou_threshold") != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Delete the attr of iou_threshold from nms."), return PARAM_INVALID);

  // set paddings attr for pad node
  FUSION_PASS_CHECK(
      !AttrUtils::SetListListInt(pad_desc_ptr, "paddings", std::vector<std::vector<int64_t>>{{0, 0}, {0, 3}}),
      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Set paddings attr for pad node."), return PARAM_INVALID);

  // add pad node to graph
  NodePtr pad_node_ptr = graph.AddNode(pad_desc_ptr);
  fusion_nodes.push_back(pad_node_ptr);
  FUSION_PASS_CHECK(pad_node_ptr == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The fusionNode: pad_node_ptr is null, fusion failed."),
                    return FAILED);

  // add the original edge of nms to pad
  auto nms_node_in_data_anchor = nms_node_ptr->GetInDataAnchor(0);
  FUSION_PASS_CHECK(nms_node_in_data_anchor == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "The fusionNode: nms_node_in_data_anchor is null, fusion failed."),
                    return PARAM_INVALID);

  FUSION_PASS_CHECK(GraphUtils::AddEdge(nms_node_in_data_anchor->GetPeerOutAnchor(),
                                        pad_node_ptr->GetInDataAnchor(0)) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add edge from fused node:%s to fusion node:%s failed.",
                            nms_node_ptr->GetName().c_str(), nms_node_ptr->GetName().c_str()),
                    return FAILED);

  // delete the first edge of nms
  FUSION_PASS_CHECK(
      GraphUtils::RemoveEdge(nms_node_in_data_anchor->GetPeerOutAnchor(), nms_node_in_data_anchor) != GRAPH_SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Remove input edge from fused node:%s.", nms_node_ptr->GetName().c_str()),
      return FAILED);

  // add the output of pad edge to nms
  FUSION_PASS_CHECK(GraphUtils::AddEdge(pad_node_ptr->GetOutDataAnchor(0), nms_node_in_data_anchor) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add edge from node:%s to node:%s failed.",
                            pad_node_ptr->GetName().c_str(), nms_node_ptr->GetName().c_str()),
                    return FAILED);

  fusion_nodes.push_back(nms_node_ptr);

  return SUCCESS;
}

REGISTER_PASS("NMSWithMaskFusionPass", BUILT_IN_GRAPH_PASS, NMSWithMaskFusionPass);
}  // namespace fe
