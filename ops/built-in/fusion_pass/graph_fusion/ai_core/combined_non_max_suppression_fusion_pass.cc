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
 * \file combined_non_max_suppression_fusion_pass.cpp
 * \brief combined_non_max_suppression fusion pass
 */
#include "combined_non_max_suppression_fusion_pass.h"
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>
#include <algorithm>
#include "op_log.h"
#include "error_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "tbe_fusion_pass_util.h"

namespace fe {
static const string PATTERN_FUSEDNODE = "FusedNodeCombinedNonMaxSuppression";
static const string FUSED_NODE = "CombinedNonMaxSuppression";

std::vector<FusionPattern*> CombinedNonMaxSuppressionFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("CombinedNonMaxSuppressionFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

Status CombinedNonMaxSuppressionFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                   std::vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define CombinedNonMaxSuppressionFusionPass fusion begin.");
  ge::NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fused_node == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "get fused_node is null, fusion failed."),
                    return NOT_CHANGED);

  // refresh op attr is_input_const and check whether the input[2:5] is const
  if (!TbeFusionPassUtil::UpdateAttrIsInputConst(fused_node)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "update the node attr is_input_const failed, donot go to aicore.");
    return NOT_CHANGED;
  }
  std::vector<bool> is_input_const = fused_node->GetOpDesc()->GetIsInputConst();
  FUSION_PASS_CHECK(is_input_const.size() != 6,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "the input num is not equal 6, donot go to aicore."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(!(is_input_const[2] && is_input_const[3] && is_input_const[4] && is_input_const[5]),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "the input 2-5 is not const node, donot go to aicore."),
                    return NOT_CHANGED);

  // check whether use aicore CombinedNonMaxSuppression op
  Operator nms_op = ge::OpDescUtils::CreateOperatorFromNode(fused_node);
  // get max_output_size_per_class const value
  std::vector<int64_t> max_size_per_class;
  if (!TbeFusionPassUtil::GetConstIntData(nms_op, "max_output_size_per_class", max_size_per_class)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "get max_output_size_per_class value failed.");
    return NOT_CHANGED;
  }
  FUSION_PASS_CHECK(max_size_per_class.empty(),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "the max_output_size_per_class can not be empty"),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(max_size_per_class[0] == 0,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "max_output_size_per_class cannot be 0."),
                    return NOT_CHANGED);
  max_size_per_class[0] = (max_size_per_class[0] + 15) / 16;
  max_size_per_class[0] = max_size_per_class[0] * 16;
  // get input 1 boxes shape
  ge::GeTensorDesc boxes_input_desc = fused_node->GetOpDesc()->GetInputDesc(0);
  vector<int64_t> boxes_input_shape = boxes_input_desc.GetShape().GetDims();
  auto boxes_size = boxes_input_shape.size();
  FUSION_PASS_CHECK(boxes_size != 4,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "the input 0 shape length must be 4. donot go to aicore."),
                    return NOT_CHANGED);
  // get input 2 score shape
  ge::GeTensorDesc scores_input_desc = fused_node->GetOpDesc()->GetInputDesc(1);
  vector<int64_t> scores_input_shape = scores_input_desc.GetShape().GetDims();
  FUSION_PASS_CHECK(scores_input_shape.size() != 3,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "the input 1 shape length must be 3. donot go to aicore"),
                    return NOT_CHANGED);
  int64_t input_boxes_classes_num = boxes_input_shape[2];
  int64_t input_scores_classes_num = scores_input_shape[2];
  auto input_classes_num = max(input_scores_classes_num, input_boxes_classes_num);

  // input_classes_num > 200 or (input_classes_num*8 + 10)*max_size_per_class > l1 size will not go ti aicore
  FUSION_PASS_CHECK(input_classes_num > 200,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "the class num is more than 200. donot go to aicore"),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(((input_classes_num*8 + 10) * max_size_per_class[0]) >= 1048576 / 2,
                    OP_LOGW(FUSED_OP_TYPE.c_str(),
                            "the (class num*max_size_per_class) is too large. donot go to aicore"),
                    return NOT_CHANGED);

  // insert transpose at input 0
  vector<int64_t> perm_boxes_list = {0, 2, 3, 1};
  AddTransposeBeforeNode(fused_node, 0, perm_boxes_list, graph);
  // insert transpose at input 1
  vector<int64_t> perm_score_list = {0, 2, 1};
  AddTransposeBeforeNode(fused_node, 1, perm_score_list, graph);

  // do infer for fused node again, and update fused node output shape
  ge::GeTensorDesc output_desc = fused_node->GetOpDesc()->GetOutputDesc(0);
  vector<int64_t> ori_output_shape = output_desc.GetShape().GetDims();
  FUSION_PASS_CHECK(ori_output_shape.size() < 3,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "can not get output shape. shape size less then 3!"),
                    return NOT_CHANGED);
  vector<int64_t> output_shape_vec;
  output_shape_vec.push_back(ori_output_shape[0]);
  output_shape_vec.push_back(ori_output_shape[2]);
  output_shape_vec.push_back(ori_output_shape[1]);
  ge::GeShape output_shape(output_shape_vec);
  output_desc.SetShape(output_shape);
  output_desc.SetOriginShape(output_shape);
  // update fused node output info
  auto op_output_desc = fused_node->GetOpDesc();
  op_output_desc->UpdateOutputDesc(0, output_desc);

  // insert transpose at output 0
  AddTransposeAfterNode(fused_node, 0, perm_score_list, graph);

  // for performance change nms_valid_num shape from [batch] to [batch, 8] and insert a SliceD
  ge::GeTensorDesc nms_num_desc = fused_node->GetOpDesc()->GetOutputDesc(3);
  vector<int64_t> ori_nms_num_shape = nms_num_desc.GetShape().GetDims();
  FUSION_PASS_CHECK(ori_nms_num_shape.empty(),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "can not get output nms valid num shape. shape is empty!"),
                    return NOT_CHANGED);
  vector<int64_t> new_shape_vec;
  new_shape_vec.push_back(ori_nms_num_shape[0]);
  new_shape_vec.push_back(8);

  // new a slice node
  std::shared_ptr<ge::OpDesc> reduce_desc = nullptr;
  std::string reduce_desc_name = fused_node->GetName() + "_Output_3_reduce";
  reduce_desc = std::make_shared<ge::OpDesc>(reduce_desc_name, "StridedSliceD");
  FUSION_PASS_CHECK(reduce_desc == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add reduce after valid num is null, fusion failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(reduce_desc->AddOutputDesc("y", nms_num_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add output y for reduce after valid num is null, fusion failed."),
                    return NOT_CHANGED);

  ge::GeShape new_shape(new_shape_vec);
  nms_num_desc.SetShape(new_shape);
  nms_num_desc.SetOriginShape(new_shape);
  op_output_desc->UpdateOutputDesc(3, nms_num_desc);
  FUSION_PASS_CHECK(reduce_desc->AddInputDesc("x", nms_num_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input x for reduce after valid num is null, fusion failed."),
                    return NOT_CHANGED);
  ge::AttrUtils::SetListInt(reduce_desc, "begin", {0, 0});
  ge::AttrUtils::SetListInt(reduce_desc, "end", {ori_nms_num_shape[0], 1});
  ge::AttrUtils::SetListInt(reduce_desc, "strides", {1, 1});
  ge::AttrUtils::SetInt(reduce_desc, "begin_mask", 0);
  ge::AttrUtils::SetInt(reduce_desc, "end_mask", 0);
  ge::AttrUtils::SetInt(reduce_desc, "ellipsis_mask", 0);
  ge::AttrUtils::SetInt(reduce_desc, "new_axis_mask", 0);
  ge::AttrUtils::SetInt(reduce_desc, "shrink_axis_mask", 0);

  // add node to graph
  ge::NodePtr reduce_node = graph.AddNode(reduce_desc);
  // add edge GraphUtils node output with other node input
  for (auto inDataAnchor : fused_node->GetOutDataAnchor(3)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(3), inDataAnchor) != SUCCESS,
                      OP_LOGW(FUSED_OP_TYPE.c_str(), "Remove edge failed."), return NOT_CHANGED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(reduce_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGW(FUSED_OP_TYPE.c_str(), "Add edge failed."), return NOT_CHANGED);
  }

  // add input for reduce node
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(fused_node->GetOutDataAnchor(3),
                                            reduce_node->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "AddEdge edge failed."), return NOT_CHANGED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define CombinedNonMaxSuppressionFusionPass fusion end");
  return SUCCESS;
}

REGISTER_PASS("CombinedNonMaxSuppressionFusionPass", BUILT_IN_GRAPH_PASS,
              CombinedNonMaxSuppressionFusionPass);
}  // namespace fe

