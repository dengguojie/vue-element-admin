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
 * \file batch_multi_class_nms_enable_vector_core_fusion_pass.cc
 * \brief
 */
#include "batch_multi_class_nms_enable_vector_core_fusion_pass.h"

#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "pattern_fusion_util.h"
#include "tbe_fusion_pass_util.h"

namespace fe {

static const string FUSED_OP_TYPE = "BatchMultiClassNonMaxSuppression";
static const string PATTERN_FUSEDNODE = "FusedNodeBatchMultiClassNonMaxSuppressionEnableVector";
static const string FUSED_NODE = "BatchMultiClassNonMaxSuppression";
static const std::string ATTR_OP_SPECIFIED_ENGINE_NAME = "_specified_engine_name";
static const std::string ATTR_OP_SPECIFIED_KERNEL_LIB_NAME = "_specified_kernel_lib_name";

vector<FusionPattern *> BatchMultiClassNonMaxSuppressionEnableVectorCoreFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern = new(std::nothrow) FusionPattern("BatchMultiClassNonMaxSuppressionFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE, "New a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns = {pattern};
  return patterns;
}

Status BatchMultiClassNonMaxSuppressionEnableVectorCoreFusionPass::Fusion(ge::ComputeGraph &graph,
                                                                          Mapping &mapping,
                                                                          std::vector<ge::NodePtr> &fusionNodes) {
  if (!InitCoreCount()) {
    OP_LOGD(FUSED_NODE, "init core count failed.");
    return NOT_CHANGED;
  }

  if (!NeedEnableVectorCore(mapping)) {
    OP_LOGD(FUSED_NODE, "need not enable vector core");
    return NOT_CHANGED;
  }

  const string split_node = PATTERN_FUSEDNODE;
  const vector<int64_t> split_input_idx = {0, 1, 2, 3};
  const vector<string> bmc_nms_input_names = {"boxes", "scores", "clip_window", "num_valid_boxes"};
  const vector<string> bmc_nms_output_names = {"nmsed_boxes", "nmsed_scores", "nmsed_classes", "nmsed_num"};
  const int32_t split_axis = 0;
  const int32_t split_num = 2; // split for aicore and vector core
  auto fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(!fused_node,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "fusedNode is null, fusion failed."),
                    return NOT_CHANGED);

  auto op_desc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(!op_desc,
                    VECTOR_FUSION_INNER_ERR_REPORT(fused_node->GetName(), "op_desc is null, fusion failed."),
                    return NOT_CHANGED);
  auto first_input_desc = op_desc->MutableInputDesc(0);
  FUSION_PASS_CHECK(!first_input_desc,
                    VECTOR_FUSION_INNER_ERR_REPORT(fused_node->GetName(), "Get first input desc failed."),
                    return NOT_CHANGED);
  auto batch_size = first_input_desc->MutableShape().GetDim(0);
  auto loop_times = GetAllCoreLoops(batch_size);
  uint32_t aicore_batch = loop_times * GetAiCoreCount();
  uint32_t vector_core_batch = batch_size - aicore_batch;
  const string split_node_name = fused_node->GetName() + "_split";

  ge::OpDescPtr vector_core_op_desc = AttrUtils::CloneOpDesc(op_desc);
  FUSION_PASS_CHECK(!vector_core_op_desc,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "Failed to create op desc for vector core."),
                    return NOT_CHANGED);
  vector_core_op_desc->SetName(fused_node->GetName() + "_vector_core");
  ge::AttrUtils::SetStr(vector_core_op_desc, ATTR_OP_SPECIFIED_ENGINE_NAME, "VectorEngine");
  ge::AttrUtils::SetStr(vector_core_op_desc, ATTR_OP_SPECIFIED_KERNEL_LIB_NAME, "VectorEngine");

  ge::OpDescPtr aicore_op_desc = AttrUtils::CloneOpDesc(op_desc);
  FUSION_PASS_CHECK(!aicore_op_desc,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "Failed to create op desc for ai core."),
                    return NOT_CHANGED);
  aicore_op_desc->SetName(fused_node->GetName() + "_ai_core");

  for (size_t i = 0; i < 4; i++) {
    ge::OpDescUtils::ClearInputDesc(aicore_op_desc, 0);
    ge::OpDescUtils::ClearOutputDesc(aicore_op_desc, 0);
    ge::OpDescUtils::ClearInputDesc(vector_core_op_desc, 0);
    ge::OpDescUtils::ClearOutputDesc(vector_core_op_desc, 0);
  }

  for (const auto &name : bmc_nms_output_names) {
    aicore_op_desc->AddOutputDesc(name, GeTensorDesc());
    vector_core_op_desc->AddOutputDesc(name, GeTensorDesc());
  }

  const vector<uint32_t> size_splits = {aicore_batch, vector_core_batch};
  // create split nodes
  vector<ge::NodePtr> split_nodes;
  split_nodes.reserve(split_input_idx.size());
  for (auto idx : split_input_idx) {
    auto input_desc = op_desc->MutableInputDesc(idx);
    if (!input_desc) {
      break;
    }

    ge::OpDescPtr split_node_desc;
    FUSION_PASS_MAKE_SHARED(split_node_desc =
                                std::make_shared<ge::OpDesc>(split_node_name + std::to_string(idx), "SplitVD"),
                            return FAILED);
    split_node_desc->AddInputDesc("x", *input_desc);
    split_node_desc->AddOutputDesc("y0", *input_desc);
    split_node_desc->AddOutputDesc("y1", *input_desc);
    ge::AttrUtils::SetListInt(split_node_desc, "size_splits", size_splits);
    ge::AttrUtils::SetInt(split_node_desc, "split_dim", split_axis);
    ge::AttrUtils::SetInt(split_node_desc, "num_split", split_num);
    auto split_node = graph.AddNode(split_node_desc);
    FUSION_PASS_CHECK(!split_node, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "add split node failed."),
                      return FAILED);
    FUSION_PASS_CHECK(split_node->InferShapeAndType() != GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "split InferShapeAndType failed."),
                      return FAILED);
    split_node_desc = split_node->GetOpDesc();
    auto first_output_desc = split_node_desc->MutableOutputDesc(0);
    FUSION_PASS_CHECK(!first_output_desc,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "Get first output desc failed."),
                      return FAILED);
    auto second_output_desc = split_node_desc->MutableOutputDesc(1);
    FUSION_PASS_CHECK(!second_output_desc,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "Get second output desc failed"),
                      return FAILED);
    aicore_op_desc->AddInputDesc(bmc_nms_input_names[idx], *first_output_desc);
    vector_core_op_desc->AddInputDesc(bmc_nms_input_names[idx], *second_output_desc);
    split_nodes.push_back(split_node);
    fusionNodes.push_back(split_node);
  }

  auto vector_core_bms_node = graph.AddNode(vector_core_op_desc);
  FUSION_PASS_CHECK(!vector_core_bms_node,
                    OP_LOGE(FUSED_OP_TYPE, "Failed to create op node for vector core."), return FAILED);
  FUSION_PASS_CHECK(vector_core_bms_node->InferShapeAndType() != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "vector_core_bms_node InferShapeAndType failed."),
                    return FAILED);
  fusionNodes.push_back(vector_core_bms_node);

  auto aicore_bms_node = graph.AddNode(aicore_op_desc);
  FUSION_PASS_CHECK(!aicore_bms_node,
                    OP_LOGE(FUSED_OP_TYPE, "Failed to create op node for aicore."), return FAILED);
  FUSION_PASS_CHECK(aicore_bms_node->InferShapeAndType() != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "aicore_bms_node InferShapeAndType failed."),
                    return FAILED);
  fusionNodes.push_back(aicore_bms_node);

  // create concat nodes
  vector<ge::NodePtr> concat_nodes;
  vector_core_op_desc = vector_core_bms_node->GetOpDesc();
  aicore_op_desc = aicore_bms_node->GetOpDesc();
  const size_t output_count = op_desc->GetOutputsSize();
  const std::string concat_node_name = fused_node->GetName() + "_concat";
  concat_nodes.reserve(output_count);
  for (size_t idx = 0; idx < output_count; idx++) {
    auto aicore_output_desc = aicore_op_desc->GetOutputDesc(idx);
    auto vector_core_output_desc = vector_core_op_desc->GetOutputDesc(idx);
    ge::OpDescPtr concat_desc;
    FUSION_PASS_MAKE_SHARED(concat_desc =
                                std::make_shared<ge::OpDesc>(concat_node_name + std::to_string(idx), "ConcatD"),
                            return FAILED);
    concat_desc->AddInputDesc("x0", aicore_output_desc);
    concat_desc->AddInputDesc("x1", vector_core_output_desc);
    concat_desc->AddOutputDesc("y", aicore_output_desc);
    ge::AttrUtils::SetInt(concat_desc, "concat_dim", split_axis);
    ge::AttrUtils::SetInt(concat_desc, "N", split_num);
    auto concat_node = graph.AddNode(concat_desc);
    FUSION_PASS_CHECK(!concat_node,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "Failed to create concat node."),
                      return FAILED);
    FUSION_PASS_CHECK(concat_node->InferShapeAndType() != GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "concat InferShapeAndType failed."),
                      return FAILED);
    concat_nodes.push_back(concat_node);
    fusionNodes.push_back(concat_node);
  }

  // link split edge
  const size_t split_count = split_nodes.size();
  for (size_t input_idx = 0; input_idx < split_count; input_idx++) {
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(input_idx)->GetPeerOutAnchor(),
                                              split_nodes[input_idx]->GetInDataAnchor(0)) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(fused_node->GetName(), "AddEdge edge failed."),
                      return FAILED);

    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(split_nodes[input_idx]->GetOutDataAnchor(0),
                                              aicore_bms_node->GetInDataAnchor(input_idx)) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(fused_node->GetName(),
                                                     "AddEdge split to aicore bms_node failed."),
                      return FAILED);

    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(split_nodes[input_idx]->GetOutDataAnchor(1),
                                              vector_core_bms_node->GetInDataAnchor(input_idx)) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(fused_node->GetName(),
                                                     "AddEdge split to vector core bms_node failed."),
                      return FAILED);

    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fused_node->GetInDataAnchor(input_idx)->GetPeerOutAnchor(),
                                                 fused_node->GetInDataAnchor(input_idx)) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(fused_node->GetName(), "RemoveEdge edge failed."),
                      return FAILED);
  }

  // link concat edge
  const auto concat_node_count = concat_nodes.size();
  for (size_t output_idx = 0; output_idx < concat_node_count; output_idx++) {
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(aicore_bms_node->GetOutDataAnchor(output_idx),
                                              concat_nodes[output_idx]->GetInDataAnchor(0)) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(fused_node->GetName(), "AddEdge edge failed."),
                      return FAILED);

    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(vector_core_bms_node->GetOutDataAnchor(output_idx),
                                              concat_nodes[output_idx]->GetInDataAnchor(1)) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(fused_node->GetName(),
                                                     "AddEdge split to aicore bms_node failed."),
                      return FAILED);

    auto output_i = fused_node->GetOutDataAnchor(output_idx);
    FUSION_PASS_CHECK(!output_i,
                      VECTOR_FUSION_INNER_ERR_REPORT(fused_node->GetName(),
                                                     "GetOutDataAnchor %zu failed.", output_idx),
                                                     return FAILED);
    for (auto output_anchor:output_i->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(output_i, output_anchor) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(fused_node->GetName(),
                                                       "RemoveEdge bms node to output node failed."),
                        return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(concat_nodes[output_idx]->GetOutDataAnchor(0),
                                                output_anchor) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(fused_node->GetName(),
                                                       "AddEdge concat to output node failed."),
                        return FAILED);
    }
  }

  FUSION_PASS_CHECK(graph.RemoveNode(fused_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE,
                                                   "Remove Node [%s] failed",
                                                   fused_node->GetName().c_str()),
                    return FAILED);

  return SUCCESS;
}

bool BatchMultiClassNonMaxSuppressionEnableVectorCoreFusionPass::NeedEnableVectorCore(const Mapping &mapping) {
  if (!TbeEnableVectorCoreFusionBasePass::NeedEnableVectorCore(mapping)) {
    return false;
  }

  auto fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(!fused_node, OP_LOGE(FUSED_OP_TYPE, "fusedNode is null, fusion failed."),
                    return false);
  auto boxes_input_desc = fused_node->GetOpDesc()->GetInputDesc(0);
  auto batch_size = boxes_input_desc.MutableShape().GetDim(0);
  OP_LOGD(fused_node->GetName(), "batch size:[%ld].", batch_size);
  if (batch_size <= 0) {
    OP_LOGD(fused_node->GetName(), "batch size may be dynamic, can not to enable vector core.",
            batch_size);
    return false;
  }

  OP_LOGD(fused_node->GetName(), "batch_size[%ld], aicore loop times[%u], all core loop times[%u], enable vector core.",
          batch_size, GetAiCoreLoops(batch_size), GetAllCoreLoops(batch_size));
  if (GetAiCoreLoops(batch_size) == GetAllCoreLoops(batch_size)) {
    OP_LOGD(fused_node->GetName(),
            "aicore loop times is equal to all core loop times, need not to enable vector core.",
            batch_size);
    return false;
  }
  return true;
}

REGISTER_PASS("BatchMultiClassNonMaxSuppressionFusionPass4VectorCore", BUILT_IN_GRAPH_PASS,
              BatchMultiClassNonMaxSuppressionEnableVectorCoreFusionPass);
} // namespace fe
