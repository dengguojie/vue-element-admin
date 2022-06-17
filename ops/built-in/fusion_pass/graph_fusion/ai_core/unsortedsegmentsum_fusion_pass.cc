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
 * \file unsortedsegmentsum_fusion_pass.cpp
 * \brief
 */
#include "unsortedsegmentsum_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include "fp16_t.hpp"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {
static const int32_t PAD_DIM_SIZE_EIGHT = 8;
static const int32_t PAD_DIM_SIZE_SIXTEEN = 16;
static const string PATTERN_UNSORTED_SEGMENT_SUM = "UnsortedSegmentSum";
static const string PATTERNUNSORTEDSEGMENT_SUM = "UnsortedSegmentSum";

/*  UnsortedSegmentSum                -->         Concat & UnsortedSegmentSum & Slice

                                           concat_dim   x0    x1  x2  x3  x4  x5  x6  x7
                                                    \    |    /   /   /   /   /   /   /
                                                     \   |   /   /   /   /   /   /   /
                                                      Concat
                                                         \
   x     segment_ids  num_segments                        \    segment_ids  num_segments
    \         |         /                                  \         |         /
     \        |        /                                    \        |        /
      UnsortedSegmentSum              -->                    UnsortedSegmentSum
              |                                                      \
              |                                                       \   offsets  size
              y                                                        \     |     /
                                                                        \    |    /
                                                                           Slice
                                      -->                                    |
                                                                             |
                                                                             y
*/

vector<FusionPattern*> UnsortedSegmentSumFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("UnsortedSegmentSum1to8Fusion");

  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_UNSORTED_SEGMENT_SUM, {PATTERNUNSORTEDSEGMENT_SUM})
      .SetOutput(PATTERN_UNSORTED_SEGMENT_SUM);

  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE, "define UnsortedSegmentSumFusionPass pattern end");
  return patterns;
}

Status UnsortedSegmentSumFusionPass::AddConcatConstNode(const int32_t pad_dim_size,
                                                        int32_t const_value,
                                                        ge::NodePtr& const_node,
                                                        std::shared_ptr<ge::OpDesc>& concat_desc) {
  ge::GeTensorPtr const_ptr = nullptr;
  ge::GeTensorDesc const_desc;
  const_desc.SetFormat(FORMAT_ND);
  const_desc.SetOriginFormat(FORMAT_ND);
  const_desc.SetShape(ge::GeShape(std::vector<int64_t>({})));
  const_desc.SetOriginShape(ge::GeShape(std::vector<int64_t>({})));
  const_desc.SetDataType(ge::DT_INT32);
  const_desc.SetOriginDataType(ge::DT_INT32);
  FUSION_PASS_MAKE_SHARED(
      (const_ptr = std::make_shared<ge::GeTensor>(const_desc, reinterpret_cast<uint8_t*>(&const_value),
                                                  sizeof(DT_INT32))),
      const_ptr = nullptr;
      return PARAM_INVALID);
  std::map<int, ge::GeTensorPtr> weights_map;
  weights_map[0] = const_ptr;
  ge::OpDescUtils::SetWeights(*const_node, weights_map);
  auto const_input_node = OpDescUtils::GetConstInputs(const_node);
  FUSION_PASS_CHECK(const_input_node.size() < 1,
                    OP_LOGE(FUSED_OP_TYPE, " the num of const nodes is less than 1."),
                    return FAILED);
  NodePtr const_input = const_input_node[0];
  FUSION_PASS_CHECK(const_input == nullptr, OP_LOGE(FUSED_OP_TYPE, "const_input is null."),
                    return PARAM_INVALID);
  const_input->GetOpDesc()->SetType("Const");
  std::map<string, uint32_t> input_name_id = {{"concat_dim", 0}};
  for (int32_t i = 0; i < pad_dim_size; i++) {
      input_name_id.insert(std::make_pair("x" + std::to_string(i), i + 1));
  }
  const_node->GetOpDesc()->UpdateInputName(input_name_id);

  return SUCCESS;
}

Status UnsortedSegmentSumFusionPass::AddSliceConstNode(const std::vector<int64_t>& const_shape,
                                                       std::vector<int32_t>& const_value,
                                                       ge::NodePtr& const_node,
                                                       std::shared_ptr<ge::OpDesc>& slice_desc) {
  ge::GeTensorPtr offsets_const_ptr = nullptr;
  ge::GeTensorDesc offsets_const_desc;
  offsets_const_desc.SetFormat(FORMAT_ND);
  offsets_const_desc.SetOriginFormat(FORMAT_ND);
  offsets_const_desc.SetShape(ge::GeShape(std::vector<int64_t>(const_shape)));
  offsets_const_desc.SetOriginShape(ge::GeShape(std::vector<int64_t>(const_shape)));
  offsets_const_desc.SetDataType(ge::DT_INT32);
  offsets_const_desc.SetOriginDataType(ge::DT_INT32);
  std::vector<int64_t> offsets_const_value(const_value.size(), 0);
  FUSION_PASS_MAKE_SHARED(
      (offsets_const_ptr = std::make_shared<ge::GeTensor>(offsets_const_desc,
                                                          reinterpret_cast<uint8_t*>(offsets_const_value.data()),
                                                          const_value.size() * sizeof(DT_INT32))),
      offsets_const_ptr = nullptr; return PARAM_INVALID);
  ge::GeTensorPtr size_const_ptr = nullptr;
  ge::GeTensorDesc size_const_desc;
  size_const_desc.SetFormat(FORMAT_ND);
  size_const_desc.SetOriginFormat(FORMAT_ND);
  size_const_desc.SetShape(ge::GeShape(std::vector<int64_t>(const_shape)));
  size_const_desc.SetOriginShape(ge::GeShape(std::vector<int64_t>(const_shape)));
  size_const_desc.SetDataType(ge::DT_INT32);
  size_const_desc.SetOriginDataType(ge::DT_INT32);
  FUSION_PASS_MAKE_SHARED(
      (size_const_ptr = std::make_shared<ge::GeTensor>(size_const_desc,
                                                       reinterpret_cast<uint8_t*>(const_value.data()),
                                                       const_value.size() * sizeof(DT_INT32))),
      size_const_ptr = nullptr; return PARAM_INVALID);
  size_t size_idx = 2;
  std::map<int, ge::GeTensorPtr> weights_map;
  weights_map[1] = offsets_const_ptr;
  weights_map[size_idx] = size_const_ptr;
  ge::OpDescUtils::SetWeights(*const_node, weights_map);
  auto const_input_node = OpDescUtils::GetConstInputs(const_node);
  FUSION_PASS_CHECK(const_input_node.size() < size_idx,
                    OP_LOGE(FUSED_OP_TYPE, " the num of const nodes is less than 2."), return FAILED);
  NodePtr offsets_const_input = const_input_node[0];
  FUSION_PASS_CHECK(offsets_const_input == nullptr, OP_LOGE(FUSED_OP_TYPE, "const_input1 is null."),
                    return PARAM_INVALID);
  offsets_const_input->GetOpDesc()->SetType("Const");
  NodePtr size_const_input = const_input_node[1];
  FUSION_PASS_CHECK(size_const_input == nullptr, OP_LOGE(FUSED_OP_TYPE, "const_input2 is null."),
                    return PARAM_INVALID);
  size_const_input->GetOpDesc()->SetType("Const");
  const_node->GetOpDesc()->UpdateInputName({{"x", 0}, {"offsets", 1}, {"size", size_idx}});
  return SUCCESS;
}

Status UnsortedSegmentSumFusionPass::CheckUnsortedSegmentSumNode(const ge::NodePtr& unsorted_segment_sum_node) {
  FUSION_PASS_CHECK(
      unsorted_segment_sum_node == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "unsortedsegmentsum node is null, fusion failed."),
      return PARAM_INVALID);

  size_t input_num = 3;
  FUSION_PASS_CHECK(
      unsorted_segment_sum_node->GetInDataNodes().size() != input_num,
      OP_LOGI(FUSED_OP_TYPE, "input node of UnsortedSegmentSum node size is [%zu], which not equal to [%zu].",
              unsorted_segment_sum_node->GetInDataNodes().size(), input_num),
      return NOT_CHANGED);

  ge::Tensor num_segments;
  ge::Operator unsorted_segment_sum_node_op = ge::OpDescUtils::CreateOperatorFromNode(unsorted_segment_sum_node);
  FUSION_PASS_CHECK(GRAPH_SUCCESS != unsorted_segment_sum_node_op.GetInputConstData("num_segments", num_segments),
      OP_LOGI(FUSED_OP_TYPE, "the num_segments of unsorted_segment_sum is not a constant."),
      return NOT_CHANGED);

  FUSION_PASS_CHECK(
      unsorted_segment_sum_node->GetOutDataNodes().size() != 1,
      OP_LOGI(FUSED_OP_TYPE, "output node of UnsortedSegmentSum node size is [%zu], which not equal to 1.",
              unsorted_segment_sum_node->GetOutDataNodes().size()),
      return NOT_CHANGED);

  return SUCCESS;
}

Status UnsortedSegmentSumFusionPass::AddEdges(const ge::NodePtr& unsorted_segment_sum_node,
                                              const int32_t pad_dim_size,
                                              ge::NodePtr& concat_node,
                                              ge::NodePtr& unsorted_segment_sum_pad_node,
                                              ge::NodePtr& slice_node) {
    // connect edge
  const auto& peer_out_anchor0 = unsorted_segment_sum_node->GetInDataAnchor(0)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(
      peer_out_anchor0 == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "peer out anchor is null, node:%s, in_anchor_idx=0.",
                                     unsorted_segment_sum_node->GetName().c_str()),
      return PARAM_INVALID);
  for (int32_t i = 0; i < pad_dim_size; i++) {
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(peer_out_anchor0, concat_node->GetInDataAnchor(i + 1)) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "add edge between %s and %s failed.",
                                                     peer_out_anchor0->GetOwnerNode()->GetName().c_str(),
                                                     concat_node->GetName().c_str()),
                      return FAILED);
  }
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0),
                                            unsorted_segment_sum_pad_node->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "add edge between %s and %s failed.",
                                                   concat_node->GetName().c_str(),
                                                   unsorted_segment_sum_pad_node->GetName().c_str()),
                    return FAILED);
  const auto& peer_out_anchor1 = unsorted_segment_sum_node->GetInDataAnchor(1)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(
      peer_out_anchor1 == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "peer out anchor is null, node:%s, in_anchor_idx=1.",
                                     unsorted_segment_sum_node->GetName().c_str()),
      return PARAM_INVALID);
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(peer_out_anchor1, unsorted_segment_sum_pad_node->GetInDataAnchor(1)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "add edge between %s and %s failed.",
                                     peer_out_anchor1->GetOwnerNode()->GetName().c_str(),
                                     unsorted_segment_sum_pad_node->GetName().c_str()),
      return FAILED);
  const auto& peer_out_anchor2 = unsorted_segment_sum_node->GetInDataAnchor(2)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(
      peer_out_anchor2 == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "peer out anchor is null, node:%s, in_anchor_idx=2.",
                                     unsorted_segment_sum_node->GetName().c_str()),
      return PARAM_INVALID);
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(peer_out_anchor2, unsorted_segment_sum_pad_node->GetInDataAnchor(2)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "add edge between %s and %s failed.",
                                     peer_out_anchor2->GetOwnerNode()->GetName().c_str(),
                                     unsorted_segment_sum_pad_node->GetName().c_str()),
      return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(unsorted_segment_sum_pad_node->GetOutDataAnchor(0),
                                            slice_node->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(
                        FUSED_OP_TYPE, "add edge between node %s. and node %s failed.",
                        unsorted_segment_sum_pad_node->GetName().c_str(), slice_node->GetName().c_str()),
                    return FAILED);
  for (const auto& in_data_anchor : unsorted_segment_sum_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(
        ge::GraphUtils::RemoveEdge(unsorted_segment_sum_node->GetOutDataAnchor(0), in_data_anchor) != SUCCESS,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(slice_node->GetOutDataAnchor(0), in_data_anchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "add data edge failed."), return FAILED);
  }

  return SUCCESS;
}

Status UnsortedSegmentSumFusionPass::CheckDims(const std::vector<int64_t>& x_dims) {
  // get shape, if not[xxx,1] return
  FUSION_PASS_CHECK(x_dims.size() <= 1,
                    OP_LOGI(FUSED_OP_TYPE,
                            "unsortedsegmentsum node don't need fusion, input x original shape rank=%zu.",
                            x_dims.size()),
                    return NOT_CHANGED);
  if (PatternFusionUtil::IsUnknownShape(x_dims.back())) {
    OP_LOGI(FUSED_OP_TYPE, "unsortedsegmentsumfusionpass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }
  FUSION_PASS_CHECK(x_dims.back() != 1,
                    OP_LOGI(FUSED_OP_TYPE, "unsortedsegmentsum node no need fusion, x_dims[%zu]=%lld",
                            x_dims.size() - 1, x_dims.back()),
                    return NOT_CHANGED);
  return SUCCESS;
}

Status UnsortedSegmentSumFusionPass::AddNodes(const ge::NodePtr& unsorted_segment_sum_node,
                                              const ge::OpDescPtr& unsorted_segment_sum_desc,
                                              const int32_t pad_dim_size,
                                              ge::ComputeGraph& graph,
                                              vector<ge::NodePtr>& fusion_nodes) {
  const auto& input_desc = unsorted_segment_sum_desc->GetInputDesc(0);
  std::vector<int64_t> x_dims = input_desc.GetOriginShape().GetDims();
  FUSION_PASS_CHECK(CheckDims(x_dims) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE, "check x_dims failed."),
                    return NOT_CHANGED);

  // output  [16000, 39, 1] -> [16000, 39, 8]
  ge::GeShape input_shape = input_desc.GetShape();
  input_shape.SetDim(x_dims.size() - 1, pad_dim_size);
  ge::DataType input_dtype = input_desc.GetDataType();
  ge::GeTensorDesc concat_output_desc(input_shape, input_desc.GetFormat(), input_dtype);
  concat_output_desc.SetOriginShape(input_shape);
  concat_output_desc.SetOriginFormat(input_desc.GetFormat());
  concat_output_desc.SetOriginDataType(input_dtype);

  ge::GeTensorDesc concat_dim_desc(ge::GeShape(std::vector<int64_t>({})), FORMAT_ND, ge::DT_INT32);
  concat_dim_desc.SetOriginFormat(FORMAT_ND);
  concat_dim_desc.SetOriginShape(ge::GeShape(std::vector<int64_t>({})));
  concat_dim_desc.SetOriginDataType(ge::DT_INT32);

  // 1 create Concat node
  std::shared_ptr<ge::OpDesc> concat_desc =
      std::make_shared<ge::OpDesc>(unsorted_segment_sum_node->GetName() + "/" + "Concat", "Concat");
  FUSION_PASS_CHECK(concat_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "concat create op_desc fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(concat_desc->AddInputDesc("concat_dim", concat_dim_desc) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "add input failed."), return FAILED);
  for (int32_t i = 0; i < pad_dim_size; i++) {
    FUSION_PASS_CHECK(concat_desc->AddInputDesc("x" + std::to_string(i), input_desc) != GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "add input of x failed."), return FAILED);
  }
  FUSION_PASS_CHECK(concat_desc->AddOutputDesc("y", concat_output_desc) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "add output failed."), return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(concat_desc, "N", pad_dim_size),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "set attr N failed."), return FAILED);
  ge::NodePtr concat_node = graph.AddNode(concat_desc);
  FUSION_PASS_CHECK(concat_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "concat add node failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(AddConcatConstNode(pad_dim_size, x_dims.size() - 1, concat_node, concat_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "add const node of concat is failed."),
                    return FAILED);
  fusion_nodes.push_back(concat_node);

  // 2 create UnsortedSegmentSumPad node
  std::shared_ptr<ge::OpDesc> unsorted_segment_sum_pad_desc = std::make_shared<ge::OpDesc>(
      unsorted_segment_sum_node->GetName() + "/" + "UnsortedSegmentSumPad", "UnsortedSegmentSum");
  FUSION_PASS_CHECK(
      unsorted_segment_sum_pad_desc == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "unsortedsegmentsumpad create op_desc failed."),
      return PARAM_INVALID);
  FUSION_PASS_CHECK(unsorted_segment_sum_pad_desc->AddInputDesc(concat_output_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "add input0 failed."), return FAILED);
  FUSION_PASS_CHECK(
      unsorted_segment_sum_pad_desc->AddInputDesc(unsorted_segment_sum_desc->GetInputDesc(1)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "add input1 failed."), return FAILED);
  FUSION_PASS_CHECK(
      unsorted_segment_sum_pad_desc->AddInputDesc(unsorted_segment_sum_desc->GetInputDesc(2)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "add input2 failed."), return FAILED);
  const auto& ori_output_desc = unsorted_segment_sum_desc->GetOutputDesc(0);
  ge::GeShape output_shape = ori_output_desc.GetShape();
  vector<int64_t> output_shape_dims = ori_output_desc.GetShape().GetDims();
  output_shape.SetDim(output_shape_dims.size() - 1, pad_dim_size);
  ge::GeTensorDesc output_desc(output_shape, ori_output_desc.GetFormat(), ori_output_desc.GetDataType());
  output_desc.SetOriginShape(output_shape);
  output_desc.SetOriginFormat(ori_output_desc.GetFormat());
  output_desc.SetOriginDataType(ori_output_desc.GetDataType());
  FUSION_PASS_CHECK(unsorted_segment_sum_pad_desc->AddOutputDesc(output_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "add output failed."), return FAILED);
  ge::NodePtr unsorted_segment_sum_pad_node = graph.AddNode(unsorted_segment_sum_pad_desc);
  fusion_nodes.push_back(unsorted_segment_sum_pad_node);

  // 3 create Slice node
  std::shared_ptr<ge::OpDesc> slice_desc =
      std::make_shared<ge::OpDesc>(unsorted_segment_sum_node->GetName() + "/" + "Slice", "Slice");
  FUSION_PASS_CHECK(slice_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "slice create op_desc failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(slice_desc->AddInputDesc(output_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "add input failed."), return FAILED);
  FUSION_PASS_CHECK(slice_desc->AddOutputDesc(unsorted_segment_sum_desc->GetOutputDesc(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "add output failed."), return FAILED);
  ge::NodePtr slice_node = graph.AddNode(slice_desc);
  FUSION_PASS_CHECK(slice_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "slice_node is null, fusion failed."),
                    return PARAM_INVALID);
  std::vector<int32_t> val;
  for (size_t i = 0; i < output_shape_dims.size(); i++) {
    val.emplace_back(output_shape_dims[i]);
  }
  std::vector<int64_t> shape = {static_cast<int64_t>(output_shape_dims.size())};
  FUSION_PASS_CHECK(AddSliceConstNode(shape, val, slice_node, slice_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "add const node of slice is failed."),
                    return FAILED);
  fusion_nodes.push_back(slice_node);
  return SUCCESS;
}

Status UnsortedSegmentSumFusionPass::GetPadDimSize(const ge::OpDescPtr& unsorted_segment_sum_desc,
                                                   int32_t& pad_dim_size) {
  const auto& input_desc = unsorted_segment_sum_desc->GetInputDesc(0);
  ge::DataType input_dtype = input_desc.GetDataType();
  if (input_dtype == DT_FLOAT) {
    pad_dim_size = PAD_DIM_SIZE_EIGHT;
  } else if (input_dtype == DT_FLOAT16) {
    pad_dim_size = PAD_DIM_SIZE_SIXTEEN;
  } else {
    OP_LOGI(FUSED_OP_TYPE, "unsortedSegmentSum dtype is not in (float32, float16), no need change");
    return NOT_CHANGED;
  }
  return SUCCESS;
}

Status UnsortedSegmentSumFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                            vector<ge::NodePtr>& fusion_nodes) {
  ge::NodePtr unsorted_segment_sum_node = GetNodeFromMapping(PATTERN_UNSORTED_SEGMENT_SUM, mapping);

  FUSION_PASS_CHECK(CheckUnsortedSegmentSumNode(unsorted_segment_sum_node) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE, "failed to check node."),
                    return NOT_CHANGED);

  ge::OpDescPtr unsorted_segment_sum_desc = unsorted_segment_sum_node->GetOpDesc();
  FUSION_PASS_CHECK(
      unsorted_segment_sum_desc == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "unsortedsegmentsum's op_desc is null, fusion failed."),
      return PARAM_INVALID);

  int32_t pad_dim_size = 0;
  FUSION_PASS_CHECK(GetPadDimSize(unsorted_segment_sum_desc, pad_dim_size) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE, "get pad_dim_size failed."),
                    return NOT_CHANGED);

  Status add_nodes_res = AddNodes(unsorted_segment_sum_node, unsorted_segment_sum_desc, pad_dim_size,
                                  graph, fusion_nodes);
  if (add_nodes_res != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE, "add node failed.");
    return add_nodes_res;
  }

  ge::NodePtr concat_node = fusion_nodes[DIM_0];
  ge::NodePtr unsorted_segment_sum_pad_node = fusion_nodes[DIM_1];
  ge::NodePtr slice_node  = fusion_nodes[DIM_2];

  // add edges
  FUSION_PASS_CHECK(AddEdges(unsorted_segment_sum_node, pad_dim_size,
                             concat_node, unsorted_segment_sum_pad_node, slice_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "add edge failed."),
                    return FAILED);
  // delete fused nodes
  FUSION_PASS_CHECK(graph.RemoveNode(unsorted_segment_sum_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "remove unsorted_segment_sum_node failed."),
                    return FAILED);
  OP_LOGI(FUSED_OP_TYPE, "UnsortedSegmentSumFusionPass graph fusion success!");
  return SUCCESS;
}

REGISTER_PASS("UnsortedSegmentSumFusionPass", BUILT_IN_GRAPH_PASS, UnsortedSegmentSumFusionPass);
}  // namespace fe
