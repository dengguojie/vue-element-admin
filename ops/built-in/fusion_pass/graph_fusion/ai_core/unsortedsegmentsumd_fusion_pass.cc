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
 * \file unsortedsegmentsumd_fusion_pass.cpp
 * \brief
 */
#include "unsortedsegmentsumd_fusion_pass.h"

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
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {

static const string PATTERN_UNSORTED_SEGMENT_SUM = "UnsortedSegmentSumD";
static const string PATTERNUNSORTEDSEGMENT_SUM = "UnsortedSegmentSumD";

vector<FusionPattern*> UnsortedSegmentSumdFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("UnsortedSegmentSumd1to8Fusion");

  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_UNSORTED_SEGMENT_SUM, {PATTERNUNSORTEDSEGMENT_SUM})
      .SetOutput(PATTERN_UNSORTED_SEGMENT_SUM);

  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define UnsortedSegmentSumdFusionPass pattern end");
  return patterns;
}

Status UnsortedSegmentSumdFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                             vector<ge::NodePtr>& fusion_nodes) {
  ge::NodePtr unsorted_segment_sum_d_node = GetNodeFromMapping(PATTERN_UNSORTED_SEGMENT_SUM, mapping);
  FUSION_PASS_CHECK(unsorted_segment_sum_d_node == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "UnsortedSegmentSumD node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(unsorted_segment_sum_d_node->GetInDataNodes().size() != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(),
                            "Input node of UnsortedSegmentSumD node size is [%zu], which not equal to 2.",
                            unsorted_segment_sum_d_node->GetInDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(unsorted_segment_sum_d_node->GetOutDataNodes().size() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(),
                            "Output node of UnsortedSegmentSumD node size is [%zu], which not equal to 1.",
                            unsorted_segment_sum_d_node->GetOutDataNodes().size()),
                    return NOT_CHANGED);

  ge::OpDescPtr unsorted_segment_sum_d_desc = unsorted_segment_sum_d_node->GetOpDesc();
  FUSION_PASS_CHECK(unsorted_segment_sum_d_desc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "UnsortedSegmentSumD's op_desc is null, fusion failed."),
                    return PARAM_INVALID);

  // get shape, if not[xxx,1] return
  const auto &input_desc = unsorted_segment_sum_d_desc->GetInputDesc(0);
  std::vector<int64_t> x_dims = input_desc.GetOriginShape().GetDims();
  FUSION_PASS_CHECK(x_dims.size() <= 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(),
                            "UnsortedSegmentSumD node no need fusion, input x original shape rank=%zu.", x_dims.size()),
                    return NOT_CHANGED);
  if (PatternFusionUtil::IsUnknownShape(x_dims.back())) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "UnsortedSegmentSumdFusionPass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }
  FUSION_PASS_CHECK(x_dims.back() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(),
                            "UnsortedSegmentSumD node no need fusion, x_dims[%zu]=%lld", x_dims.size() - 1,
                            x_dims.back()),
                    return NOT_CHANGED);
  ge::DataType input_dtype = input_desc.GetDataType();
  int pad_dim_size = 0;
  if (input_dtype == DT_FLOAT) {
    pad_dim_size = 8;
  } else if (input_dtype == DT_FLOAT16) {
    pad_dim_size = 16;
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "UnsortedSegmentSumD dtype is not in (float32, float16), no need change");
    return NOT_CHANGED;
  }

  // output  [16000, 39, 1] -> [16000, 39, 8]
  ge::GeShape input_shape = input_desc.GetShape();
  input_shape.SetDim(x_dims.size() - 1, pad_dim_size);
  ge::GeTensorDesc concat_output_desc(input_shape, input_desc.GetFormat(), input_dtype);
  concat_output_desc.SetOriginShape(input_shape);
  concat_output_desc.SetOriginFormat(input_desc.GetFormat());
  concat_output_desc.SetOriginDataType(input_dtype);

  // 1 create ConcatD node
  ge::OpDescPtr concat_d_desc = std::make_shared<ge::OpDesc>(unsorted_segment_sum_d_node->GetName() + "/" + "ConcatD",
                                                             "ConcatD");
  FUSION_PASS_CHECK(concat_d_desc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "ConcatD create op_desc fusion failed."), return PARAM_INVALID);
  for (int32_t i = 0; i < pad_dim_size; i++) {
    FUSION_PASS_CHECK(concat_d_desc->AddInputDesc("x" + std::to_string(i), input_desc) != GRAPH_SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add input failed."), return FAILED);
  }
  FUSION_PASS_CHECK(concat_d_desc->AddOutputDesc("y", concat_output_desc) != GRAPH_SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add output failed."), return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(concat_d_desc, "concat_dim", x_dims.size() - 1),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "set attr concat_dim failed."), return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(concat_d_desc, "N", pad_dim_size),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "set attr N failed."), return FAILED);
  ge::NodePtr concat_d_node = graph.AddNode(concat_d_desc);
  FUSION_PASS_CHECK(concat_d_node == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "ConcatD add node failed."),
                    return PARAM_INVALID);
  fusion_nodes.push_back(concat_d_node);

  // 2 create UnsortedSegmentSumDPad node
  std::shared_ptr<ge::OpDesc> unsorted_segment_sum_d_pad_desc =
      std::make_shared<ge::OpDesc>(unsorted_segment_sum_d_node->GetName() + "/" + "UnsortedSegmentSumDPad",
                                   "UnsortedSegmentSumD");
  FUSION_PASS_CHECK(unsorted_segment_sum_d_pad_desc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "UnsortedSegmentSumDPad create op_desc failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(unsorted_segment_sum_d_pad_desc->AddInputDesc(concat_output_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add input failed."), return FAILED);
  FUSION_PASS_CHECK(
      unsorted_segment_sum_d_pad_desc->AddInputDesc(unsorted_segment_sum_d_desc->GetInputDesc(1)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "add input failed."), return FAILED);
  const auto &ori_output_desc = unsorted_segment_sum_d_desc->GetOutputDesc(0);
  ge::GeShape output_shape = ori_output_desc.GetShape();
  vector<int64_t> output_shape_dims = ori_output_desc.GetShape().GetDims();
  output_shape.SetDim(output_shape_dims.size() - 1, pad_dim_size);
  ge::GeTensorDesc output_desc(output_shape, ori_output_desc.GetFormat(), ori_output_desc.GetDataType());
  output_desc.SetOriginShape(output_shape);
  output_desc.SetOriginFormat(ori_output_desc.GetFormat());
  output_desc.SetOriginDataType(ori_output_desc.GetDataType());
  FUSION_PASS_CHECK(unsorted_segment_sum_d_pad_desc->AddOutputDesc(output_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add output failed."), return FAILED);
  ge::AttrUtils::SetInt(unsorted_segment_sum_d_pad_desc, "num_segments", output_shape_dims[0]);
  ge::NodePtr unsorted_segment_sum_d_pad_node = graph.AddNode(unsorted_segment_sum_d_pad_desc);
  fusion_nodes.push_back(unsorted_segment_sum_d_pad_node);

  // 3 create SliceD node
  std::shared_ptr<ge::OpDesc> slice_d_desc =
      std::make_shared<ge::OpDesc>(unsorted_segment_sum_d_node->GetName() + "/" + "SliceD", "SliceD");
  FUSION_PASS_CHECK(slice_d_desc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "SliceD create op_desc failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(slice_d_desc->AddInputDesc(output_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add input failed."), return FAILED);
  FUSION_PASS_CHECK(slice_d_desc->AddOutputDesc(unsorted_segment_sum_d_desc->GetOutputDesc(0)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add output failed."), return FAILED);
  ge::AttrUtils::SetListInt(slice_d_desc, "size", output_shape_dims);
  vector<int64_t> sliced_offsets(output_shape_dims.size(), 0);
  ge::AttrUtils::SetListInt(slice_d_desc, "offsets", sliced_offsets);
  ge::NodePtr slice_d_node = graph.AddNode(slice_d_desc);
  FUSION_PASS_CHECK(slice_d_node == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "slice_d_node is null, fusion failed."),
                    return PARAM_INVALID);
  fusion_nodes.push_back(slice_d_node);

  // connect edge
  const auto &peer_out_anchor0 = unsorted_segment_sum_d_node->GetInDataAnchor(0)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(peer_out_anchor0 == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Peer out anchor is null, node:%s, in_anchor_idx=0.",
                            unsorted_segment_sum_d_node->GetName().c_str()),
                    return PARAM_INVALID);
  for (int32_t i = 0; i < pad_dim_size; i++) {
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(peer_out_anchor0, concat_d_node->GetInDataAnchor(i)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between %s and %s failed.",
                              peer_out_anchor0->GetOwnerNode()->GetName().c_str(), concat_d_node->GetName().c_str()),
                      return FAILED);
  }

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(concat_d_node->GetOutDataAnchor(0),
                                            unsorted_segment_sum_d_pad_node->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between %s and %s failed.",
                            concat_d_node->GetName().c_str(), unsorted_segment_sum_d_pad_node->GetName().c_str()),
                    return FAILED);
  const auto &peer_out_anchor1 = unsorted_segment_sum_d_node->GetInDataAnchor(1)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(peer_out_anchor1 == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Peer out anchor is null, node:%s, in_anchor_idx=1.",
                            unsorted_segment_sum_d_node->GetName().c_str()),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(peer_out_anchor1,
                                            unsorted_segment_sum_d_pad_node->GetInDataAnchor(1)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between %s and %s failed.",
                            peer_out_anchor1->GetOwnerNode()->GetName().c_str(),
                            unsorted_segment_sum_d_pad_node->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(unsorted_segment_sum_d_pad_node->GetOutDataAnchor(0),
                                            slice_d_node->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            unsorted_segment_sum_d_pad_node->GetName().c_str(), slice_d_node->GetName().c_str()),
                    return FAILED);
  for (const auto &in_data_anchor : unsorted_segment_sum_d_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(unsorted_segment_sum_d_node->GetOutDataAnchor(0),
                                                 in_data_anchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(slice_d_node->GetOutDataAnchor(0), in_data_anchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add data edge failed."), return FAILED);
  }

  // delete fused nodes
  FUSION_PASS_CHECK(graph.RemoveNode(unsorted_segment_sum_d_node) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove unsorted_segment_sum_d_node failed."), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "UnsortedSegmentSumdFusionPass graph fusion success!");
  return SUCCESS;
}

REGISTER_PASS("UnsortedSegmentSumdFusionPass", BUILT_IN_GRAPH_PASS, UnsortedSegmentSumdFusionPass);

}  // namespace fe
