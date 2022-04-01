/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file depthwise_df_fusion_pass.cpp
 * \brief depthwise_df_fusion_pass
 */
#include "depthwise_df_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "conv_fusion_pass_utils.h"
#include "quant_host_cpu_op_common.h"
#include "op_log.h"
#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "securec.h"

using namespace std;
using namespace ge;

namespace fe {
static const string PATTERN_DEPTHWISE_DF = "DepthwiseConv2DBackpropInputD";
static const char* DEPTHWISE = "DepthwiseConv2DBackpropInputD";
static const char* DEPTHWISE_DYN = "DepthwiseConv2DBackpropInput";
const std::vector<ge::Format> kFormatList = {FORMAT_NCHW, FORMAT_HWCN};
const std::vector<int32_t> kTransposePerm = {1, 0, 2, 3};
const int32_t MAX_DIM_NUM = 4;
const int32_t NCHW_N_DIM = 0;
const int32_t NCHW_C_DIM = 1;

vector<FusionPattern*> DepthwiseDfFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  // define DepthwiseDfFusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("DepthwiseDfFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  // define origin graph
  pattern->AddOpDesc(PATTERN_DEPTHWISE_DF, {DEPTHWISE, DEPTHWISE_DYN}).SetOutput(PATTERN_DEPTHWISE_DF);
  patterns.push_back(pattern);

  return patterns;
}

ge::NodePtr DepthwiseDfFusionPass::CreateTransposeNode(ge::ComputeGraph& graph, const ge::GeTensorDesc& filter_desc,
                                                       const string& node_name) {
  ge::Format filter_format = filter_desc.GetOriginFormat();
  FUSION_PASS_CHECK(filter_format != ge::FORMAT_NCHW,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "only filter_format is nchw, need insert transpose node."),
                    return nullptr);
  ge::DataType filter_dtype = filter_desc.GetDataType();
  vector<int64_t> filter_output_shape_vec = filter_desc.GetShape().GetDims();
  std::swap(filter_output_shape_vec[NCHW_N_DIM], filter_output_shape_vec[NCHW_C_DIM]);

  ge::GeTensorDesc transpose_output_desc;
  transpose_output_desc.SetOriginShape(ge::GeShape(filter_output_shape_vec));
  transpose_output_desc.SetShape(ge::GeShape(filter_output_shape_vec));
  transpose_output_desc.SetOriginFormat(filter_format);
  transpose_output_desc.SetFormat(filter_format);
  transpose_output_desc.SetOriginDataType(filter_dtype);
  transpose_output_desc.SetDataType(filter_dtype);

  ge::OpDescPtr transpose_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(transpose_desc = std::make_shared<ge::OpDesc>(node_name + "/TransposeD", "TransposeD"),
                          return nullptr);
  FUSION_PASS_CHECK(transpose_desc->AddInputDesc("x", filter_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add input desc x to transposed."), return nullptr);
  FUSION_PASS_CHECK(transpose_desc->AddOutputDesc("y", transpose_output_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add output desc ]y to transposed."), return nullptr);
  ge::AttrUtils::SetListInt(transpose_desc, "perm", kTransposePerm);
  NodePtr transpose_node = graph.AddNode(transpose_desc);
  FUSION_PASS_CHECK(transpose_node == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add transpose node into graph."), return nullptr);
  return transpose_node;
}

Status DepthwiseDfFusionPass::InsertNode(const ge::OutDataAnchorPtr& src, const ge::InDataAnchorPtr& dst,
                                         ge::NodePtr& new_node) {
  ge::NodePtr src_node = src->GetOwnerNode();
  ge::NodePtr dst_node = dst->GetOwnerNode();
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(src, dst) != SUCCESS,
                    OP_LOGE(dst_node->GetName().c_str(), "Remove ori_filter edge error."), return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(src, new_node->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGE(src_node->GetName().c_str(), "Add edge to node %s failed.", new_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(new_node->GetOutDataAnchor(0), dst) != SUCCESS,
                    OP_LOGE(new_node->GetName().c_str(), "Add edge to node %s failed.", dst_node->GetName().c_str()),
                    return FAILED);
  return SUCCESS;
}

Status DepthwiseDfFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGI("Enter DepthwiseDfFusionPass");
  ge::NodePtr depthwise_node = GetNodeFromMapping(PATTERN_DEPTHWISE_DF, mapping);
  OpDescPtr depthwise_desc = depthwise_node->GetOpDesc();
  OP_LOGD(depthwise_desc->GetName().c_str(), "dealing with df fusion");

  uint32_t index = 0;
  if (depthwise_desc->GetType() == DEPTHWISE_DYN) {
    index = 1;
  }

  ge::GeTensorDesc filter_desc = depthwise_desc->GetInputDesc(index);
  ge::Format origin_format = filter_desc.GetOriginFormat();

  int groups = 0;
  ge::AttrUtils::GetInt(depthwise_node->GetOpDesc(), "groups", groups);
  FUSION_PASS_CHECK(groups <= 0, OP_LOGE(FUSED_OP_TYPE.c_str(), "groups should not be less or equal with 0."),
                    return PARAM_INVALID);

  FUSION_PASS_CHECK(std::find(kFormatList.begin(), kFormatList.end(), origin_format) == kFormatList.end(),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "input origin_format only support NCHW and HWCN."),
                    return PARAM_INVALID);

  vector<int64_t> filter_shape_vec = filter_desc.GetShape().GetDims();
  vector<int64_t> filter_reset_shape_vec;
  vector<int64_t> fractal_reset_shape_vec;
  FUSION_PASS_CHECK(!ConvFusionPassUtils::GetResizeDepthwiseFilter(filter_shape_vec, origin_format, groups,
                                                                   filter_reset_shape_vec, fractal_reset_shape_vec),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "get filter resize shape failed."), return PARAM_INVALID);

  ge::NodePtr transpose_node = CreateTransposeNode(graph, filter_desc, depthwise_node->GetName());
  ge::GeTensorDesc reshape_input_desc =
      transpose_node == nullptr ? filter_desc : transpose_node->GetOpDesc()->GetOutputDesc(0);

  filter_desc.SetShape(ge::GeShape(filter_reset_shape_vec));
  filter_desc.SetOriginShape(ge::GeShape(filter_reset_shape_vec));

  std::string node_name = depthwise_node->GetName();
  ge::OpDescPtr reshape_desc_ptr =
      ConvFusionPassUtils::CreateReshape(depthwise_node->GetName(), reshape_input_desc, filter_desc);
  ge::NodePtr reshape_node = graph.AddNode(reshape_desc_ptr);
  FUSION_PASS_CHECK(reshape_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add reshape into graph."),
                    return PARAM_INVALID);
  auto in_anchor = depthwise_node->GetInDataAnchor(index);
  FUSION_PASS_CHECK(in_anchor == nullptr || in_anchor->GetPeerOutAnchor() == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "filter in anchor is nullptr or peer anchor is nullptr."),
                    return FAILED);
  FUSION_PASS_CHECK(InsertNode(in_anchor->GetPeerOutAnchor(), in_anchor, reshape_node) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to link reshape to graph."), return FAILED);
  if (transpose_node != nullptr) {
    in_anchor = reshape_node->GetInDataAnchor(0);
    FUSION_PASS_CHECK(InsertNode(in_anchor->GetPeerOutAnchor(), in_anchor, transpose_node) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to link transpose node."), return FAILED);
  }
  // update filter desc
  FUSION_PASS_CHECK(depthwise_node->GetOpDesc()->UpdateInputDesc(index, filter_desc) != GRAPH_SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "update depthwise filter input desc failed."), return FAILED);

  OP_LOGI("Leave DepthwiseDfFusionPass");
  return SUCCESS;
}
REGISTER_PASS("DepthwiseDfFusionPass", BUILT_IN_GRAPH_PASS, DepthwiseDfFusionPass);
}  // namespace fe
