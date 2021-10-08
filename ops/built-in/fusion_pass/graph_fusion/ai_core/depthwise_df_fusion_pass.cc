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
 * \file depthwise_df_fusion_pass.cpp
 * \brief depthwise_df_fusion_pass
 */
#include "depthwise_df_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

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
const int MAX_DIM_NUM = 4;

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

ge::NodePtr DepthwiseDfFusionPass::CreateReshapeNode(ge::ComputeGraph& graph,
                                                     vector<int64_t> src_shape,
                                                     vector<int64_t> tar_shape,
                                                     const ge::DataType& desc_type,
                                                     const std::string& node_name) {
  ge::GeShape input_shape(src_shape);
  ge::GeTensorDesc filter_input_desc(input_shape, FORMAT_HWCN, desc_type);
  filter_input_desc.SetShape(input_shape);
  filter_input_desc.SetOriginShape(input_shape);
  filter_input_desc.SetOriginDataType(desc_type);

  ge::GeShape output_shape(tar_shape);
  ge::GeTensorDesc filter_output_desc(output_shape, FORMAT_HWCN, desc_type);
  filter_output_desc.SetShape(output_shape);
  filter_output_desc.SetOriginShape(output_shape);
  filter_output_desc.SetOriginDataType(desc_type);

  ge::OpDescPtr reshape_desc;
  FUSION_PASS_MAKE_SHARED((reshape_desc = std::make_shared<ge::OpDesc>(
                          node_name + "/Reshape", "Reshape")), return nullptr);
  FUSION_PASS_CHECK(reshape_desc->AddInputDesc("x", filter_input_desc) != GRAPH_SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add input desc x to reshape."), return nullptr);
  FUSION_PASS_CHECK(reshape_desc->AddOutputDesc("y", filter_output_desc) != GRAPH_SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add input desc y to reshape."), return nullptr);
  ge::AttrUtils::SetListInt(reshape_desc, "shape", tar_shape);

  auto reshape_node = graph.AddNode(reshape_desc);
  FUSION_PASS_CHECK(reshape_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add reshape to graph."),
                    return nullptr);
  return reshape_node;
}

Status DepthwiseDfFusionPass::InsertNode(const ge::OutDataAnchorPtr &src, const ge::InDataAnchorPtr &dst,
                                         ge::NodePtr& new_node) {
  ge::NodePtr src_node = src->GetOwnerNode();
  ge::NodePtr dst_node = dst->GetOwnerNode();

  if (new_node->GetOpDesc()->UpdateInputDesc(0, src_node->GetOpDesc()->GetOutputDesc(src->GetIdx())) != GRAPH_SUCCESS) {
    OP_LOGI(new_node->GetName().c_str(), "update input_desc failed.");
    return FAILED;
  }
  if (new_node->GetOpDesc()->UpdateOutputDesc(0, dst_node->GetOpDesc()->GetInputDesc(dst->GetIdx())) != GRAPH_SUCCESS) {
    OP_LOGI(new_node->GetName().c_str(), "update output_desc failed.");
    return FAILED;
  }
  if(ge::GraphUtils::RemoveEdge(src, dst) != SUCCESS) {
    OP_LOGE(dst_node->GetName().c_str(), "Remove ori_filter edge error.");
    return FAILED;
  }
  if(ge::GraphUtils::AddEdge(src, new_node->GetInDataAnchor(0)) != SUCCESS) {
    OP_LOGE(src_node->GetName().c_str(), "Add edge to node %s failed.", new_node->GetName().c_str());
    return FAILED;
  }
  if(ge::GraphUtils::AddEdge(new_node->GetOutDataAnchor(0), dst)!= SUCCESS) {
   OP_LOGE(new_node->GetName().c_str(), "Add edge to node %s failed.", dst_node->GetName().c_str());
    return FAILED; 
  }
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
  ge::DataType filter_dtype = filter_desc.GetDataType();

  if (origin_format != FORMAT_HWCN) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "filter format is not HWCN, no need to do fusion.");
    return NOT_CHANGED;
  }

  vector<int64_t> filter_src_shape = filter_desc.GetShape().GetDims();
  vector<int64_t> filter_tar_shape = filter_src_shape;
  filter_tar_shape[2] = 1;
  filter_tar_shape[3] = filter_src_shape[2] * filter_src_shape[3];

  // update filter input desc
  filter_desc.SetShape(ge::GeShape(filter_tar_shape));
  filter_desc.SetOriginShape(ge::GeShape(filter_tar_shape));
  FUSION_PASS_CHECK(depthwise_node->GetOpDesc()->UpdateInputDesc(index, filter_desc) != GRAPH_SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "update depthwise filter input desc failed."),
                    return FAILED);


  std::string node_name = depthwise_node->GetName();
  ge::NodePtr reshape_node = CreateReshapeNode(graph, filter_src_shape, filter_tar_shape, filter_dtype, node_name);
  FUSION_PASS_CHECK(reshape_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add reshape to graph."),
                    return FAILED);
  auto in_anchor = depthwise_node->GetInDataAnchor(index);
  FUSION_PASS_CHECK(in_anchor == nullptr || in_anchor->GetPeerOutAnchor() == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "filter in anchor is nullptr or peer anchor is nullptr."),
                    return FAILED);
  FUSION_PASS_CHECK(InsertNode(in_anchor->GetPeerOutAnchor(), in_anchor, reshape_node) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add reshape to graph."),
                    return FAILED);

  OP_LOGI("Leave DepthwiseDfFusionPass");
  return SUCCESS;
}
REGISTER_PASS("DepthwiseDfFusionPass", BUILT_IN_GRAPH_PASS, DepthwiseDfFusionPass);
}
