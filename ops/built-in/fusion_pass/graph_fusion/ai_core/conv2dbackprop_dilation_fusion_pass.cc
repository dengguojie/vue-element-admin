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
 * \file conv2dbackprop_dilation_fusion_pass.cc
 * \brief conv2dbackprop_input dilation fusion pass
 * dilation->conv2dbackinput or conv2dbackinput->dilation
 */

#include "conv2dbackprop_dilation_fusion_pass.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "error_util.h"
#include "common/util/platform_info.h"

namespace fe {
static const string PATTERN_CONV2DBACKPROPINPUT = "Conv2DBackpropInputD";

vector<FusionPattern*> Conv2DbpInputDilationFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter Conv2DbpInputDilationFusionPass::DefinePatterns.");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern =
    new (std::nothrow) FusionPattern("Conv2DbpInputDilationFusionPass");
  FUSION_PASS_CHECK(
    pattern == nullptr,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
    return patterns);

  pattern->AddOpDesc(PATTERN_CONV2DBACKPROPINPUT,
                     {"Conv2DBackpropInputD"}).SetOutput(PATTERN_CONV2DBACKPROPINPUT);

  patterns.push_back(pattern);

  return patterns;
}

static Status generate_dilation_node(
  ge::ComputeGraph* graph,
  ge::GeTensorDesc* prev_out_desc,
  ge::GeTensorDesc* next_in_desc,
  ge::NodePtr* dilation_node,
  const vector<int64_t> strides,
  const vector<int64_t> pads,
  const std::string& basename,
  bool pre_dilation){
  ge::OpDescPtr dilation_desc;
  FUSION_PASS_MAKE_SHARED(
    (dilation_desc = std::make_shared<ge::OpDesc>(basename + "_dilation", "Dilation")),
    return FAILED);

  // nc1hwc0
  if (pre_dilation){
    dilation_desc->AddInputDesc("x", *prev_out_desc);
    auto next_in_shape = prev_out_desc->GetShape().GetDims();
    FUSION_PASS_CHECK(next_in_shape.empty() || next_in_shape.size() < 4,
                      ge::CommonRuntimeErrLog("Conv2dBackpropInputD", "shape size can not be less than 4"),
                      return FAILED);
    next_in_shape[2] = (next_in_shape[2] - 1) * strides[2] + 1;
    next_in_shape[3] = (next_in_shape[3] - 1) * strides[3] + 1;
    next_in_desc->SetShape(ge::GeShape(next_in_shape));

    auto next_in_ori_shape = prev_out_desc->GetOriginShape().GetDims();
    FUSION_PASS_CHECK(next_in_ori_shape.empty() || next_in_ori_shape.size() < 4,
                      ge::CommonRuntimeErrLog("Conv2dBackpropInputD", "shape size can not be less than 4"),
                      return FAILED);
    next_in_ori_shape[2] = (next_in_ori_shape[2] - 1) * strides[2] + 1;
    next_in_ori_shape[3] = (next_in_ori_shape[3] - 1) * strides[3] + 1;
    next_in_desc->SetOriginShape(ge::GeShape(next_in_ori_shape));

    dilation_desc->AddOutputDesc("y", *next_in_desc);
  } else {
    dilation_desc->AddOutputDesc("y", *next_in_desc);
    auto prev_out_shape = next_in_desc->GetShape().GetDims();
    FUSION_PASS_CHECK(prev_out_shape.empty() || prev_out_shape.size() < 4,
                      ge::CommonRuntimeErrLog("Conv2dBackpropInputD", "shape size can not be less than 4"),
                      return FAILED);
    prev_out_shape[2] = (prev_out_shape[2] - 1) / strides[2] + 1;
    prev_out_shape[3] = (prev_out_shape[3] - 1) / strides[3] + 1;
    prev_out_desc->SetShape(ge::GeShape(prev_out_shape));

    auto prev_out_ori_shape = next_in_desc->GetOriginShape().GetDims();
    FUSION_PASS_CHECK(prev_out_ori_shape.empty() || prev_out_ori_shape.size() < 4,
                      ge::CommonRuntimeErrLog("Conv2dBackpropInputD", "shape size can not be less than 4"),
                      return FAILED);
    prev_out_ori_shape[2] = (prev_out_ori_shape[2] - 1) / strides[2] + 1;
    prev_out_ori_shape[3] = (prev_out_ori_shape[3] - 1) / strides[3] + 1;
    prev_out_desc->SetOriginShape(ge::GeShape(prev_out_ori_shape));

    dilation_desc->AddInputDesc("x", *prev_out_desc);
  }

  vector<int64_t> dilations_new = strides;
  dilations_new.push_back(1);
  ge::AttrUtils::SetListInt(dilation_desc, "dilations", dilations_new);
  ge::AttrUtils::SetFloat(dilation_desc, "padding_value", 0.0);
  ge::AttrUtils::SetListInt(dilation_desc, "pads", pads);
  *dilation_node = graph->AddNode(dilation_desc);
  return SUCCESS;
}


Status Conv2DbpInputDilationFusionPass::Relink(
  ge::NodePtr out_backprop_node,
  ge::NodePtr dilation_node,
  ge::NodePtr conv2dbp_input_node,
  ge::NodePtr y_node,
  const int pre_anchor,
  const int sub_anchor,
  bool pre_dilation){
  if (pre_dilation){
    FUSION_PASS_CHECK(
      ge::GraphUtils::RemoveEdge(out_backprop_node->GetOutDataAnchor(0),
                                 conv2dbp_input_node->GetInDataAnchor(pre_anchor)) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
        "fail to remove edge between out_backprop_node and conv2dbp_input_node"),
      return FAILED);

    FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(out_backprop_node->GetOutDataAnchor(0),
                              dilation_node->GetInDataAnchor(0)) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
        "fail to add edge between out_backprop_node and dilation_node"),
      return FAILED);

    FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(dilation_node->GetOutDataAnchor(0),
                              conv2dbp_input_node->GetInDataAnchor(pre_anchor)) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
        "fail to add edge between dilation_node and conv2dbp_input_node"),
      return FAILED);

    FUSION_PASS_CHECK(
      conv2dbp_input_node->GetOpDesc()->UpdateInputDesc(
        pre_anchor, dilation_node->GetOpDesc()->GetOutputDesc(0)) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
        "fail to update input description of conv2dbp_input_node"),
      return FAILED);
  } else {
    FUSION_PASS_CHECK(
      ge::GraphUtils::RemoveEdge(conv2dbp_input_node->GetOutDataAnchor(sub_anchor),
                                 y_node->GetInDataAnchor(0)) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
        "fail to remove edge between conv2dbp_input_node and y_node"),
      return FAILED);

    FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(dilation_node->GetOutDataAnchor(0),
                              y_node->GetInDataAnchor(0)) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
        "fail to add edge between dilation_node and y_node"),
      return FAILED);

    FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(conv2dbp_input_node->GetOutDataAnchor(sub_anchor),
                              dilation_node->GetInDataAnchor(0)) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
        "fail to add edge between conv2dbp_input_node and dilation_node"),
      return FAILED);

    FUSION_PASS_CHECK(
      conv2dbp_input_node->GetOpDesc()->UpdateOutputDesc(
        sub_anchor, dilation_node->GetOpDesc()->GetInputDesc(0)) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
        "fail to update output description of conv2dbp_input_node"),
      return FAILED);
  }
  return SUCCESS;
}


Status Conv2DbpInputDilationFusionPass::Fusion(
  ge::ComputeGraph& graph, Mapping& mapping,
  vector<ge::NodePtr>& fusion_nodes){
  PlatformInfo platform_info;
  OptionalInfo opti_compilation_info;
  FUSION_PASS_CHECK(PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(
                    platform_info, opti_compilation_info) != SUCCESS,
                    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Get platform_info failed."),
                    return FAILED);
  bool cube_vector_split_flag = platform_info.ai_core_spec.cube_vector_split;
  FUSION_PASS_CHECK(!cube_vector_split_flag,
                    OP_LOGI(FUSED_OP_TYPE.c_str(),
                      "Not cube vector split, Dilation is not required."),
                    return NOT_CHANGED);

  ge::NodePtr conv2dbp_input_node = GetNodeFromMapping(PATTERN_CONV2DBACKPROPINPUT, mapping);
  FUSION_PASS_CHECK(conv2dbp_input_node == nullptr,
                    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
                      "Conv2DBackpropInputNode is null, fusion failed."),
                    return PARAM_INVALID);

  int filter_anchor = 0;
  int out_bp_anchor = 1;
  int y_anchor = 0;
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(conv2dbp_input_node);

  // get attr
  std::vector<int64_t> input_size;
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;

  FUSION_PASS_CHECK(op.GetAttr("strides", strides) != GRAPH_SUCCESS,
                    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
                      "op Conv2DBackpropInput get attribute strides failed or strides not exist"),
                    return FAILED);
  FUSION_PASS_CHECK(op.GetAttr("pads", pads) != GRAPH_SUCCESS,
                    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
                      "op Conv2DBackpropInput get attribute pads failed or pads not exist"),
                    return FAILED);
  FUSION_PASS_CHECK(op.GetAttr("input_size", input_size) != GRAPH_SUCCESS,
                    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
                      "op Conv2DBackpropInput get attribute input_size failed or input_size not exist"),
                    return FAILED);

  ge::NodePtr filter_node = conv2dbp_input_node->GetInDataAnchor(filter_anchor)->GetPeerOutAnchor()->GetOwnerNode();
  int filter_idx = conv2dbp_input_node->GetInDataAnchor(filter_anchor)->GetPeerOutAnchor()->GetIdx();

  ge::NodePtr out_backprop_node = conv2dbp_input_node->GetInDataAnchor(out_bp_anchor)\
                                  ->GetPeerOutAnchor()->GetOwnerNode();
  int out_backprop_idx = conv2dbp_input_node->GetInDataAnchor(out_bp_anchor)->GetPeerOutAnchor()->GetIdx();

  ge::NodePtr y_node = conv2dbp_input_node->GetOutDataAnchor(y_anchor)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
  int y_idx = conv2dbp_input_node->GetOutDataAnchor(y_anchor)->GetPeerInDataAnchors().at(0)->GetIdx();

  // get info of Node
  ge::GeTensorDesc filter_out_desc = filter_node->GetOpDesc()->GetOutputDesc(filter_idx);
  ge::GeTensorDesc out_backprop_out_desc = out_backprop_node->GetOpDesc()->GetOutputDesc(out_backprop_idx);
  ge::GeTensorDesc y_desc = y_node->GetOpDesc()->GetInputDesc(y_idx);
  ge::GeTensorDesc conv2dbp_input_outbackprop_desc = conv2dbp_input_node\
                                                     ->GetOpDesc()->GetInputDesc(out_backprop_idx);
  ge::GeTensorDesc conv2dbp_input_y_desc = conv2dbp_input_node->GetOpDesc()->GetInputDesc(y_idx);

  // get shape
  auto filter_ori_shape = filter_out_desc.GetOriginShape().GetDims();
  FUSION_PASS_CHECK(filter_ori_shape.empty() || filter_ori_shape.size() < 4,
                    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "filter size can not be less than 4"),
                    return FAILED);
  bool pre_dilation = false;
  bool need_dilation_flag = false;

  // if pre_dilation is true, Dilation -> conv2dbp_input
  // else conv2dbp_input -> Dilation
  if (filter_ori_shape[2] > 1 || filter_ori_shape[3] > 1)
    pre_dilation = true;
  FUSION_PASS_CHECK(strides.empty() || strides.size() < 4 || strides[2] <= 0 || strides[3] <= 0,
                    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "filter size can not be less than 4"),
                    return FAILED);
  for (unsigned int i = 0; i < strides.size(); ++i){
    if (strides[i] > 1){
      need_dilation_flag = true;
      break;
    }
  }

  if (!need_dilation_flag){
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Because stride is 1, Dilation is not required");
    return NOT_CHANGED;
  }

  ge::NodePtr dilation_node = nullptr;
  auto basename = conv2dbp_input_node->GetName();
  if (pre_dilation){
    FUSION_PASS_CHECK(
      generate_dilation_node(
        &graph, &out_backprop_out_desc, &conv2dbp_input_outbackprop_desc,
        &dilation_node, strides, pads, basename, pre_dilation) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to generate dilation node"),
      return FAILED);
  } else {
    FUSION_PASS_CHECK(
      generate_dilation_node(
        &graph, &conv2dbp_input_y_desc, &y_desc,
        &dilation_node, strides, pads, basename, pre_dilation) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to generate dilation node"),
      return FAILED);
  }

  FUSION_PASS_CHECK(
    Relink(out_backprop_node, dilation_node, conv2dbp_input_node,
           y_node, out_bp_anchor, y_anchor, pre_dilation) != SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to generate relink node"),
    return FAILED);

  fusion_nodes.push_back(dilation_node);
  op.SetAttr("strides", {1, 1, 1, 1});

  if (!pre_dilation){
    std::vector<int64_t> input_size_new = input_size;
    input_size_new[2] = (input_size_new[2] - 1) / strides[2] + 1;
    input_size_new[3] = (input_size_new[3] - 1) / strides[3] + 1;
    op.SetAttr("input_size", input_size_new);
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "End Conv2DbpInputDilationFusionPass.");

  return SUCCESS;
}

REGISTER_PASS("Conv2DbpInputDilationFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, Conv2DbpInputDilationFusionPass);
} // namespace fe