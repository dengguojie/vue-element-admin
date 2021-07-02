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
static const string kAttrOrgFmt  = "origin_format";

#define CHECK_POSITION(position)                                                                        \
  {                                                                                                     \
    if (position == std::string::npos) {                                                                \
        OP_LOGI(PATTERN_CONV2DBACKPROPINPUT.c_str(), "get position failed:%s:%d", #position, position); \
        return FAILED;                                                                                  \
    }                                                                                                   \
  }

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

vector<int64_t> post_get_hw_not_pad(
  const vector<int64_t> out_backprop_ori_shape,
  const vector<int64_t> strides,
  size_t pos_backprop_out_h,
  size_t pos_backprop_out_w) {
  int64_t h_dilation = (out_backprop_ori_shape[pos_backprop_out_h] - 1) * strides[pos_backprop_out_h] + 1;
  int64_t w_dilation = (out_backprop_ori_shape[pos_backprop_out_w] - 1) * strides[pos_backprop_out_w] + 1;
  vector<int64_t> post_dilation_hw;
  post_dilation_hw.push_back(h_dilation);
  post_dilation_hw.push_back(w_dilation);
  return post_dilation_hw;
}

vector<int64_t> post_get_pad_value(
  const vector<int64_t> post_dilation_hw,
  const vector<int64_t> dx_shape,
  size_t pos_y_h,
  size_t pos_y_w) {
  int pad_down = dx_shape[pos_y_h] - post_dilation_hw[0];
  int pad_right = dx_shape[pos_y_w] - post_dilation_hw[1];
  vector<int64_t> post_pad_hw;
  post_pad_hw.push_back(pad_down);
  post_pad_hw.push_back(pad_right);
  return post_pad_hw;
}

vector<int64_t> pre_get_pad_value(
  const vector<int64_t> post_dilation_hw,
  const vector<int64_t> dx_shape,
  size_t pos_y_h,
  size_t pos_y_w,
  const vector<int64_t> filter_shape,
  size_t pos_filter_h,
  size_t pos_filter_w,
  const vector<int64_t> pads
  ) {
  int pad_down = dx_shape[pos_y_h] - filter_shape[pos_filter_h] + pads[0] + pads[2] + 1 - post_dilation_hw[0];
  int pad_right = dx_shape[pos_y_w] - filter_shape[pos_filter_w] + pads[1] + pads[3] + 1 - post_dilation_hw[1];
  vector<int64_t> post_pad_hw;
  post_pad_hw.push_back(pad_down);
  post_pad_hw.push_back(pad_right);
  return post_pad_hw;
}

static Status generate_pre_dilation_node(
  ge::ComputeGraph* graph,
  ge::GeTensorDesc* prev_out_desc,
  ge::GeTensorDesc* next_in_desc,
  ge::NodePtr* dilation_node,
  const vector<int64_t> strides,
  const std::string& basename){
  ge::OpDescPtr dilation_desc;
  FUSION_PASS_MAKE_SHARED(
    (dilation_desc = std::make_shared<ge::OpDesc>(basename + "_dilation", "Dilation")),
    return FAILED);

  std::string fmt_str;
  AttrUtils::GetStr(next_in_desc, kAttrOrgFmt, fmt_str);
  size_t pos_h = fmt_str.find('H');
  CHECK_POSITION(pos_h);
  size_t pos_w = fmt_str.find('W');
  CHECK_POSITION(pos_w);

  // nc1hwc0
  dilation_desc->AddInputDesc("x", *prev_out_desc);
  auto next_in_shape = prev_out_desc->GetShape().GetDims();
  next_in_shape[2] = (next_in_shape[2] - 1) * strides[pos_h] + 1;
  next_in_shape[3] = (next_in_shape[3] - 1) * strides[pos_w] + 1;
  next_in_desc->SetShape(ge::GeShape(next_in_shape));

  auto next_in_ori_shape = prev_out_desc->GetOriginShape().GetDims();
  next_in_ori_shape[pos_h] = (next_in_ori_shape[pos_h] - 1) * strides[pos_h] + 1;
  next_in_ori_shape[pos_w] = (next_in_ori_shape[pos_w] - 1) * strides[pos_w] + 1;
  next_in_desc->SetOriginShape(ge::GeShape(next_in_ori_shape));

  dilation_desc->AddOutputDesc("y", *next_in_desc);
  vector<int64_t> dilations_new = {1, 1, 1, 1, 1};
  dilations_new[2] = strides[pos_h];
  dilations_new[3] = strides[pos_w];

  ge::AttrUtils::SetListInt(dilation_desc, "dilations", dilations_new);
  ge::AttrUtils::SetFloat(dilation_desc, "padding_value", 0.0);
  *dilation_node = graph->AddNode(dilation_desc);
  return SUCCESS;
}

static Status generate_post_dilation_node(
  ge::ComputeGraph* graph,
  ge::GeTensorDesc* prev_out_desc,
  ge::GeTensorDesc* next_in_desc,
  ge::GeTensorDesc* conv2dbp_input_outbackprop_desc,
  ge::NodePtr* dilation_node,
  const vector<int64_t> strides,
  const vector<int64_t> pads,
  const std::string& basename){
  ge::OpDescPtr dilation_desc;
  FUSION_PASS_MAKE_SHARED(
    (dilation_desc = std::make_shared<ge::OpDesc>(basename + "_dilation", "Dilation")),
    return FAILED);

  std::string y_fmt_str;
  AttrUtils::GetStr(prev_out_desc, kAttrOrgFmt, y_fmt_str);
  size_t pos_y_h = y_fmt_str.find('H');
  CHECK_POSITION(pos_y_h);
  size_t pos_y_w = y_fmt_str.find('W');
  CHECK_POSITION(pos_y_w);

  std::string outbackprop_fmt_str;
  AttrUtils::GetStr(conv2dbp_input_outbackprop_desc, kAttrOrgFmt, outbackprop_fmt_str);
  size_t pos_outbackprop_out_h = outbackprop_fmt_str.find('H');
  CHECK_POSITION(pos_outbackprop_out_h);
  size_t pos_outbackprop_out_w = outbackprop_fmt_str.find('W');
  CHECK_POSITION(pos_outbackprop_out_w);
  auto out_backprop_ori_shape = conv2dbp_input_outbackprop_desc->GetOriginShape().GetDims();

  dilation_desc->AddOutputDesc("y", *next_in_desc);
  auto prev_out_shape = next_in_desc->GetShape().GetDims();
  prev_out_shape[2] = out_backprop_ori_shape[pos_outbackprop_out_h];
  prev_out_shape[3] = out_backprop_ori_shape[pos_outbackprop_out_w];
  prev_out_desc->SetShape(ge::GeShape(prev_out_shape));

  auto prev_out_ori_shape = next_in_desc->GetOriginShape().GetDims();
  prev_out_ori_shape[pos_y_h] = out_backprop_ori_shape[pos_outbackprop_out_h];
  prev_out_ori_shape[pos_y_w] = out_backprop_ori_shape[pos_outbackprop_out_w];
  prev_out_desc->SetOriginShape(ge::GeShape(prev_out_ori_shape));
  dilation_desc->AddInputDesc("x", *prev_out_desc);

  vector<int64_t> dilations_new = {1, 1, 1, 1, 1};
  dilations_new[2] = strides[pos_outbackprop_out_h];
  dilations_new[3] = strides[pos_outbackprop_out_w];
  ge::AttrUtils::SetListInt(dilation_desc, "dilations", dilations_new);
  ge::AttrUtils::SetFloat(dilation_desc, "padding_value", 0.0);
  const vector<int64_t> pads_in = {0, 0, pads[pos_outbackprop_out_h], pads[pos_outbackprop_out_w], 0};
  ge::AttrUtils::SetListInt(dilation_desc, "pads", pads_in);
  *dilation_node = graph->AddNode(dilation_desc);
  return SUCCESS;
}

static Status generate_pad_node(
  ge::ComputeGraph* graph,
  ge::GeTensorDesc* prev_out_desc,
  ge::GeTensorDesc* next_in_desc,
  ge::NodePtr* pad_node,
  const vector<int64_t> pad_hw,
  const std::string &basename) {
  ge::OpDescPtr pad_desc;
  FUSION_PASS_MAKE_SHARED(
    (pad_desc = std::make_shared<ge::OpDesc>(basename + "_pad", "PadD")),
    return FAILED);
  std::string fmt_str;
  AttrUtils::GetStr(next_in_desc, kAttrOrgFmt, fmt_str);
  size_t pos_h = fmt_str.find('H');
  CHECK_POSITION(pos_h);
  size_t pos_w = fmt_str.find('W');
  CHECK_POSITION(pos_w);
  pad_desc->AddOutputDesc("y", *next_in_desc);

  int64_t pad_down = pad_hw[0];
  int64_t pad_after = pad_hw[1];
  auto prev_out_shape = next_in_desc->GetShape().GetDims();
  prev_out_shape[2] -= pad_down;
  prev_out_shape[3] -= pad_after;
  prev_out_desc->SetShape(ge::GeShape(prev_out_shape));
  
  auto prev_out_ori_shape = next_in_desc->GetOriginShape().GetDims();
  prev_out_ori_shape[pos_h] -= pad_down;
  prev_out_ori_shape[pos_w] -= pad_after;
  prev_out_desc->SetOriginShape(ge::GeShape(prev_out_ori_shape));
  pad_desc->AddInputDesc("x", *prev_out_desc);

  vector<vector<int64_t>> pads;
  pads.push_back({0, 0});
  pads.push_back({0, 0});
  pads.push_back({0, pad_down});
  pads.push_back({0, pad_after});
  pads.push_back({0, 0});
  AttrUtils::SetListListInt(pad_desc, "paddings", pads);
  *pad_node = graph->AddNode(pad_desc);
  return SUCCESS;
}

Status Conv2DbpInputDilationFusionPass::Relink(
  ge::NodePtr out_backprop_node,
  ge::NodePtr dilation_node,
  ge::NodePtr conv2dbp_input_node,
  ge::NodePtr y_node,
  ge::NodePtr pad_node,
  const int pre_anchor,
  const int sub_anchor,
  bool pre_dilation,
  const vector<int> &y_idxs) {
  if (pre_dilation){
    OP_LOGD(FUSED_OP_TYPE.c_str(), "pre relink");
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
    OP_LOGD(FUSED_OP_TYPE.c_str(), "post relink");
    for (int i : y_idxs) {
    FUSION_PASS_CHECK(
      ge::GraphUtils::RemoveEdge(conv2dbp_input_node->GetOutDataAnchor(sub_anchor),
                                 y_node->GetInDataAnchor(i)) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
        "fail to remove edge between conv2dbp_input_node and y_node"),
      return FAILED);
    if (pad_node != nullptr) { 
        FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(pad_node->GetOutDataAnchor(0),
          y_node->GetInDataAnchor(i)) != SUCCESS,
          ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
          "fail to add the edge between pad_node and y_node"),
          return FAILED);
    } else {
        FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(dilation_node->GetOutDataAnchor(0),
        y_node->GetInDataAnchor(i)) != SUCCESS,
        ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
        "fail to add edge between dilation_node and y_node"),
        return FAILED);
    }
   }
   if(pad_node != nullptr) {
         FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(dilation_node->GetOutDataAnchor(0),
                                pad_node->GetInDataAnchor(0)) != SUCCESS,
                                ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
                                "fail to add edge between dilation_node  and pad_node"),
                                return FAILED);
         FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(conv2dbp_input_node->GetOutDataAnchor(sub_anchor),
                                dilation_node->GetInDataAnchor(0)) != SUCCESS,
                                ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
                                "fail to add edge between conv2dbp_input_node  and dilation_node"),
                                return FAILED);
 
   } else {
        FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(conv2dbp_input_node->GetOutDataAnchor(sub_anchor),
                                dilation_node->GetInDataAnchor(0)) != SUCCESS,
                                ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), 
                                "fail to add edge between conv2dbp_input_node and dilation_node"),
                                return FAILED);
   }

   FUSION_PASS_CHECK(conv2dbp_input_node->GetOpDesc()->UpdateOutputDesc(sub_anchor,
                   dilation_node->GetOpDesc()->GetInputDesc(0)) !=  SUCCESS,
                   ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
                   "fail to update output description of conv2dbp_input_node"),
                   return FAILED);

   if (pad_node != nullptr) {
        FUSION_PASS_CHECK(dilation_node->GetOpDesc()->UpdateOutputDesc(
                                0, pad_node->GetOpDesc()->GetInputDesc(0)) !=  SUCCESS,
                                ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
                                "fail to update output description of dilation_node "),
                                return FAILED);
   
   }
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "finish  relink");
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

  ge::NodePtr out_backprop_node = conv2dbp_input_node->GetInDataAnchor(out_bp_anchor)\
                                  ->GetPeerOutAnchor()->GetOwnerNode();
  int out_backprop_idx = conv2dbp_input_node->GetInDataAnchor(out_bp_anchor)->GetPeerOutAnchor()->GetIdx();
  ge::GeTensorDesc out_backprop_out_desc = out_backprop_node->GetOpDesc()->GetOutputDesc(out_backprop_idx);
  vector<int> y_idxs;
  ge::NodePtr y_node = conv2dbp_input_node->GetOutDataAnchor(y_anchor)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
  for (auto in_data_anchor : conv2dbp_input_node->GetOutDataAnchor(y_anchor)->GetPeerInDataAnchors()) {
        OP_LOGD(FUSED_OP_TYPE.c_str(), "all idx is:");
        OP_LOGD(FUSED_OP_TYPE.c_str(), "idx = %d",in_data_anchor->GetIdx());
        y_idxs.push_back(in_data_anchor->GetIdx());
  }
  int y_idx = conv2dbp_input_node->GetOutDataAnchor(y_anchor)->GetPeerInDataAnchors().at(0)->GetIdx();

  ge::GeTensorDesc y_desc = y_node->GetOpDesc()->GetInputDesc(y_idx);
  ge::GeTensorDesc filter_desc = conv2dbp_input_node->GetOpDesc()->GetInputDesc(filter_anchor);
  ge::GeTensorDesc conv2dbp_input_outbackprop_desc = conv2dbp_input_node\
                                                     ->GetOpDesc()->GetInputDesc(out_bp_anchor);
  ge::GeTensorDesc conv2dbp_input_y_desc = conv2dbp_input_node->GetOpDesc()->GetOutputDesc(y_anchor);

  // get shape
  auto filter_ori_shape = filter_desc.GetOriginShape().GetDims();
  auto out_backprop_ori_shape = conv2dbp_input_outbackprop_desc.GetOriginShape().GetDims();
  auto y_shape = conv2dbp_input_y_desc.GetOriginShape().GetDims();
  vector<int64_t> dilation_hw;
  vector<int64_t> pad_hw;
  std::string backprop_out_fmt_str;
  AttrUtils::GetStr(conv2dbp_input_outbackprop_desc, kAttrOrgFmt, backprop_out_fmt_str);
  size_t pos_backprop_out_h = backprop_out_fmt_str.find('H');
  CHECK_POSITION(pos_backprop_out_h);
  size_t pos_backprop_out_w = backprop_out_fmt_str.find('W');
  CHECK_POSITION(pos_backprop_out_w);

  std::string y_fmt_str;
  AttrUtils::GetStr(conv2dbp_input_y_desc, kAttrOrgFmt, y_fmt_str);
  size_t pos_y_h = y_fmt_str.find('H');
  CHECK_POSITION(pos_y_h);
  size_t pos_y_w = y_fmt_str.find('W');
  CHECK_POSITION(pos_y_w);

  bool pre_dilation = false;
  bool need_dilation_flag = false;
  std::string filter_fmt_str;
  AttrUtils::GetStr(filter_desc, kAttrOrgFmt, filter_fmt_str);
  size_t pos_filter_h = filter_fmt_str.find('H');
  CHECK_POSITION(pos_filter_h);
  size_t pos_filter_w = filter_fmt_str.find('W');
  CHECK_POSITION(pos_filter_w);

  // if pre_dilation is true, Dilation -> conv2dbp_input
  // else conv2dbp_input -> Dilation
  if (filter_ori_shape[pos_filter_h] > 1 || filter_ori_shape[pos_filter_w] > 1) {
    pre_dilation = true;
  }
  for (unsigned int i = 0; i < strides.size(); ++i){
    if (strides[i] > 2 || (strides[i] > 1 and !pre_dilation)){
      need_dilation_flag = true;
      break;
    }
  }

  if (!need_dilation_flag){
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Because stride is 1, Dilation is not required");
    return NOT_CHANGED;
  }

  dilation_hw = post_get_hw_not_pad(out_backprop_ori_shape,strides, pos_backprop_out_h, pos_backprop_out_w);
  if (pre_dilation) {
    pad_hw = pre_get_pad_value(dilation_hw, y_shape, pos_y_h, pos_y_w,
                               filter_ori_shape, pos_filter_h, pos_filter_w, pads);
  } else {
    pad_hw = post_get_pad_value(dilation_hw, y_shape, pos_y_h, pos_y_w);
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "the pad_hw is %d and %d", pad_hw[0], pad_hw[1]);

  ge::NodePtr dilation_node = nullptr;
  ge::NodePtr pad_node = nullptr;
  auto basename = conv2dbp_input_node->GetName();
  if (pre_dilation){
    FUSION_PASS_CHECK(
      generate_pre_dilation_node(
        &graph, &out_backprop_out_desc, &conv2dbp_input_outbackprop_desc,
        &dilation_node, strides, basename) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to generate dilation node"),
      return FAILED);
  } else {
    FUSION_PASS_CHECK(
      generate_post_dilation_node(
        &graph, &conv2dbp_input_y_desc, &y_desc, &conv2dbp_input_outbackprop_desc,
        &dilation_node, strides, pad_hw, basename) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to generate dilation node"),
      return FAILED);

    ge::GeTensorDesc dilation_y_desc = dilation_node->GetOpDesc()->GetOutputDesc(0);
    if (pad_hw[0] != 0 || pad_hw[1] != 0) {
        FUSION_PASS_CHECK(generate_pad_node(&graph, &dilation_y_desc,
        &y_desc, &pad_node, pad_hw, basename) != SUCCESS, 
        ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
        "fail to generate pad node"),
        return FAILED);
    
    }
  }

  FUSION_PASS_CHECK(
    Relink(out_backprop_node, dilation_node, conv2dbp_input_node,
           y_node, pad_node, out_bp_anchor, y_anchor, pre_dilation, y_idxs) != SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to generate relink node"),
    return FAILED);

  fusion_nodes.push_back(dilation_node);
  op.SetAttr("strides", {1, 1, 1, 1});

  if (pre_dilation) {
    vector<int64_t> new_pads = pads;
    new_pads[1] = pads[1] - pad_hw[0];
    new_pads[3] = pads[3] - pad_hw[1];
    OP_LOGD(FUSED_OP_TYPE.c_str(), "the pad_new is %d and %d", new_pads[1], new_pads[3]);
    op.SetAttr("pads", new_pads);
  }

  if (!pre_dilation) {
    std::vector<int64_t> input_size_new = input_size;
    input_size_new[pos_backprop_out_h] = out_backprop_ori_shape[pos_backprop_out_h];
    input_size_new[pos_backprop_out_w] = out_backprop_ori_shape[pos_backprop_out_w];
    op.SetAttr("input_size", input_size_new);
    if (pad_node != nullptr) {
      fusion_nodes.push_back(pad_node) ;
    }
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "End Conv2DbpInputDilationFusionPass.");

  return SUCCESS;
}

REGISTER_PASS("Conv2DbpInputDilationFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, Conv2DbpInputDilationFusionPass);
} // namespace fe
