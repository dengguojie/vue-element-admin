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
 * \file conv2dbackprop_dilation_fusion_pass.cc
 * \brief conv2dbackprop_input dilation fusion pass
 * dilation->conv2dbackinput or conv2dbackinput->dilation
 */

#include "conv2dbackprop_dilation_fusion_pass.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "anchor_util.h"
#include "common/util/platform_info.h"
#include "error_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace fe {
const string Conv2DbpInputDilationFusionPass::FUSED_OP_TYPE = "Conv2DBackpropInputD";
const vector<std::string> Conv2DbpInputDilationFusionPass::kOriFormatSupportByFilter{"HWCN", "NHWC", "NCHW"};
const vector<std::string> Conv2DbpInputDilationFusionPass::kOriFormatSupportByOutBackprop{"NHWC", "NCHW"};
const vector<std::string> Conv2DbpInputDilationFusionPass::kOriFormatSupportByY{
    Conv2DbpInputDilationFusionPass::kOriFormatSupportByOutBackprop.begin(),
    Conv2DbpInputDilationFusionPass::kOriFormatSupportByOutBackprop.end()};

static const string PATTERN_CONV2DBACKPROPINPUT = "Conv2DBackpropInputD";
static const string kAttrOrgFmt = "origin_format";
static const int kNum2 = 2;
static const int kIndex2 = 2;
static const int kIndex3 = 3;
static const int kBiasAnchoridx = 3;
static const int kOriShapeLen = 4;
static const string kIntrinsicUbToL1 = "Intrinsic_data_move_ub2l1";
static const string kConv2dBackpropInputOpType = "Conv2DBackpropInputD";
static const string kDeconvolutionOpType = "Deconvolution";
static const string kConv2dTransposeOpType = "Conv2DTransposeD";

#define CHECK_POSITION(position)                                                                      \
  {                                                                                                   \
    if ((position) == std::string::npos) {                                                            \
      OP_LOGI(PATTERN_CONV2DBACKPROPINPUT.c_str(), "get position failed:%s:%d", #position, position); \
      return FAILED;                                                                                  \
    }                                                                                                 \
  }

vector<FusionPattern *> Conv2DbpInputDilationFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter Conv2DbpInputDilationFusionPass::DefinePatterns.");
  vector<FusionPattern *> patterns;
  FusionPattern *pattern = new (std::nothrow) FusionPattern("Conv2DbpInputDilationFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern
      ->AddOpDesc(PATTERN_CONV2DBACKPROPINPUT,
                  {kConv2dBackpropInputOpType, kDeconvolutionOpType, kConv2dTransposeOpType})
      .SetOutput(PATTERN_CONV2DBACKPROPINPUT);

  patterns.push_back(pattern);

  return patterns;
}


vector<int64_t> get_hw_after_dilation(const vector<int64_t> input_shape, const vector<int64_t> dilations,
                                      string::size_type pos_input_h, string::size_type pos_input_w) {
  int64_t h_dilation = (input_shape[pos_input_h] - 1) * dilations[pos_input_h] + 1;
  int64_t w_dilation = (input_shape[pos_input_w] - 1) * dilations[pos_input_w] + 1;
  vector<int64_t> dilation_hw;
  dilation_hw.push_back(h_dilation);
  dilation_hw.push_back(w_dilation);
  return dilation_hw;
}


Status Conv2DbpInputDilationFusionPass::generate_pre_dilation_node(
    ge::ComputeGraph *graph, ge::GeTensorDesc *prev_out_desc, ge::GeTensorDesc *next_in_desc,
    ge::NodePtr *dilation_node, const vector<int64_t> strides, const std::string &basename) {
  ge::OpDescPtr dilation_desc;
  FUSION_PASS_MAKE_SHARED((dilation_desc = std::make_shared<ge::OpDesc>(basename + "_dilation", "Dilation")),
                          return FAILED);

  std::string fmt_str = TypeUtils::FormatToSerialString(next_in_desc->GetOriginFormat());
  size_t pos_h = fmt_str.find('H');
  CHECK_POSITION(pos_h);
  size_t pos_w = fmt_str.find('W');
  CHECK_POSITION(pos_w);

  // nc1hwc0
  FUSION_PASS_CHECK(dilation_desc->AddInputDesc("x", *prev_out_desc) != GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input desc to dilation failed"), return FAILED);

  auto next_in_shape = prev_out_desc->GetShape().GetDims();
  FUSION_PASS_CHECK(next_in_shape.size() < kOriShapeLen,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "size of input shape of next node must >=4"),
                    return FAILED);
  next_in_shape[kIndex2] = (next_in_shape[kIndex2] - 1) * strides[pos_h] + 1;
  next_in_shape[kIndex3] = (next_in_shape[kIndex3] - 1) * strides[pos_w] + 1;
  next_in_desc->SetShape(ge::GeShape(next_in_shape));

  auto next_in_ori_shape = prev_out_desc->GetOriginShape().GetDims();
  FUSION_PASS_CHECK(next_in_ori_shape.size() < kOriShapeLen,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "size of origin input shape of next node must >=4"),
                    return FAILED);
  next_in_ori_shape[pos_h] = (next_in_ori_shape[pos_h] - 1) * strides[pos_h] + 1;
  next_in_ori_shape[pos_w] = (next_in_ori_shape[pos_w] - 1) * strides[pos_w] + 1;
  next_in_desc->SetOriginShape(ge::GeShape(next_in_ori_shape));

  FUSION_PASS_CHECK(dilation_desc->AddOutputDesc("y", *next_in_desc) != GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output desc y to dilation failed"),
                    return FAILED);
  vector<int64_t> dilations_new = {1, 1, 1, 1, 1};
  dilations_new[kIndex2] = strides[pos_h];
  dilations_new[kIndex3] = strides[pos_w];

  ge::AttrUtils::SetListInt(dilation_desc, "dilations", dilations_new);
  ge::AttrUtils::SetFloat(dilation_desc, "padding_value", 0.0);

  auto new_dilation_node = graph->AddNode(dilation_desc);
  FUSION_PASS_CHECK(new_dilation_node == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to add dilation node to graph"),
                    return FAILED);

  *dilation_node = new_dilation_node;
  return SUCCESS;
}


Status Conv2DbpInputDilationFusionPass::generate_post_dilation_node(
    ge::ComputeGraph *graph, ge::GeTensorDesc *prev_out_desc, ge::GeTensorDesc *next_in_desc,
    ge::GeTensorDesc *conv2dbp_input_outbackprop_desc, ge::NodePtr *dilation_node, const vector<int64_t> strides,
    const vector<int64_t> pads, const std::string &basename) {
  ge::OpDescPtr dilation_desc;
  FUSION_PASS_MAKE_SHARED((dilation_desc = std::make_shared<ge::OpDesc>(basename + "_dilation", "Dilation")),
                          return FAILED);

  std::string y_fmt_str = TypeUtils::FormatToSerialString(prev_out_desc->GetOriginFormat());
  size_t pos_y_h = y_fmt_str.find('H');
  CHECK_POSITION(pos_y_h);
  size_t pos_y_w = y_fmt_str.find('W');
  CHECK_POSITION(pos_y_w);

  std::string outbackprop_fmt_str = TypeUtils::FormatToSerialString(
      conv2dbp_input_outbackprop_desc->GetOriginFormat());
  size_t pos_outbackprop_out_h = outbackprop_fmt_str.find('H');
  CHECK_POSITION(pos_outbackprop_out_h);
  size_t pos_outbackprop_out_w = outbackprop_fmt_str.find('W');
  CHECK_POSITION(pos_outbackprop_out_w);

  auto out_backprop_ori_shape = conv2dbp_input_outbackprop_desc->GetOriginShape().GetDims();
  FUSION_PASS_CHECK(
      out_backprop_ori_shape.size() < kOriShapeLen,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "size of origin output shape of backprop node must >=4"),
      return FAILED);

  FUSION_PASS_CHECK(dilation_desc->AddOutputDesc("y", *next_in_desc) != GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output desc y to dilation failed"),
                    return FAILED);
  auto prev_out_shape = next_in_desc->GetShape().GetDims();
  FUSION_PASS_CHECK(prev_out_shape.size() < kOriShapeLen,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "size of output shape of previous node must >=4"),
                    return FAILED);
  prev_out_shape[kIndex2] = out_backprop_ori_shape[pos_outbackprop_out_h];
  prev_out_shape[kIndex3] = out_backprop_ori_shape[pos_outbackprop_out_w];
  prev_out_desc->SetShape(ge::GeShape(prev_out_shape));

  auto prev_out_ori_shape = next_in_desc->GetOriginShape().GetDims();
  FUSION_PASS_CHECK(
      prev_out_ori_shape.size() < kOriShapeLen,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "size of origin output shape of previous node must >=4"),
      return FAILED);
  prev_out_ori_shape[pos_y_h] = out_backprop_ori_shape[pos_outbackprop_out_h];
  prev_out_ori_shape[pos_y_w] = out_backprop_ori_shape[pos_outbackprop_out_w];
  prev_out_desc->SetOriginShape(ge::GeShape(prev_out_ori_shape));
  FUSION_PASS_CHECK(dilation_desc->AddInputDesc("x", *prev_out_desc) != GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input desc to dilation failed"), return FAILED);

  vector<int64_t> dilations_new = {1, 1, 1, 1, 1};
  dilations_new[kIndex2] = strides[pos_outbackprop_out_h];
  dilations_new[kIndex3] = strides[pos_outbackprop_out_w];
  ge::AttrUtils::SetListInt(dilation_desc, "dilations", dilations_new);
  ge::AttrUtils::SetFloat(dilation_desc, "padding_value", 0.0);
  const vector<int64_t> pads_in = {0, pads[0], 0, pads[1]};
  ge::AttrUtils::SetListInt(dilation_desc, "pads", pads_in);

  auto new_dilation_node = graph->AddNode(dilation_desc);
  FUSION_PASS_CHECK(new_dilation_node == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to add dilation node to graph"),
                    return FAILED);

  *dilation_node = new_dilation_node;
  return SUCCESS;
}


Status Conv2DbpInputDilationFusionPass::Relink(ge::NodePtr out_backprop_node, ge::NodePtr dilation_node,
                                               ge::NodePtr conv2dbp_input_node, ge::NodePtr y_node,
                                               const int pre_anchor, const int sub_anchor,
                                               bool pre_dilation, const vector<int> &y_idxs) {
  if (pre_dilation) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "pre relink");
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(out_backprop_node->GetOutDataAnchor(0),
                                                 conv2dbp_input_node->GetInDataAnchor(pre_anchor)) != SUCCESS,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                            "fail to remove edge between out_backprop_node and conv2dbp_input_node"),
                      return FAILED);

    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(out_backprop_node->GetOutDataAnchor(0), dilation_node->GetInDataAnchor(0)) != SUCCESS,
        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fail to add edge between out_backprop_node and dilation_node"),
        return FAILED);

    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(dilation_node->GetOutDataAnchor(0),
                                              conv2dbp_input_node->GetInDataAnchor(pre_anchor)) != SUCCESS,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                            "fail to add edge between dilation_node and conv2dbp_input_node"),
                      return FAILED);

    auto dilation_out_desc = GetCurrNodeOutputDesc(dilation_node, 0);
    FUSION_PASS_CHECK(dilation_out_desc == nullptr,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dilation_out_desc is null"), return FAILED);
    auto conv2dbp_input_op_desc = conv2dbp_input_node->GetOpDesc();
    FUSION_PASS_CHECK(conv2dbp_input_op_desc == nullptr,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dilation_out_desc is null"), return FAILED);
    FUSION_PASS_CHECK(
        conv2dbp_input_op_desc->UpdateInputDesc(pre_anchor, *(dilation_out_desc.get())) != SUCCESS,
        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fail to update input description of conv2dbp_input_node"),
        return FAILED);
  } else {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "post relink");
    for (int i : y_idxs) {
      FUSION_PASS_CHECK(
          ge::GraphUtils::RemoveEdge(conv2dbp_input_node->GetOutDataAnchor(sub_anchor), y_node->GetInDataAnchor(i)) !=
              SUCCESS,
          CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fail to remove edge between conv2dbp_input_node and y_node"),
          return FAILED);
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(dilation_node->GetOutDataAnchor(0), y_node->GetInDataAnchor(i)) != SUCCESS,
          CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fail to add edge between dilation_node and y_node"),
          return FAILED);
    }
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(conv2dbp_input_node->GetOutDataAnchor(sub_anchor),
                                              dilation_node->GetInDataAnchor(0)) != SUCCESS,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                            "fail to add edge between conv2dbp_input_node and dilation_node"),
                      return FAILED);
    auto dilation_in_desc = GetCurrNodeInputDesc(dilation_node, 0);
    FUSION_PASS_CHECK(dilation_in_desc == nullptr,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dilation_in_desc is null"), return FAILED);
    auto conv2dbp_input_op_desc = conv2dbp_input_node->GetOpDesc();
    FUSION_PASS_CHECK(
        conv2dbp_input_op_desc->UpdateOutputDesc(sub_anchor, *(dilation_in_desc.get())) != SUCCESS,
        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fail to update output description of conv2dbp_input_node"),
        return FAILED);
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "finish  relink");
  return SUCCESS;
}

Status Conv2DbpInputDilationFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                                               vector<ge::NodePtr> &fusion_nodes) {
  PlatformInfo platform_info;
  OptionalInfo opti_compilation_info;
  FUSION_PASS_CHECK(
      PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, opti_compilation_info) != SUCCESS,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get platform_info failed."), return FAILED);
  bool cube_vector_split_flag = platform_info.ai_core_spec.cube_vector_split;
  const auto &intrinsic_map = platform_info.ai_core_intrinsic_dtype_map;
  bool support_ub_dilation_flag = (intrinsic_map.find(kIntrinsicUbToL1) != intrinsic_map.end());
  FUSION_PASS_CHECK(!cube_vector_split_flag || support_ub_dilation_flag,
                    OP_LOGI(FUSED_OP_TYPE.c_str(),
                      "This platform support dilation in ub, dilation op is not required."),
                    return NOT_CHANGED);

  ge::NodePtr conv2dbp_input_node = GetNodeFromMapping(PATTERN_CONV2DBACKPROPINPUT, mapping);
  FUSION_PASS_CHECK(conv2dbp_input_node == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Conv2DBackpropInputNode is null, fusion failed."),
                    return PARAM_INVALID);

  auto dx_node_name = conv2dbp_input_node->GetName();
  OP_LOGD(FUSED_OP_TYPE.c_str(), "The conv2dbp input node name is:[%s]", dx_node_name.c_str());

  int filter_anchor = 0;
  int out_bp_anchor = 1;
  int y_anchor = 0;
  auto conv2dbp_input_op_desc = conv2dbp_input_node->GetOpDesc();
  string op_type = conv2dbp_input_op_desc->GetType();
  bool is_support_bias = false;
  if (op_type == kDeconvolutionOpType || op_type == kConv2dTransposeOpType) {
    out_bp_anchor = 0;
    filter_anchor = 1;
    is_support_bias = true;
  }

  Operator op = ge::OpDescUtils::CreateOperatorFromNode(conv2dbp_input_node);

  // get attr
  std::vector<int64_t> input_size;
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;

  FUSION_PASS_CHECK(op.GetAttr("strides", strides) != GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                          "op [%s] get attribute strides failed or strides not exist",
                                          dx_node_name.c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(op.GetAttr("pads", pads) != GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                          "op [%s] get attribute pads failed or pads not exist",
                                          dx_node_name.c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(
      op.GetAttr("input_size", input_size) != GRAPH_SUCCESS && op_type != kDeconvolutionOpType,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                            "op [%s] get attribute input_size failed or input_size not exist",
                            dx_node_name.c_str()),
      return FAILED);

  auto peer_out_anchor = GetPeerOutAnchorWithInDataAnchor(conv2dbp_input_node, out_bp_anchor);
  FUSION_PASS_CHECK(peer_out_anchor == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "op [%s] get peer out anchor failed",
                                          dx_node_name.c_str()),
                    return FAILED);
  auto out_backprop_node = peer_out_anchor->GetOwnerNode();
  FUSION_PASS_CHECK(out_backprop_node == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "op [%s] get owner node failed",
                                          dx_node_name.c_str()),
                    return FAILED);

  ge::GeTensorDesc filter_desc = *(GetCurrNodeInputDesc(conv2dbp_input_node, filter_anchor).get());
  ge::GeTensorDesc conv2dbp_input_outbackprop_desc = *(GetCurrNodeInputDesc(conv2dbp_input_node, out_bp_anchor).get());
  ge::GeTensorDesc conv2dbp_input_y_desc = *(GetCurrNodeOutputDesc(conv2dbp_input_node, y_anchor).get());
  FUSION_PASS_CHECK(GetCurrNodeInputDesc(conv2dbp_input_node, filter_anchor) == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "filter_desc is null"), return FAILED);
  FUSION_PASS_CHECK(GetCurrNodeInputDesc(conv2dbp_input_node, out_bp_anchor) == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "outbackprop_desc is null"), return FAILED);
  FUSION_PASS_CHECK(GetCurrNodeInputDesc(conv2dbp_input_node, y_anchor) == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "y_desc is null"), return FAILED);
  // get shape
  auto filter_ori_shape = filter_desc.GetOriginShape().GetDims();
  FUSION_PASS_CHECK(filter_ori_shape.size() != kOriShapeLen,
                    OP_LOGE(FUSED_OP_TYPE, "size of ori_shape of filter = 4, actually %ld", filter_ori_shape.size()),
                    return FAILED);
  auto out_backprop_ori_shape = conv2dbp_input_outbackprop_desc.GetOriginShape().GetDims();
  FUSION_PASS_CHECK(
      out_backprop_ori_shape.size() != kOriShapeLen,
      OP_LOGE(FUSED_OP_TYPE, "size of ori_shape of out_backprop = 4, actually %ld",
                            out_backprop_ori_shape.size()),
      return FAILED);
  auto y_shape = conv2dbp_input_y_desc.GetOriginShape().GetDims();
  FUSION_PASS_CHECK(y_shape.size() != kOriShapeLen,
                    OP_LOGE(FUSED_OP_TYPE, "size of ori_shape of y = 4, actually %ld", y_shape.size()),
                    return FAILED);

  vector<int64_t> dilation_hw;
  vector<int64_t> pad_hw(kNum2);
  std::string backprop_out_fmt_str = TypeUtils::FormatToSerialString(conv2dbp_input_outbackprop_desc.GetOriginFormat());
  FUSION_PASS_CHECK(find(kOriFormatSupportByOutBackprop.begin(), kOriFormatSupportByOutBackprop.end(),
                         backprop_out_fmt_str) == kOriFormatSupportByOutBackprop.end(),
                    OP_LOGE(FUSED_OP_TYPE, "%s only support NHWC, NCHW, actually %s", kAttrOrgFmt.c_str(),
                            backprop_out_fmt_str.c_str()),
                    return FAILED);
  size_t pos_backprop_out_h = backprop_out_fmt_str.find('H');
  CHECK_POSITION(pos_backprop_out_h);
  size_t pos_backprop_out_w = backprop_out_fmt_str.find('W');
  CHECK_POSITION(pos_backprop_out_w);

  std::string y_fmt_str = TypeUtils::FormatToSerialString(conv2dbp_input_y_desc.GetOriginFormat());
  FUSION_PASS_CHECK(
      find(kOriFormatSupportByY.begin(), kOriFormatSupportByY.end(), y_fmt_str) == kOriFormatSupportByY.end(),
      OP_LOGE(FUSED_OP_TYPE, "%s only support NHWC, NCHW, actually %s", kAttrOrgFmt.c_str(), y_fmt_str.c_str()), 
      return FAILED);
  size_t pos_y_h = y_fmt_str.find('H');
  CHECK_POSITION(pos_y_h);
  size_t pos_y_w = y_fmt_str.find('W');
  CHECK_POSITION(pos_y_w);

  bool pre_dilation = false;
  bool need_dilation_flag = false;
  std::string filter_fmt_str = TypeUtils::FormatToSerialString(filter_desc.GetOriginFormat());
  FUSION_PASS_CHECK(find(kOriFormatSupportByFilter.begin(), kOriFormatSupportByFilter.end(), filter_fmt_str) ==
                        kOriFormatSupportByFilter.end(),
                    OP_LOGE(FUSED_OP_TYPE, "%s only support HWCN, NHWC, NCHW, actually %s", kAttrOrgFmt.c_str(),
                            filter_fmt_str.c_str()),
                    return FAILED);
  size_t pos_filter_h = filter_fmt_str.find('H');
  CHECK_POSITION(pos_filter_h);
  size_t pos_filter_w = filter_fmt_str.find('W');
  CHECK_POSITION(pos_filter_w);

  // if pre_dilation is true, Dilation -> conv2dbp_input
  // else conv2dbp_input -> Dilation
  if (filter_ori_shape[pos_filter_h] > 1 || filter_ori_shape[pos_filter_w] > 1) {
    pre_dilation = true;
  } else if (is_support_bias) {
    // if add bias table before back dilation will cause precision error.
    // when has bias input, dx should use pre dilation
    pre_dilation = (conv2dbp_input_op_desc->GetInputDescPtr(kBiasAnchoridx) != nullptr);
  }

  for (size_t i = 0; i < strides.size(); ++i){
    if (strides[i] > kNum2 || (strides[i] > 1 and !pre_dilation)) {
      need_dilation_flag = true;
      break;
    }
  }

  if (!need_dilation_flag) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Because op [%s]'s stride is 1, Dilation is not required",
            dx_node_name.c_str());
    return NOT_CHANGED;
  }

  dilation_hw = get_hw_after_dilation(out_backprop_ori_shape, strides, pos_backprop_out_h, pos_backprop_out_w);
  pad_hw[0] = y_shape[pos_y_h] - filter_ori_shape[pos_filter_h] + pads[0] + pads[1] + 1 - dilation_hw[0];
  pad_hw[1] = y_shape[pos_y_w] - filter_ori_shape[pos_filter_w] + pads[kIndex2] + pads[kIndex3] + 1 - dilation_hw[1];

  OP_LOGD(FUSED_OP_TYPE.c_str(), "the pad_hw is %d and %d", pad_hw[0], pad_hw[1]);

  int out_backprop_idx = peer_out_anchor->GetIdx();
  auto out_desc_ptr = GetCurrNodeOutputDesc(out_backprop_node, out_backprop_idx);
  FUSION_PASS_CHECK(out_desc_ptr == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "out_desc_ptr is null"),
                    return FAILED);
  ge::GeTensorDesc out_backprop_out_desc = *(out_desc_ptr.get());
  vector<int> y_idxs;

  auto outdata_anchor = conv2dbp_input_node->GetOutDataAnchor(y_anchor);
  FUSION_PASS_CHECK(
      outdata_anchor == nullptr,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "op [%s] failed to get out data anchor of index 0",
                            dx_node_name.c_str()),
      return FAILED);
  for (auto in_data_anchor : outdata_anchor->GetPeerInDataAnchors()) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "all idx is:");
    OP_LOGD(FUSED_OP_TYPE.c_str(), "idx = %d", in_data_anchor->GetIdx());
    y_idxs.push_back(in_data_anchor->GetIdx());
  }

  auto y_anchor_ptr = GetPeerInAnchorByOutDataAnchor(outdata_anchor, 0);
  FUSION_PASS_CHECK(y_anchor_ptr == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "fail to get peer in anchor by y."),
                    return NOT_CHANGED);
  int y_idx = y_anchor_ptr->GetIdx();

  ge::NodePtr y_node = GetPeerInNodeByOutDataAnchor(outdata_anchor, 0);
  FUSION_PASS_CHECK(y_node == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "fail to get node of y"),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(GetCurrNodeInputDesc(y_node, y_idx) == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "y_desc is null"), return FAILED);
  ge::GeTensorDesc y_desc = *(GetCurrNodeInputDesc(y_node, y_idx).get());

  ge::NodePtr dilation_node = nullptr;

  if (pre_dilation) {
    FUSION_PASS_CHECK(generate_pre_dilation_node(&graph, &out_backprop_out_desc, &conv2dbp_input_outbackprop_desc,
                                                 &dilation_node, strides, dx_node_name) != SUCCESS,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fail to generate dilation node"), return FAILED);
  } else {
    FUSION_PASS_CHECK(
        generate_post_dilation_node(&graph, &conv2dbp_input_y_desc, &y_desc, &conv2dbp_input_outbackprop_desc,
                                    &dilation_node, strides, pad_hw, dx_node_name) != SUCCESS,
        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fail to generate dilation node"), return FAILED);
  }

  FUSION_PASS_CHECK(Relink(out_backprop_node, dilation_node, conv2dbp_input_node, y_node, out_bp_anchor,
                           y_anchor, pre_dilation, y_idxs) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fail to generate relink node"), return FAILED);

  fusion_nodes.push_back(dilation_node);
  op.SetAttr("strides", {1, 1, 1, 1});

  if (pre_dilation) {
    vector<int64_t> new_pads = pads;
    new_pads[1] = pads[1] - pad_hw[0];
    new_pads[kIndex3] = pads[kIndex3] - pad_hw[1];
    OP_LOGD(FUSED_OP_TYPE.c_str(), "the pad_new is %d and %d", new_pads[1], new_pads[kIndex3]);
    op.SetAttr("pads", new_pads);
  } else if (op_type != kDeconvolutionOpType) {
    std::vector<int64_t> input_size_new = input_size;
    input_size_new[pos_backprop_out_h] = out_backprop_ori_shape[pos_backprop_out_h];
    input_size_new[pos_backprop_out_w] = out_backprop_ori_shape[pos_backprop_out_w];
    op.SetAttr("input_size", input_size_new);
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "End Conv2DbpInputDilationFusionPass.");

  return SUCCESS;
}

REGISTER_PASS("Conv2DbpInputDilationFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, Conv2DbpInputDilationFusionPass);
}  // namespace fe
