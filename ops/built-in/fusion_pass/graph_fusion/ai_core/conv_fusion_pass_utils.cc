/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file conv_fusion_pass_utils.cc
 * \brief
 */
#include "conv_fusion_pass_utils.h"

#include <string>
#include <vector>
#include <map>
#include <cmath>

#include "graph/debug/ge_attr_define.h"
#include "pattern_fusion_util.h"
#include "op_log.h"

using namespace std;
using namespace ge;

namespace fe {
const static string FUSED_OP_TYPE = "ConvFusionPass";
const static constexpr int CHANNEL_MIN = 16;
const static int32_t N_DIM = 0;
const static int32_t C_DIM = 1;
const static int32_t H_DIM = 2;
const static int32_t W_DIM = 3;
const static int64_t COUT = 16;
const static int64_t CIN = 16;
const size_t ORI_SHAPE_DIM = 4;
const std::map<ge::Format, std::vector<int32_t>> kFormatInNchwDimMap = {{FORMAT_NCHW, {0, 1, 2, 3}},
                                                                        {FORMAT_HWCN, {3, 2, 0, 1}}};

int64_t ConvFusionPassUtils::LCM(int64_t numL, int64_t numR) {
  if (numR == 0 || numL == 0) {
    return 1;
  }
  int64_t product = numL * numR;
  while (numL % numR != 0) {
    int64_t tmp = numL % numR;
    numL = numR;
    numR = tmp;
  }

  return product / numR;
}

bool ConvFusionPassUtils::ReplaceOutputAnchor(const ge::NodePtr& pre_node,
                                              uint32_t pre_idx,
                                              const ge::NodePtr& back_node,
                                              uint32_t back_idx) {
  FUSION_PASS_CHECK(pre_node == nullptr || pre_node->GetOutDataAnchor(pre_idx) == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "pre node is nullptr or output data anchor is nullptr."),
                    return false);
  FUSION_PASS_CHECK(back_node == nullptr || back_node->GetOutDataAnchor(back_idx) == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "back node is nullptr or output data anchor is nullptr."),
                    return false);
  ge::OutDataAnchorPtr pre_node_anchor = pre_node->GetOutDataAnchor(pre_idx);
  ge::OutDataAnchorPtr back_node_anchor = back_node->GetOutDataAnchor(back_idx);
  ge::NodePtr post_node = nullptr;
  for (auto post_anchor_ptr : pre_node_anchor->GetPeerInDataAnchors()) {
    post_node = post_anchor_ptr->GetOwnerNode();
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(pre_node_anchor, post_anchor_ptr) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "remove edge failed!"), return false);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(back_node_anchor, post_anchor_ptr) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge failed!"), return false);
  }
  return true;
}

ge::OpDescPtr ConvFusionPassUtils::CreateReshape(const string& node_name, const ge::GeTensorDesc& reshape_input_desc,
                                                 const ge::GeTensorDesc& reshape_output_desc) {
  vector<int64_t> reshape_output_shape_vec = reshape_output_desc.GetShape().GetDims();
  ge::OpDescPtr reshape_desc;
  FUSION_PASS_MAKE_SHARED(reshape_desc = std::make_shared<ge::OpDesc>(node_name + "/Reshape", "Reshape"),
                          return nullptr);
  FUSION_PASS_CHECK(reshape_desc->AddInputDesc("x", reshape_input_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "reshape failed to add input desc x to reshape."), return nullptr);
  FUSION_PASS_CHECK(reshape_desc->AddOutputDesc("y", reshape_output_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "reshape failed to add output desc y to reshape."), return nullptr);
  ge::AttrUtils::SetListInt(reshape_desc, "shape", reshape_output_shape_vec);
  return reshape_desc;
}

bool ConvFusionPassUtils::GetResizeDepthwiseFilter(const vector<int64_t>& ori_shape, const ge::Format& format,
                                                   int groups, vector<int64_t>& resize_shape,
                                                   vector<int64_t>& fractal_shape) {
  FUSION_PASS_CHECK(kFormatInNchwDimMap.find(format) == kFormatInNchwDimMap.end(),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "filter format only support HWCN and NCHW."), return false);
  FUSION_PASS_CHECK(ori_shape.size() != ORI_SHAPE_DIM, OP_LOGE(FUSED_OP_TYPE.c_str(),
                    "filter dim only support 4 dim."),
                    return false);
  vector<int32_t> dim_vec = kFormatInNchwDimMap.at(format);
  if (format == ge::FORMAT_NCHW) {
    resize_shape = {ori_shape[dim_vec[N_DIM]] * ori_shape[dim_vec[C_DIM]], 1, ori_shape[dim_vec[H_DIM]],
                    ori_shape[dim_vec[W_DIM]]};
  } else {
    resize_shape = {ori_shape[dim_vec[H_DIM]], ori_shape[dim_vec[W_DIM]], 1,
                    ori_shape[dim_vec[N_DIM]] * ori_shape[dim_vec[C_DIM]]};
  }

  GroupDict group_dict;
  ConvFusionPassUtils::CalculateGroup(groups, ori_shape[dim_vec[N_DIM]] * ori_shape[dim_vec[C_DIM]], groups,
                                      group_dict);
  fractal_shape = {
      group_dict.real_g * group_dict.cin1_g * ori_shape[dim_vec[H_DIM]] * ori_shape[dim_vec[W_DIM]],
      (group_dict.cout_g + COUT - 1) / COUT, COUT, CIN};
  return true;
}

bool ConvFusionPassUtils::CalculateGroup(int64_t in_channel, int64_t out_channel, int64_t groups,
                                         GroupDict& group_dict) {
  if (groups == 0 || in_channel % groups != 0 || out_channel % groups != 0) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "The number of input is invalid.");
    return false;
  }

  int64_t mag_factor0 = LCM(in_channel / groups, CHANNEL_MIN) / (in_channel / groups);
  int64_t mag_factor1 = LCM(out_channel / groups, CHANNEL_MIN) / (out_channel / groups);
  int64_t mag_factor = min(LCM(mag_factor0, mag_factor1), groups);
  if (mag_factor == 0) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "mag_factor is 0.");
    return false;
  }
  int64_t cin1_g = (mag_factor * in_channel / groups + CHANNEL_MIN - 1) / CHANNEL_MIN;
  int64_t cout_g = (mag_factor * out_channel / groups + CHANNEL_MIN - 1) / CHANNEL_MIN * CHANNEL_MIN;
  int64_t real_g = (groups + mag_factor - 1) / mag_factor;

  group_dict.real_g = real_g;
  group_dict.mag_factor = mag_factor;
  group_dict.cin1_g = cin1_g;
  group_dict.cout_g = cout_g;
  group_dict.cin_ori = in_channel;
  group_dict.cout_ori = out_channel;
  group_dict.groups = groups;

  return true;
}
}  // namespace fe
