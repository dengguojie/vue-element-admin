/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file deconv_group_fusion_pass.cpp
 * \brief split fusion pass(conv2d_backprop_input --> conv2d_backprop_input_d)
 */
#include "deconv_group_fusion_pass.h"

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

using namespace ge;
namespace fe {
namespace {
const string DECONV = "Deconvolution";
const string PATTERN_DECONV = "Deconvolution";
}  // namespace

vector<FusionPattern*> DeconvGroupFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter DeconvGroupPass::DefinePatterns.");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("Deconvolution");
  FUSION_PASS_CHECK(
      pattern == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
      return patterns);

  pattern->AddOpDesc(PATTERN_DECONV, {DECONV}).SetOutput(PATTERN_DECONV);

  patterns.push_back(pattern);

  return patterns;
}

Status DeconvGroupFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                     vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter DeconvGroupPass::Fusion.");
  NodePtr deconv_node = GetNodeFromMapping(PATTERN_DECONV, mapping);
  OpDescPtr deconv_desc = deconv_node->GetOpDesc();

  // 1.if the deconv node doesn't have the attribute groups or the value is 1,
  // just return not changed.
  int64_t groups = 1;
  bool has_group_flag = ge::AttrUtils::GetInt(deconv_desc, "groups", groups);
  if (!has_group_flag || groups == 1) {
    OP_LOGD(FUSED_OP_TYPE.c_str(),
            "The deconv node[name=%s, type=%s] doesn't have the attribute "
            "groups, or the value is 1.",
            deconv_desc->GetName().c_str(), deconv_desc->GetType().c_str());
    return NOT_CHANGED;
  }

  GeTensorDesc input_desc = deconv_desc->GetInputDesc(0);
  size_t in_channel_index = -1;
  FUSION_PASS_CHECK(
      SUCCESS != PatternFusionUtil::ParseChannelIdx(input_desc, in_channel_index),
      OP_LOGE(
          FUSED_OP_TYPE.c_str(),
          "The original format of the deconv node[name=%s, type=%s]'s input0 "
          "is %s, which is unsupportable.",
          deconv_desc->GetName().c_str(), deconv_desc->GetType().c_str(),
          ge::TypeUtils::FormatToSerialString(input_desc.GetFormat()).c_str()),
      return FAILED);
  int64_t input_channel = input_desc.GetOriginShape().GetDim(in_channel_index);

  GeTensorDesc output_desc = deconv_desc->GetOutputDesc(0);
  size_t out_channel_index = -1;
  FUSION_PASS_CHECK(
      SUCCESS != PatternFusionUtil::ParseChannelIdx(output_desc, out_channel_index),
      OP_LOGE(
          FUSED_OP_TYPE.c_str(),
          "The original format of the deconv node[name=%s, type=%s]'s output0 "
          "is %s, which is unsupportable.",
          deconv_desc->GetName().c_str(), deconv_desc->GetType().c_str(),
          ge::TypeUtils::FormatToSerialString(output_desc.GetFormat()).c_str()),
      return FAILED);
  int64_t output_channel = output_desc.GetOriginShape().GetDim(out_channel_index);

  // 2. if the number of input channel and output channel are both divisible by
  // groups, then process group padding, otherwise, return failed.
  if (groups != 0 && input_channel % groups == 0 && output_channel % groups == 0) {
    return PatternFusionUtil::ProcessGroupPadding(graph, deconv_node, groups);
  } else {
    OP_LOGE(
        FUSED_OP_TYPE.c_str(),
        "The number of input channel(%lld) or output channel(%lld) of "
        "the deconv node[name=%s, type=%s] is not divisible by groups(%lld)",
        input_channel, output_channel, deconv_desc->GetName().c_str(),
        deconv_desc->GetType().c_str(), groups);
    return FAILED;
  }
}

REGISTER_PASS("ADeconvGroupFusionPass", BUILT_IN_GRAPH_PASS,
              DeconvGroupFusionPass);
}  // namespace fe
