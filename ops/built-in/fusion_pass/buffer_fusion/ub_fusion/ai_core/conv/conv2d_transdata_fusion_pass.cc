/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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

#include "conv2d_transdata_fusion_pass.h"
#include "anchor_util.h"
#include "common/util/platform_info.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {
static const char kPatternTransData1[] = "transdata1";
static const char kPatternCube[] = "Convolution";
static const char kPatternTransData2[] = "transdata2";

static const char kOpTransData[] = "TransData";
static const char kOpConv2D[] = "Convolution";

static pair<int64_t, int64_t> kNoRange = {1, -1};
/*
 * @brief: define transdata + conv2d + transdata ub fusion pattern
 *
 *   input             
 *      \         
 *  transdata_1   weight
 *         \      /
 *          conv2d
 *            |
 *         transdata_2
 *
 * @return BufferFusionPattern: return all valid patterns
 */
vector<BufferFusionPattern *> Conv2dTransDataFusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;
  string pass_name = "Conv2dTransDataFusionPass";
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(pass_name);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(kFusedOpType.c_str(), "New an object failed."), return patterns);

  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pattern.", pass_name.c_str());
  pattern->AddOpDesc(kPatternTransData1, {kOpTransData}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(kPatternCube, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(kPatternTransData2, {kOpTransData}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .SetHead({kPatternTransData1})
          .SetOutputs(kPatternTransData1, {kPatternCube})
          .SetOutputs(kPatternCube, {kPatternTransData2});
  patterns.push_back(pattern);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pattern.", pass_name.c_str());

  return patterns;
}

bool Conv2dTransDataFusionPass::CheckTransDataFormat(const ge::NodePtr &node, const bool &is_input) const {
  bool is_support_format = false;
  if (is_input) {
    is_support_format = (node->GetOpDesc()->GetInputDesc(0).GetFormat() == ge::FORMAT_NCHW) &&
                        (node->GetOpDesc()->GetOutputDesc(0).GetFormat() == ge::FORMAT_NC1HWC0);
    FUSION_PASS_CHECK(!is_support_format,
                      OP_LOGE(kFusedOpType.c_str(), "Only support format NCHW of input."),
                      return is_support_format);
  } else {
    is_support_format = (node->GetOpDesc()->GetInputDesc(0).GetFormat() == ge::FORMAT_NC1HWC0) &&
                        (node->GetOpDesc()->GetOutputDesc(0).GetFormat() == ge::FORMAT_NCHW);
    FUSION_PASS_CHECK(!is_support_format,
                      OP_LOGE(kFusedOpType.c_str(), "Only support format NC1HWC0 of output."),
                      return is_support_format);
  }
  return is_support_format;
}

bool Conv2dTransDataFusionPass::CheckInputNoRange(const ge::NodePtr &cube_node) const {
  // 2 means out_backprop
  auto input_desc = GetCurrNodeMutableInputDesc(cube_node, 2);
  vector<int64_t> input_dims = input_desc->GetOriginShape().GetDims();
  for (auto input_dim : input_dims) {
    FUSION_PASS_CHECK(input_dim != -1,
                      OP_LOGE(kFusedOpType.c_str(), "Only support dynamic nchw."), return false);
  }

  vector<pair<int64_t, int64_t> > range_data;
  FUSION_PASS_CHECK(input_desc->GetShapeRange(range_data) == ge::GRAPH_FAILED,
                    OP_LOGE(kFusedOpType.c_str(), "Failed to get input shape range of cube_node:2."),
                    return false);
  FUSION_PASS_CHECK(range_data.empty(),
                    OP_LOGE(kFusedOpType.c_str(), "range_data is empty."),
                    return false);

  // range: ((1, -1), (1, -1), (1, -1), (1, -1), (16, 16))
  for (size_t i = 0; i < range_data.size() - 1; i++) {
    FUSION_PASS_CHECK(range_data[i] != kNoRange,
                      OP_LOGE(kFusedOpType.c_str(), "Only support nchw no range."), return false);
  }
  return true;
}

bool Conv2dTransDataFusionPass::CheckOpCube(const ge::NodePtr &cube_node) const {
  FUSION_PASS_CHECK(cube_node->GetType() != kOpConv2D,
                    OP_LOGD(kFusedOpType.c_str(),
                            "The op_type of node [%s] should be Conv2D, but actually is [%s].",
                            cube_node->GetName().c_str(), cube_node->GetType().c_str()),
                    return false);

  FUSION_PASS_CHECK(!CheckInputNoRange(cube_node),
                    OP_LOGD(kFusedOpType.c_str(), "Only support NCHW no range."),
                    return false);
  return true;
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status Conv2dTransDataFusionPass::GetFusionNodes(const BufferFusionMapping &mapping,
                                                 vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(kFusedOpType.c_str(), "Begin to do Conv2dTransDataFusionPass.");

  vector<ge::NodePtr> transdata1_nodes = GetMatchedNodesByDescName(kPatternTransData1, mapping);
  vector<ge::NodePtr> cube_nodes = GetMatchedNodesByDescName(kPatternCube, mapping);
  vector<ge::NodePtr> transdata2_nodes = GetMatchedNodesByDescName(kPatternTransData2, mapping);
  FUSION_PASS_CHECK(cube_nodes.empty(),
                    OP_LOGD(kFusedOpType.c_str(), "Conv2d node is not matched."),
                    return SUCCESS);
  FUSION_PASS_CHECK(transdata2_nodes.empty(),
                    OP_LOGD(kFusedOpType.c_str(), "TransData node of output is not matched."),
                    return SUCCESS);

  ge::NodePtr cube_node = cube_nodes[0];
  FUSION_PASS_CHECK(!CheckOpCube(cube_node), OP_LOGE(kFusedOpType.c_str(), "Check op cube failed."),
                    return SUCCESS);

  fusion_nodes = GetMatchedNodes(mapping);

  for (auto &transdata1_node : transdata1_nodes) {
    ge::AttrUtils::SetStr(transdata1_node->GetOpDesc(), UB_FUSION_OP_TYPE, kOpConv2D);
  }

  OP_LOGD(kFusedOpType.c_str(), "End to do Conv2dTransDataFusionPass.");
  return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("Conv2dTransDataFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, Conv2dTransDataFusionPass);
}  // namespace fe
