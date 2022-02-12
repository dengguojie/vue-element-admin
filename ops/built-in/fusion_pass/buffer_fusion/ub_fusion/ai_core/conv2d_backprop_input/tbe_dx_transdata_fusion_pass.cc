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

#include "tbe_dx_transdata_fusion_pass.h"

#include "anchor_util.h"
#include "common/util/platform_info.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {
static const char kPatternTransData1[] = "transdata1";
static const char kPatternCube[] = "cube";
static const char kPatternTransData2[] = "transdata2";

static const char kOpTransData[] = "TransData";
static const char kOpConv2DBackpropInput[] = "Conv2DBackpropInput";

static vector<int64_t> kDilationsDefault = {1, 1, 1, 1};
static pair<int64_t, int64_t> kNoRange = {1, -1};
/*
 * @brief: define (transdata) + dx + transdata ub fusion pattern
 *
 * grad                  weight
 *    \                  /
 *    (transdata_1)    (transdata_other)
 *       \           /
 *    conv2d_backprop_input
 *              |
 *         transdata_2
 *
 * @return BufferFusionPattern: return all valid patterns
 */
vector<BufferFusionPattern *> TbeDxTransDataFusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;
  string pass_name = "TbeDxTransDataFusionPass";
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(pass_name);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGW(kFusedOpType.c_str(), "New an object failed."), return patterns);

  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pattern.", pass_name.c_str());
  pattern->AddOpDesc(kPatternTransData1, {kOpTransData}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT,
                     TBE_PATTERN_GROUPID_INVALID, ONLY_SUPPORT_DYNAMIC, true)
          .AddOpDesc(kPatternCube, {OP_PATTERN_CONV_BACKPROP_INPUT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                     TBE_PATTERN_GROUPID_INVALID, ONLY_SUPPORT_DYNAMIC)
          .AddOpDesc(kPatternTransData2, {kOpTransData}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                     TBE_PATTERN_GROUPID_INVALID, ONLY_SUPPORT_DYNAMIC, true)
          .SetHead({kPatternTransData1, kPatternCube})
          .SetOutputs(kPatternTransData1, {kPatternCube})
          .SetOutputs(kPatternCube, {kPatternTransData2}, TBE_OUTPUT_BRANCH_SINGLE, true);
  patterns.push_back(pattern);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pattern.", pass_name.c_str());

  return patterns;
}

bool TbeDxTransDataFusionPass::CheckPlatform() const {
  PlatformInfo platformInfo;
  OptionalInfo optionalInfo;
  FUSION_PASS_CHECK(
    PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS,
    OP_LOGW(kFusedOpType.c_str(), "Get platform info failed."),
    return false);
  FUSION_PASS_CHECK(optionalInfo.soc_version.find("Ascend910") == string::npos,
                    OP_LOGW(kFusedOpType.c_str(), "Only support platform Ascend910"),
                    return false);
  return true;
}

bool TbeDxTransDataFusionPass::CheckTransDataFormat(const ge::NodePtr &node, const bool &is_input) const {
  bool is_support_format = false;
  if (is_input) {
    is_support_format = (node->GetOpDesc()->GetInputDesc(0).GetFormat() == ge::FORMAT_NCHW) &&
                        (node->GetOpDesc()->GetOutputDesc(0).GetFormat() == ge::FORMAT_NC1HWC0);
    FUSION_PASS_CHECK(!is_support_format,
                      OP_LOGW(kFusedOpType.c_str(), "Only support format NCHW of input."),
                      return is_support_format);
  } else {
    is_support_format = (node->GetOpDesc()->GetInputDesc(0).GetFormat() == ge::FORMAT_NC1HWC0) &&
                        (node->GetOpDesc()->GetOutputDesc(0).GetFormat() == ge::FORMAT_NCHW);
    FUSION_PASS_CHECK(!is_support_format,
                      OP_LOGW(kFusedOpType.c_str(), "Only support format NC1HWC0 of output."),
                      return is_support_format);
  }
  return is_support_format;
}

bool TbeDxTransDataFusionPass::CheckInputNoRange(const ge::NodePtr &cube_node) const {
  // 2 means out_backprop
  auto input_desc = GetCurrNodeMutableInputDesc(cube_node, 2);
  vector<int64_t> input_dims = input_desc->GetOriginShape().GetDims();
  for (auto input_dim : input_dims) {
    FUSION_PASS_CHECK(input_dim != -1,
                      OP_LOGW(kFusedOpType.c_str(), "Only support dynamic nchw."), return false);
  }

  vector<pair<int64_t, int64_t> > range_data;
  FUSION_PASS_CHECK(input_desc->GetShapeRange(range_data) == ge::GRAPH_FAILED,
                    OP_LOGW(kFusedOpType.c_str(), "Failed to get input shape range of cube_node:2."),
                    return false);
  FUSION_PASS_CHECK(range_data.empty(),
                    OP_LOGW(kFusedOpType.c_str(), "range_data is empty."),
                    return false);

  // range: ((1, -1), (1, -1), (1, -1), (1, -1), (16, 16))
  for (size_t i = 0; i < range_data.size() - 1; i++) {
    FUSION_PASS_CHECK(range_data[i] != kNoRange,
                      OP_LOGW(kFusedOpType.c_str(), "Only support nchw no range."), return false);
  }
  return true;
}

bool TbeDxTransDataFusionPass::CheckOpCube(const ge::NodePtr &cube_node) const {
  FUSION_PASS_CHECK(cube_node->GetType() != kOpConv2DBackpropInput,
                    OP_LOGD(kFusedOpType.c_str(),
                            "The op_type of node [%s] should be Conv2DBackpropInput, but actually is [%s].",
                            cube_node->GetName().c_str(), cube_node->GetType().c_str()),
                    return false);
  int64_t groups = 1;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetInt(cube_node->GetOpDesc(), "groups", groups),
                    OP_LOGW(kFusedOpType.c_str(), "Get attr groups of Conv2DBackpropInput node failed."),
                    return false);
  FUSION_PASS_CHECK(groups != 1,
                    OP_LOGW(kFusedOpType.c_str(), "Only support groups = 1."),
                    return false);
  vector<int64_t> dilations;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(cube_node->GetOpDesc(), "dilations", dilations),
                    OP_LOGW(kFusedOpType.c_str(), "Get attr dilations of Conv2DBackpropInput node failed."),
                    return false);
  FUSION_PASS_CHECK((dilations != kDilationsDefault),
                    OP_LOGW(kFusedOpType.c_str(), "Only support dilations = {1, 1, 1, 1}."),
                    return false);

  FUSION_PASS_CHECK(!CheckInputNoRange(cube_node),
                    OP_LOGD(kFusedOpType.c_str(), "Only support NCHW no range."),
                    return false);
  return true;
}

void TbeDxTransDataFusionPass::DeleteFusionNodes(const ge::NodePtr &transdata_node,
                                                 vector<ge::NodePtr> &fusion_nodes,
                                                 const bool &erase_transdata_flag,
                                                 const bool &is_input) {
  if (erase_transdata_flag) {
    auto iter = find(fusion_nodes.begin(), fusion_nodes.end(), transdata_node);
    if (iter != fusion_nodes.end()) {
      fusion_nodes.erase(iter);
    }
  }
  if (!CheckTransDataFormat(transdata_node, is_input)) {
    fusion_nodes.clear();
  }
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status TbeDxTransDataFusionPass::GetFusionNodes(const BufferFusionMapping &mapping,
                                                vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(kFusedOpType.c_str(), "Begin to do TbeDxTransDataFusionPass.");
  FUSION_PASS_CHECK(!CheckPlatform(),
                    OP_LOGW(kFusedOpType.c_str(), "Platform unsupported, abort fusion."),
                    return SUCCESS);

  vector<ge::NodePtr> transdata1_nodes = GetMatchedNodesByDescName(kPatternTransData1, mapping);
  vector<ge::NodePtr> cube_nodes = GetMatchedNodesByDescName(kPatternCube, mapping);
  vector<ge::NodePtr> transdata2_nodes = GetMatchedNodesByDescName(kPatternTransData2, mapping);
  FUSION_PASS_CHECK(cube_nodes.empty(),
                    OP_LOGD(kFusedOpType.c_str(), "Conv2DBackpropInput node is not matched."),
                    return SUCCESS);
  FUSION_PASS_CHECK(transdata2_nodes.empty(),
                    OP_LOGD(kFusedOpType.c_str(), "TransData node of output is not matched."),
                    return SUCCESS);

  ge::NodePtr cube_node = cube_nodes[0];
  FUSION_PASS_CHECK(!CheckOpCube(cube_node), OP_LOGW(kFusedOpType.c_str(), "Check op cube failed."),
                    return SUCCESS);
  vector<int64_t> strides;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(cube_node->GetOpDesc(), "strides", strides),
                    OP_LOGW(kFusedOpType.c_str(), "Get attr strides of Conv2DBackpropInput node failed."),
                    return SUCCESS);
  auto iter = find_if(strides.begin(), strides.end(), [](int64_t stride) { return stride > 1; });
  bool erase_transdata_flag = (iter != strides.end());
  FUSION_PASS_CHECK(!erase_transdata_flag && transdata1_nodes.empty(),
                    OP_LOGW(kFusedOpType.c_str(), "There is no transdata of input when stride is 1"),
                    return SUCCESS);

  fusion_nodes = GetMatchedNodes(mapping);

  if (!transdata1_nodes.empty()) {
    DeleteFusionNodes(transdata1_nodes[0], fusion_nodes, erase_transdata_flag, true);
  }
  DeleteFusionNodes(transdata2_nodes[0], fusion_nodes, false, false);

  for (auto &transdata1_node : transdata1_nodes) {
    ge::AttrUtils::SetStr(transdata1_node->GetOpDesc(), UB_FUSION_OP_TYPE, kOpConv2DBackpropInput);
  }

  OP_LOGD(kFusedOpType.c_str(), "End to do TbeDxTransDataFusionPass.");
  return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("TbeDxTransDataFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, TbeDxTransDataFusionPass);
}  // namespace fe
