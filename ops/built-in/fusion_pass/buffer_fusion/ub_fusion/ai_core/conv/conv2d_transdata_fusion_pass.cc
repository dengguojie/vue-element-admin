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
#include <unistd.h>
#include <limits.h>

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
static const char kOpConv2D[] = "Conv2D";
static constexpr const char *BINARY_OPP_RELATIVE_PATH = "/op_impl/built-in/ai_core/tbe/kernel/config/";

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
  string pattern_name1 = "Conv2dTransDataFusionPassPattern1";
  BufferFusionPattern *pattern1 = new (std::nothrow) BufferFusionPattern(pattern_name1);
  FUSION_PASS_CHECK(pattern1 == nullptr, OP_LOGI(kFusedOpType.c_str(), "New an object failed."), return patterns);

  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pattern.", pattern_name1.c_str());
  pattern1->AddOpDesc(kPatternTransData1, {kOpTransData}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                      TBE_PATTERN_GROUPID_INVALID, ONLY_SUPPORT_DYNAMIC)
           .AddOpDesc(kPatternCube, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                      TBE_PATTERN_GROUPID_INVALID, ONLY_SUPPORT_DYNAMIC)
           .AddOpDesc(kPatternTransData2, {kOpTransData}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                      TBE_PATTERN_GROUPID_INVALID, ONLY_SUPPORT_DYNAMIC)
           .SetHead({kPatternTransData1})
           .SetOutputs(kPatternTransData1, {kPatternCube})
           .SetOutputs(kPatternCube, {kPatternTransData2}, TBE_OUTPUT_BRANCH_SINGLE, true);
  patterns.push_back(pattern1);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pattern.", pattern_name1.c_str());

  return patterns;
}

bool Conv2dTransDataFusionPass::CheckTransDataFormat(const ge::NodePtr &node, const bool &is_input) const {
  bool is_support_format = false;
  if (is_input) {
    is_support_format = (node->GetOpDesc()->GetInputDesc(0).GetFormat() == ge::FORMAT_NCHW) &&
                        (node->GetOpDesc()->GetOutputDesc(0).GetFormat() == ge::FORMAT_NC1HWC0);
    FUSION_PASS_CHECK(!is_support_format,
                      OP_LOGI(kFusedOpType.c_str(), "Only support format NCHW of input."),
                      return is_support_format);
  } else {
    is_support_format = (node->GetOpDesc()->GetInputDesc(0).GetFormat() == ge::FORMAT_NC1HWC0) &&
                        (node->GetOpDesc()->GetOutputDesc(0).GetFormat() == ge::FORMAT_NCHW);
    FUSION_PASS_CHECK(!is_support_format,
                      OP_LOGI(kFusedOpType.c_str(), "Only support format NC1HWC0 of output."),
                      return is_support_format);
  }
  return is_support_format;
}

bool Conv2dTransDataFusionPass::CheckInputNoRange(const ge::NodePtr &cube_node) const {
  auto input_desc = GetCurrNodeMutableInputDesc(cube_node, 0);
  if (input_desc == nullptr) {
    OP_LOGI(kFusedOpType.c_str(), "Failed to get input_desc of cube_node.");
    return false;
  }
  vector<int64_t> input_dims = input_desc->GetOriginShape().GetDims();
  for (auto input_dim : input_dims) {
    FUSION_PASS_CHECK(input_dim != -1,
                      OP_LOGI(kFusedOpType.c_str(), "Only support dynamic nchw."), return false);
  }

  vector<pair<int64_t, int64_t>> range_data;
  FUSION_PASS_CHECK(input_desc->GetShapeRange(range_data) == ge::GRAPH_FAILED,
                    OP_LOGI(kFusedOpType.c_str(), "Failed to get input shape range of cube_node."),
                    return false);
  FUSION_PASS_CHECK(range_data.empty(),
                    OP_LOGI(kFusedOpType.c_str(), "range_data is empty."),
                    return false);

  // range: ((1, -1), (1, -1), (1, -1), (1, -1), (16, 16))
  for (size_t i = 0; i < range_data.size() - 1; i++) {
    FUSION_PASS_CHECK(range_data[i] != kNoRange,
                      OP_LOGI(kFusedOpType.c_str(), "Only support input no shape range."), return false);
  }
  return true;
}

bool Conv2dTransDataFusionPass::CheckBinaryReuse() const
{
  PlatformInfo platformInfo;
  OptionalInfo optiCompilationInfo;
  const char *ascendOppPath = nullptr;
  char oppParentPath[PATH_MAX] = {0};

  if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(
      platformInfo, optiCompilationInfo) != fe::SUCCESS) {
    OP_LOGD("Fail to get platform info. ");
    return false;
  }
  OP_LOGD(kFusedOpType.c_str(), "Get optiCompilationInfo.soc_version[%s]", optiCompilationInfo.soc_version.c_str());

  std::string socVersion = optiCompilationInfo.soc_version;
  std::transform(socVersion.begin(), socVersion.end(), socVersion.begin(), ::tolower);

  std::string numStr = "0123456789";
  size_t firstNumPos = socVersion.find_first_of(numStr);
  std::string subSocVersion;

  if (firstNumPos == std::string::npos) {
    subSocVersion = socVersion;
  } else {
    size_t lastNumPos = socVersion.find_first_not_of(numStr, firstNumPos);
    subSocVersion = socVersion.substr(0, lastNumPos);
  }

  ascendOppPath = std::getenv("ASCEND_OPP_PATH");
  if (ascendOppPath == nullptr) {
    OP_LOGD("Fail to get opp path. ");
    return false;
  }

  if (realpath(ascendOppPath, oppParentPath) == nullptr) {
    OP_LOGD("Fail to get opp realpath. ");
    return false;
  }

  const std::string oppKernelPath = string(oppParentPath) + BINARY_OPP_RELATIVE_PATH +
                                    subSocVersion + "/fusion_ops.json";
  OP_LOGD(kFusedOpType.c_str(), "Get oppKernelPath[%s]", oppKernelPath.c_str());

  if (access(oppKernelPath.c_str(), F_OK) != -1) {
    OP_LOGD("opp kernel binary file exists. ");
    return true;
  }
  return false;
}

bool Conv2dTransDataFusionPass::CheckOpCube(const ge::NodePtr &cube_node) const {
  FUSION_PASS_CHECK(cube_node->GetType() != kOpConv2D,
                    OP_LOGD(kFusedOpType.c_str(),
                            "The op_type of node [%s] should be Conv2D, but actually is [%s].",
                            cube_node->GetName().c_str(), cube_node->GetType().c_str()),
                    return false);

  if (!CheckBinaryReuse()) {
    OP_LOGD(kFusedOpType.c_str(), "Cannot reuse kernel binary, check input no range. ");
    FUSION_PASS_CHECK(!CheckInputNoRange(cube_node),
                      OP_LOGD(kFusedOpType.c_str(), "Check input shape dim and range fail."),
                      return false);
  }
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
  FUSION_PASS_CHECK(transdata1_nodes.empty(),
                    OP_LOGD(kFusedOpType.c_str(), " Pre TransData node is not matched."),
                    return SUCCESS);
  FUSION_PASS_CHECK(cube_nodes.empty(),
                    OP_LOGD(kFusedOpType.c_str(), "Conv2d node is not matched."),
                    return SUCCESS);
  FUSION_PASS_CHECK(transdata2_nodes.empty(),
                    OP_LOGD(kFusedOpType.c_str(), "Post TransData node is not matched."),
                    return SUCCESS);

  ge::NodePtr cube_node = cube_nodes[0];
  FUSION_PASS_CHECK(!CheckOpCube(cube_node), OP_LOGI(kFusedOpType.c_str(), "Check op cube failed."),
                    return SUCCESS);

  for (auto &transdata1_node : transdata1_nodes) {
    FUSION_PASS_CHECK(transdata1_node == nullptr,
                      OP_LOGI(kFusedOpType.c_str(), "Failed to get pre transData node. "),
                      return SUCCESS);
    FUSION_PASS_CHECK(!CheckTransDataFormat(transdata1_node, true),
                      OP_LOGI(kFusedOpType.c_str(), "unsupport input TransData format. "),
                      return SUCCESS);
    ge::OpDescPtr transdata_desc = transdata1_node->GetOpDesc();
    FUSION_PASS_CHECK(transdata_desc == nullptr,
                      OP_LOGI(kFusedOpType.c_str(), "Failed to get opdesc of pre transData node. "),
                      return SUCCESS);
    FUSION_PASS_CHECK(!ge::AttrUtils::SetStr(transdata_desc, UB_FUSION_OP_TYPE, kOpConv2D),
                      OP_LOGI(kFusedOpType.c_str(), "set op_type from TransData to Conv2D failed. "),
                      return SUCCESS);
    FUSION_PASS_CHECK(!ge::AttrUtils::SetStr(transdata_desc, "graph_pattern", "_transdata_conv2d_transdata"),
                      OP_LOGI(kFusedOpType.c_str(), "set graph_pattern failed. "),
                      return SUCCESS);
  }
  FUSION_PASS_CHECK(!CheckTransDataFormat(transdata2_nodes[0], false),
                    OP_LOGI(kFusedOpType.c_str(), "unsupport output TransData format. "),
                    return SUCCESS);

  fusion_nodes = GetMatchedNodes(mapping);

  OP_LOGD(kFusedOpType.c_str(), "End to do Conv2dTransDataFusionPass.");
  return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("Conv2dTransDataFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, Conv2dTransDataFusionPass);
}  // namespace fe
