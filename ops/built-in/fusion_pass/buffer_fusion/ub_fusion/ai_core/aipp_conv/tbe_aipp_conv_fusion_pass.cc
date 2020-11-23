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
 * \file tbe_aipp_conv_fusion_pass.cpp
 * \brief tbe aipp convolution ops fusion pattern
 */
#include "tbe_aipp_conv_fusion_pass.h"
#include <math.h>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "common/util/platform_info.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "tbe_aipp_fusion_rule.h"
#include "common/lxfusion_json_util.h"
#include "graph/utils/attr_utils.h"

namespace fe {
static const char kPatternAipp[] = "aipp";
static const char kPatternConv[] = "convolution";
static const char kPatternOtherInput[] = "otherInput";
static const char kPatternOtherInput1[] = "otherInput1";

/*
 * @brief:  define aipp and convolution op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    1) Aipp-->Convolution
 *
 * fusion node: Aipp, Convolution
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern*> TbeAippConvFusionPass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;
  string pass_name1 = "TbeAippConvFusion1";

  BufferFusionPattern* pattern1 = new (std::nothrow) BufferFusionPattern(pass_name1);
  FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pass_name1.c_str());
  // define pattern rules
  pattern1->AddOpDesc(kPatternAipp, {OP_PATTERN_AIPP}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternAipp})
      .SetOutputs(kPatternAipp, {kPatternConv})
      .SetOutputs(kPatternOtherInput, {kPatternConv});

  patterns.push_back(pattern1);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pass_name1.c_str());

  string pass_name2 = "TbeAippConvFusion2";
  BufferFusionPattern* pattern2 = new (std::nothrow) BufferFusionPattern(pass_name2);
  FUSION_PASS_CHECK((pattern2 == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pass_name2.c_str());
  // define pattern rules
  pattern2->AddOpDesc(kPatternAipp, {OP_PATTERN_AIPP}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternAipp})
      .SetOutputs(kPatternAipp, {kPatternConv})
      .SetOutputs(kPatternOtherInput, {kPatternConv})
      .SetOutputs(kPatternOtherInput1, {kPatternConv});

  patterns.push_back(pattern2);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pass_name2.c_str());
  return patterns;
}

void TbeAippConvFusionPass::DelSplitInfoByAxis(std::vector<AxisSplitMap> &split_maps, int axis) {
  std::vector<AxisSplitMap> temp_maps;
  for(auto it = split_maps.begin(); it != split_maps.end(); ++it) {
    bool del_axis = false;
    auto input_split_infos = (*it).GetInputSplitInfoVec();
    for (auto input_split_info : input_split_infos) {
      if (input_split_info.GetAxis()[0] == axis) {
        del_axis = true;
      }
    }
    if (!del_axis) {
      temp_maps.push_back(*it);
    }
  }
  split_maps = temp_maps;
}

/*  [images] <-
 *     |
 *    aipp
 *     |
 *   conv2d  [x]-> delete h/w
 *     |
 *  [output] ->
 */
void TbeAippConvFusionPass::SetSplitInfo(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes) {
  vector<ge::NodePtr> conv_nodes = GetMatchedNodesByDescName(kPatternConv, mapping);
  string op_slice_info_str = "";
  for (auto conv_node: conv_nodes) {
    ge::AttrUtils::GetStr(conv_node->GetOpDesc(), "_op_slice_info", op_slice_info_str);
  }
  OP_LOGD(fused_op_type_.c_str(), "ori _op_slice_info is %s", op_slice_info_str.c_str());
  OpCalcInfo op_calc_info;
  GetOpSliceInfoFromJson(op_calc_info, op_slice_info_str);
  auto split_maps = op_calc_info.GetAxisSplitMapVec();
  int h_axis = 2;
  int w_axis = 3;
  DelSplitInfoByAxis(split_maps, h_axis);
  DelSplitInfoByAxis(split_maps, w_axis);

  op_calc_info.SetL1FusionEnable(L1FUSION_DISABLE);
  op_calc_info.SetAxisSplitMaps(split_maps);
  SetOpSliceInfoToJson(op_calc_info, op_slice_info_str);
  for (auto fusion_node : fusion_nodes) {
    ge::AttrUtils::SetStr(fusion_node->GetOpDesc(), "_op_slice_info", op_slice_info_str);
  }
  OP_LOGD(fused_op_type_.c_str(), "set _op_slice_info is %s", op_slice_info_str.c_str());
}
/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status TbeAippConvFusionPass::GetFusionNodes(const BufferFusionMapping& mapping, vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(fused_op_type_.c_str(), "Begin to do TbeConvReluFusionPass!");
  vector<ge::NodePtr> conv_nodes = GetMatchedNodesByDescName(kPatternConv, mapping);
  vector<ge::NodePtr> aipp_nodes = GetMatchedNodesByDescName(kPatternAipp, mapping);

  string input_format = "";

  for (auto aipp_node : aipp_nodes) {
    string aipp_config_str = "";
    FUSION_PASS_CHECK(!ge::AttrUtils::GetStr(aipp_node->GetOpDesc(), "aipp_config_path", aipp_config_str),
                      OP_LOGI(fused_op_type_.c_str(), "Get node[%s]'s aipp_config_path attr not success.",
                              aipp_node->GetName().c_str()),
                      return FAILED);

    nlohmann::json aipp_config_json = nlohmann::json::parse(aipp_config_str);
    FUSION_PASS_CHECK(!aipp_config_json.is_object(),
                      OP_LOGE(fused_op_type_.c_str(), "The aipp_config_str is not an object, the aipp_config_str is %s.",
                              aipp_config_str.c_str()),
                      return FAILED);
    input_format = aipp_config_json["input_format"];
    OP_LOGI(fused_op_type_.c_str(), "aipp input_format is %s!", input_format.c_str());
  }

  for (auto conv_node : conv_nodes) {
    if (!TbeAippFusionRule::CheckConvload2dNodeValidation(conv_node)) {
      OP_LOGI(fused_op_type_.c_str(), "Node[%s] not satisfied with fusion condition.", conv_node->GetName().c_str());
      return SUCCESS;
    }
    if (!TbeAippFusionRule::CheckAippConvEltwiseFusionValidation(conv_node, input_format)) {
      OP_LOGI(fused_op_type_.c_str(),
              "The AIPP YUV exceed the L1 buffer, "
              "Node[%s] not satisfied with fusion condition.",
              conv_node->GetName().c_str());
      return SUCCESS;
    }
    if (!TbeAippFusionRule::CheckAippConvStridehValidation(conv_node)) {
      OP_LOGI(fused_op_type_.c_str(),
              "The case is the strideh optim. "
              "Node[%s] not satisfied with fusion condition.",
              conv_node->GetName().c_str());
      return SUCCESS;
    }
  }

  fusion_nodes = GetMatchedNodes(mapping);
  SetSplitInfo(mapping, fusion_nodes);
  OP_LOGD(fused_op_type_.c_str(), "End to do TbeAippConvFusionPass!");
  return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("TbeAippConvZ", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, TbeAippConvFusionPass);
}  // namespace fe
