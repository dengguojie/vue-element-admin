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
 * \file conv2d_dequant_add_mul_quant_pass.cpp
 * \brief tbe conv2d + add + mul + quant ops fusion pattern
 */
#include "conv2d_dequant_add_mul_quant_pass.h"
#include <string>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "common/lxfusion_json_util.h"
#include "graph/utils/attr_utils.h"

namespace fe {

static const char kPatternConv[] = "conv2d";
static const char kPatternDeq[] = "dequant";
static const char kPatternAdd[] = "add";
static const char kPatternQuant[] = "quant";
static const char kPatternOtherInput[] = "otherInput";
static const char kPatternOtherInput1[] = "otherInput1";
static const char kPatternOutput1[] = "OUTPUT1";
static const char kPatternOutput2[] = "OUTPUT2";

/*
 * @brief:  define conv2d op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    conv2d --> dequant --> add --> quant
 *                            | --> other
 *                            \ --> other
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern*> TbeConv2DAddMulQuantPass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;
  string pass_name1 = "TbeConv2DAddMutioutQuantFusion";
  BufferFusionPattern* pattern1 = new (std::nothrow) BufferFusionPattern(pass_name1);
  FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE(fused_op_type_.c_str(), "create new pattern failed."),
                    return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pass_name1.c_str());
  pattern1->AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternDeq, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternAdd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternQuant, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOutput1, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOutput2, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternDeq})
      .SetOutputs(kPatternOtherInput, {kPatternDeq})
      .SetOutputs(kPatternDeq, {kPatternAdd})
      .SetOutputs(kPatternOtherInput1, {kPatternAdd})
      .SetOutputs(kPatternAdd, {kPatternQuant, kPatternOutput1, kPatternOutput2}, TBE_OUTPUT_BRANCH_MULTI);
  patterns.push_back(pattern1);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pass_name1.c_str());

  return patterns;
}

void TbeConv2DAddMulQuantPass::DelSplitInfoByAxis(std::vector<AxisSplitMap> &split_maps, int axis) {
  std::vector<AxisSplitMap> temp_maps;
  for(auto it = split_maps.begin(); it != split_maps.end(); ++it) {
    bool del_axis = false;
    auto input_split_infos = (*it).GetInputSplitInfoVec();
    for (auto input_split_info : input_split_infos) {
      if (!input_split_info.GetAxis().empty()) {
        if (input_split_info.GetAxis()[0] == axis) {
          del_axis = true;
        }
      }
    }
    if (!del_axis) {
      temp_maps.push_back(*it);
    }
  }
  split_maps = temp_maps;
}

void TbeConv2DAddMulQuantPass::SetSplitInfo(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes) {
  vector<ge::NodePtr> conv_nodes = GetMatchedNodesByDescName(kPatternConv, mapping);
  int inpre = 0;
  string op_slice_info_str = "";
  for (auto conv_node : conv_nodes) {
    inpre = conv_node->GetInDataNodes().size() - 1;
    ge::AttrUtils::GetStr(conv_node->GetOpDesc(), fe::OP_SLICE_INFO, op_slice_info_str);
  }
  OP_LOGD(fused_op_type_.c_str(), "ori _op_slice_info is %s", op_slice_info_str.c_str());

  // dequant have one input, add have one input
  inpre += 2;
  OpCalcInfo op_calc_info;
  GetOpSliceInfoFromJson(op_calc_info, op_slice_info_str);
  auto split_maps = op_calc_info.GetAxisSplitMapVec();
  if (split_maps.empty()) {
    OP_LOGD(fused_op_type_.c_str(), "axis split map vector is empty");
    return;
  }
  int c_axis = 1;
  DelSplitInfoByAxis(split_maps, c_axis);
  for(auto it = split_maps.begin(); it != split_maps.end(); ++it) {
    auto input_split_infos = (*it).GetInputSplitInfoVec();
    vector<int64_t> minus_one_vec = {-1};
    if (input_split_infos.empty()) {
      continue;
    }
    vector<int64_t> axis_vec = input_split_infos[0].GetAxis();
    InputSplitInfo input_split_info = input_split_infos[0];
    input_split_info.SetIndex(inpre);
    input_split_info.SetAxis(axis_vec);
    input_split_info.SetHeadOverLap(minus_one_vec);
    input_split_info.SetTailOverLap(minus_one_vec);
    (*it).AddInputSplitInfo(input_split_info);

    auto output_split_infos = (*it).GetOutputSplitInfoVec();
    if (output_split_infos.empty()) {
      continue;
    }
    OutputSplitInfo output_split_info = output_split_infos[0];
    output_split_info.SetIndex(1);
    output_split_info.SetAxis(axis_vec);
    (*it).AddOutputSplitInfo(output_split_info);
  }
  op_calc_info.SetAxisSplitMaps(split_maps);
  SetFusionOpSliceInfoToJson(op_calc_info, op_slice_info_str);
  for (auto fusion_node : fusion_nodes) {
    ge::AttrUtils::SetStr(fusion_node->GetOpDesc(), fe::FUSION_OP_SLICE_INFO, op_slice_info_str);
  }
  OP_LOGD(fused_op_type_.c_str(), "set _fusion_op_slice_info is %s", op_slice_info_str.c_str());
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status TbeConv2DAddMulQuantPass::GetFusionNodes(const BufferFusionMapping& mapping, vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(fused_op_type_.c_str(), "Begin to do Conv2DAddMulQuant!");
  fusion_nodes = GetMatchedNodes(mapping);
  // the outputData can't be fusd
  for (auto& item : mapping) {
    auto opdesc = find(item.first->types.begin(), item.first->types.end(), TBE_PATTERN_OUTPUT_NODE);

    if (opdesc != item.first->types.end()) {
      for (auto& node : item.second) {
        auto nodePtr = find(fusion_nodes.begin(), fusion_nodes.end(), node);
        fusion_nodes.erase(nodePtr);
      }
    }
  }
  SetSplitInfo(mapping, fusion_nodes);
  OP_LOGD(fused_op_type_.c_str(), "End to do Conv2DAddMulQuant!");

  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("TbeConv2DAddMulQuantPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, TbeConv2DAddMulQuantPass);
}  // namespace fe
