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
#include "lx_fusion_func.h"

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

void TbeConv2DAddMulQuantPass::SetSplitInfo(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes) {
  vector<ge::NodePtr> conv_nodes = GetMatchedNodesByDescName(kPatternConv, mapping);
  if (conv_nodes.empty()) {
    OP_LOGD(fused_op_type_.c_str(), "conv node not matched");
    return;
  }
  vector<AxisSplitMap> split_maps;
  if (!GetSplitMap(split_maps, conv_nodes[0], fused_op_type_)) {
    return;
  }

  int inpre = conv_nodes[0]->GetInDataNodes().size() - 1;
  // dequant have one input, add have one input
  inpre += 2;
  // quant do not split c_out
  int c_axis = 1;
  // deq_scale do not have to process without c_out
  DelSplitInfoByInputAxis(split_maps, c_axis);
  for(auto it = split_maps.begin(); it != split_maps.end(); ++it) {
    auto exist_in = (*it).GetInputSplitInfoVec();
    if (!exist_in.empty()) {
      // the index 0 info is the base op info
      InputSplitInfo input_info;
      if(!input_info.Initialize()) {
        OP_LOGD(fused_op_type_.c_str(), "init input_info failed");
      } else {
        input_info.SetIndex(inpre);
        auto in_axis = exist_in[0].GetAxis();
        input_info.SetAxis(in_axis);
        auto head_overlap = exist_in[0].GetHeadOverLap();
        auto tail_overlap = exist_in[0].GetTailOverLap();
        input_info.SetHeadOverLap(head_overlap);
        input_info.SetTailOverLap(tail_overlap);
        (*it).AddInputSplitInfo(input_info);
      }
    }
    auto exist_out = (*it).GetOutputSplitInfoVec();
    if (!exist_out.empty()) {
      // the index 0 info is the base op info
      OutputSplitInfo output_info;
      if (!output_info.Initialize()) {
        OP_LOGD(fused_op_type_.c_str(), "init output_info failed");
      } else {
        int out_idx = 1;
        output_info.SetIndex(out_idx);
        auto out_axis = exist_out[0].GetAxis();
        output_info.SetAxis(out_axis);
        (*it).AddOutputSplitInfo(output_info);
      }
    }
  }
  SetSplitMap(split_maps, fusion_nodes, fused_op_type_);
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
  SetSplitInfo(mapping, fusion_nodes);
  OP_LOGD(fused_op_type_.c_str(), "End to do Conv2DAddMulQuant!");

  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("TbeConv2DAddMulQuantPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, TbeConv2DAddMulQuantPass);
}  // namespace fe
