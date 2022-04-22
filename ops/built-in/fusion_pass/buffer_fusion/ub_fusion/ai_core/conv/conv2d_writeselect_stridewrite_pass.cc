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
 * \file conv2d_writeselect_stridewrite_pass.cpp
 * \brief  tbe conv2d + write_select + stride_write ops fusion pattern
 */
#include "conv2d_writeselect_stridewrite_pass.h"
#include <string>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "common/lxfusion_json_util.h"
#include "graph/utils/attr_utils.h"
#include "lx_fusion_func.h"

namespace fe {

static const char kPatternConv[] = "conv2d";
static const char kPatternDequant[] = "dequant";
static const char kPatternQuant[] = "quant";
static const char kPatternWriteselect[] = "writeselect";
static const char kPatternStridedWrite[] = "stridedwrite";
static const char kPatternOtherInput[] = "otherInput";
static const int64_t kMaxFuseNode = 6;

/*
 * @brief:  define conv2d op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    conv2d --> dequant --> quant --> write_select --> stride_write
 *    conv2d --> dequant --> write_select --> stride_write
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern*> TbeConv2dWrtselStridewrtPass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;
  string pass_name1 = "TbeConvDequantQuantWriteselectStridewriteFusion";
  BufferFusionPattern* pattern1 = new (std::nothrow) BufferFusionPattern(pass_name1, kMaxFuseNode);
  FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pass_name1.c_str());
  // conv2d --> dequant --> quant --> write_select --> stride_write
  pattern1->AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternDequant, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternQuant, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternStridedWrite, {OP_PATTERN_STRIDED_WRITE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternWriteselect, {OP_PATTERN_WRITE_SELECT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternDequant})
      .SetOutputs(kPatternOtherInput, {kPatternDequant})
      .SetOutputs(kPatternDequant, {kPatternQuant})
      .SetOutputs(kPatternQuant, {kPatternWriteselect})
      .SetOutputs(kPatternWriteselect, {kPatternStridedWrite});
  patterns.push_back(pattern1);

  return patterns;
}

void TbeConv2dWrtselStridewrtPass::SetSplitInfo(const BufferFusionMapping &mapping,
                                                std::vector<ge::NodePtr> &fusion_nodes) {
  vector<ge::NodePtr> conv_nodes = GetMatchedNodesByDescName(kPatternConv, mapping);
  if (conv_nodes.empty()) {
    OP_LOGD(fused_op_type_.c_str(), "conv node not matched");
    return;
  }
  vector<AxisSplitMap> split_maps;
  OpL1FusionType L1_fusion_type = L1FUSION_DISABLE;
  int64_t min_tbe_L1Space = 0;
  if (!GetSplitMap(split_maps, conv_nodes[0], fused_op_type_, L1_fusion_type, min_tbe_L1Space)) {
    return;
  }
  bool existed_quant = false;
  vector<ge::NodePtr> quant_nodes = GetMatchedNodesByDescName(kPatternQuant, mapping);
  if (!quant_nodes.empty()){
    existed_quant = true;
  }

  bool is_add_deq_scale_splitinfo = false;
  vector<int64_t> deq_scale_shape;
  if (!existed_quant) {
    vector<ge::NodePtr> dequant_nodes = GetMatchedNodesByDescName(kPatternDequant, mapping);
    for (auto dequant_node : dequant_nodes) {
      deq_scale_shape = dequant_node->GetOpDesc()->GetInputDesc("deq_scale").GetOriginShape().GetDims();
      if(!(deq_scale_shape.size() == 1 && deq_scale_shape[0] == 1)) {
        is_add_deq_scale_splitinfo = true;
      }
    }
  }

  int32_t axis = 0;
  vector<ge::NodePtr> stride_write_nodes = GetMatchedNodesByDescName(kPatternStridedWrite, mapping);
  for (auto stride_write_node : stride_write_nodes) {
    AttrUtils::GetInt(stride_write_node->GetOpDesc(), "axis", axis);
  }
  int h_axis = 2;
  int w_axis = 3;
  // stride_write axis do not split
  DelSplitInfoByInputAxis(split_maps, axis);
  DelSplitInfoByInputAxis(split_maps, h_axis);
  DelSplitInfoByInputAxis(split_maps, w_axis);
  if (existed_quant) {
    int c_axis = 1;
    DelSplitInfoByInputAxis(split_maps, c_axis);
  }

  int inpos = conv_nodes[0]->GetInDataNodes().size() - 1;
  if (is_add_deq_scale_splitinfo) {
    inpos += 1;
    for(auto it = split_maps.begin(); it != split_maps.end(); ++it) {
      auto exist_out = (*it).GetOutputSplitInfoVec();
      std::vector<int64_t> c_out = {1};
      bool valid = !exist_out.empty() && exist_out[0].GetAxis() == c_out;
      if (valid) {
        // process dequant deq_scale if exists
        auto exist_in = (*it).GetInputSplitInfoVec();
        if (!exist_in.empty()) {
          InputSplitInfo input_info;
          if (!input_info.Initialize()) {
            OP_LOGD(fused_op_type_.c_str(), "init input_info failed");
          } else {
            input_info.SetIndex(inpos);
            // deq_scale is 5hd format
            std::vector<int64_t> axis_c = {1};
            input_info.SetAxis(axis_c);
            // the index 0 info is the base op info
            auto head_overlap = exist_in[0].GetHeadOverLap();
            auto tail_overlap = exist_in[0].GetTailOverLap();
            input_info.SetHeadOverLap(head_overlap);
            input_info.SetTailOverLap(tail_overlap);
            (*it).AddInputSplitInfo(input_info);
          }
        }
        break;
      }
    }
  }
  SetSplitMap(split_maps, fusion_nodes, fused_op_type_, L1_fusion_type, min_tbe_L1Space);
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status TbeConv2dWrtselStridewrtPass::GetFusionNodes(const BufferFusionMapping& mapping, vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(fused_op_type_.c_str(), "Begin to do Conv2dWrtselStridewrt!");
  fusion_nodes = GetMatchedNodes(mapping);
  SetSplitInfo(mapping, fusion_nodes);
  OP_LOGD(fused_op_type_.c_str(), "End to do Conv2dWrtselStridewrt!");

  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("TbeConv2dWrtselStridewrtPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeConv2dWrtselStridewrtPass);
}  // namespace fe
