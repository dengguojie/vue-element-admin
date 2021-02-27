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
      .AddOpDesc(kPatternQuant, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
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

  string pass_name2 = "TbeConvDequantWriteselectStridewriteFusion";
  BufferFusionPattern* pattern2 = new (std::nothrow) BufferFusionPattern(pass_name2);
  FUSION_PASS_CHECK((pattern2 == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pass_name2.c_str());
  // conv2d --> dequant --> write_select --> stride_write
  pattern2->AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternDequant, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternStridedWrite, {OP_PATTERN_STRIDED_WRITE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternWriteselect, {OP_PATTERN_WRITE_SELECT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternDequant})
      .SetOutputs(kPatternDequant, {kPatternWriteselect})
      .SetOutputs(kPatternWriteselect, {kPatternStridedWrite})
      .SetOutputs(kPatternOtherInput, {kPatternDequant});
  patterns.push_back(pattern2);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pass_name2.c_str());

  return patterns;
}

void TbeConv2dWrtselStridewrtPass::DelSplitInfoByAxis(std::vector<AxisSplitMap> &split_maps, int axis) {
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

void TbeConv2dWrtselStridewrtPass::SetSplitInfo(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes){
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
  vector<ge::NodePtr> conv_nodes = GetMatchedNodesByDescName(kPatternConv, mapping);
  int inpos = 0;
  string op_slice_info_str = "";
  for (auto conv_node : conv_nodes) {
    ge::AttrUtils::GetStr(conv_node->GetOpDesc(), fe::OP_SLICE_INFO, op_slice_info_str);
    inpos = conv_node->GetInDataNodes().size() - 1;
  }
  OP_LOGD(fused_op_type_.c_str(), "ori _op_slice_info is %s", op_slice_info_str.c_str());
  OpCalcInfo op_calc_info;
  GetOpSliceInfoFromJson(op_calc_info, op_slice_info_str);
  auto split_maps = op_calc_info.GetAxisSplitMapVec();
  if (split_maps.empty()) {
    OP_LOGW(fused_op_type_.c_str(), "axis split map vector is empty");
    return;
  }
  int h_axis = 2;
  int w_axis = 3;
  DelSplitInfoByAxis(split_maps, axis);
  DelSplitInfoByAxis(split_maps, h_axis);
  DelSplitInfoByAxis(split_maps, w_axis);
  if (existed_quant) {
    int c_axis = 1;
    DelSplitInfoByAxis(split_maps, c_axis);
  }

  if (is_add_deq_scale_splitinfo) {
    inpos += 1;
    for(auto it = split_maps.begin(); it != split_maps.end(); ++it) {
      auto input_split_infos = it->GetInputSplitInfos();
      for (auto input_split_info : input_split_infos) {
        if (!input_split_info->GetAxis().empty()) {
          if (input_split_info->GetAxis()[0] == 1) {
            InputSplitInfo deq_scale_splitinfo;
            vector<int64_t> minus_one_vec = {-1};
            vector<int64_t> one_vec = {1};
            deq_scale_splitinfo.SetIndex(inpos);
            deq_scale_splitinfo.SetAxis(one_vec);
            deq_scale_splitinfo.SetHeadOverLap(minus_one_vec);
            deq_scale_splitinfo.SetTailOverLap(minus_one_vec);
            it->AddInputSplitInfo(deq_scale_splitinfo);
          }
        }
      }
    }
  }

  op_calc_info.SetL1FusionEnable(L1FUSION_DISABLE);
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
