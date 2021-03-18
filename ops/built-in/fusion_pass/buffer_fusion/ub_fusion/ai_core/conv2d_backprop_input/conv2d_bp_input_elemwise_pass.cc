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

/*
 * \file conv2d_bp_input_elemwise_pass.cpp
 * \brief tbe conv2d_backprop_input + elemwise ops fusion pattern
 */
#include "conv2d_bp_input_elemwise_pass.h"
#include <string>
#include "pattern_fusion_util.h"
#include "op_log.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "common/lxfusion_json_util.h"
#include "graph/utils/attr_utils.h"

namespace fe {

static const char PATTERN_DX[] = "conv2dbackpropinput";
static const char PATTERN_ELEM[] = "elemwise";
static const char PATTERN_OTHER_INPUT[] = "InputData";
static vector<string> typelist = {"Relu", "LeakyRelu", "PRelu"};
/*
 * @brief:  define conv2dbackpropinput op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    conv2d_backprop_input --> elemwise
 *    now only supporse relu/leakyrelu/prelu
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern*> TbeDxElemwisePass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;
  string pass_name = "TbeDxElemwiseFusion";
  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pass_name);

  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", pass_name.c_str());
  pattern->AddOpDesc(PATTERN_DX, {OP_PATTERN_CONV_BACKPROP_INPUT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEM, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_DX})
      .SetOutputs(PATTERN_DX, {PATTERN_ELEM});
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", pass_name.c_str());

  string pass_name0 = "TbeDxElemwiseFusion0";
  BufferFusionPattern* pattern0 = new (std::nothrow) BufferFusionPattern(pass_name0);
  FUSION_PASS_CHECK((pattern0 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", pass_name0.c_str());

  pattern0->AddOpDesc(PATTERN_DX, {OP_PATTERN_CONV_BACKPROP_INPUT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEM, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_DX})
      .SetOutputs(PATTERN_DX, {PATTERN_ELEM})
      .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_ELEM});
  patterns.push_back(pattern0);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", pass_name0.c_str());

  return patterns;
}

void TbeDxElemwisePass::SetSplitInfo(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes) {
  vector<ge::NodePtr> deconv_nodes = GetMatchedNodesByDescName(PATTERN_DX, mapping);
  if (deconv_nodes.empty()) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Deconv node not matched");
    return;
  }
  vector<ge::NodePtr> elemwise_nodes = GetMatchedNodesByDescName(PATTERN_ELEM, mapping);
  if (elemwise_nodes.empty()) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Elemwise node not matched");
    return;
  }

  int inpre = 0;
  string op_slice_info_str = "";
  for (auto deconv_node: deconv_nodes) {
    inpre = deconv_node->GetInDataNodes().size() - 1;
    ge::AttrUtils::GetStr(deconv_node->GetOpDesc(), fe::OP_SLICE_INFO, op_slice_info_str);
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "ori _op_slice_info is %s", op_slice_info_str.c_str());

  OpCalcInfo op_calc_info;
  GetOpSliceInfoFromJson(op_calc_info, op_slice_info_str);
  auto split_maps = op_calc_info.GetAxisSplitMapVec();
  if (split_maps.empty()) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "axis split map vector is empty");
    return;
  }
  // when deconv + prelu, add one input
  if (elemwise_nodes[0]->GetType() == "PRelu") {
    inpre += 1;
    vector<int64_t> cout_dim = {1};
    vector<int64_t> split_flag = {0};
    for(auto it = split_maps.begin(); it != split_maps.end(); ++it) {
      auto output_split_infos = (*it).GetOutputSplitInfoVec();
      if (output_split_infos.empty()) {
        OP_LOGW(FUSED_OP_TYPE.c_str(), "output_split_infos is empty");
        return;
      }
      if (output_split_infos[0].GetAxis()[0] == 1) {
        InputSplitInfo input_split_info;
        input_split_info.SetIndex(inpre);
        input_split_info.SetAxis(cout_dim);
        input_split_info.SetHeadOverLap(split_flag);
        input_split_info.SetTailOverLap(split_flag);
        (*it).AddInputSplitInfo(input_split_info);
      }
    }
    op_calc_info.SetAxisSplitMaps(split_maps);
    SetFusionOpSliceInfoToJson(op_calc_info, op_slice_info_str);
  }

  for (auto fusion_node : fusion_nodes) {
    ge::AttrUtils::SetStr(fusion_node->GetOpDesc(), fe::FUSION_OP_SLICE_INFO, op_slice_info_str);
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "set _fusion_op_slice_info is %s", op_slice_info_str.c_str());
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status TbeDxElemwisePass::GetFusionNodes(const BufferFusionMapping& mapping, vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do conv2d_bp_input_elemwise!");

  fusion_nodes = GetMatchedNodes(mapping);

  // buffer fusion do not support dynamic shape now
  vector<ge::NodePtr> dxNodes = GetMatchedNodesByDescName(PATTERN_DX, mapping);
  for (const auto& dxNode : dxNodes){
    vector<int64_t> input0Dims = dxNode->GetOpDesc()->GetInputDesc(0).GetOriginShape().GetDims();
    vector<int64_t> input1Dims = dxNode->GetOpDesc()->GetInputDesc(1).GetOriginShape().GetDims();
    vector<int64_t> allDims;
    allDims.resize(input0Dims.size() + input1Dims.size());
    merge(input0Dims.begin(), input0Dims.end(), input1Dims.begin(), input1Dims.end(), allDims.begin());
    for (auto singleDim : allDims) {
      if (singleDim < 0) {
        fusion_nodes.clear();
        OP_LOGW(FUSED_OP_TYPE.c_str(), "ub fusion not support dynamic shape");
        return SUCCESS;
      }
    }
  }

  // the outputData can't be fused
  for (auto& item : mapping) {
    auto opdesc = find(item.first->types.begin(), item.first->types.end(), TBE_PATTERN_OUTPUT_NODE);
    if (opdesc != item.first->types.end()) {
      for (auto& node : item.second) {
        auto node_ptr = find(fusion_nodes.begin(), fusion_nodes.end(), node);
        fusion_nodes.erase(node_ptr);
      }
    }
  }
  for (auto& item : mapping) {
    // judge LeakyRelu/Relu/Prelu node
    if (item.first->desc_name == PATTERN_ELEM) {
      for (auto& node : item.second) {
        vector<string>::iterator ret;
        ret = find(typelist.begin(), typelist.end(), node->GetType());
        if (ret != typelist.end()) {
          OP_LOGD(FUSED_OP_TYPE.c_str(),
                  "relu or leakly_relu or prelu is vaild, "
                  "support ub fusion.");
        } else {
          fusion_nodes.clear();
          OP_LOGW(FUSED_OP_TYPE.c_str(),
                  "relu is not vaild, only support "
                  "Relu or LeakyRelu or Prelu.");
          return SUCCESS;
        }
      }
    }
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do conv2d_bp_input_elemwise!");

  return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("TbeDxElemwisePass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, TbeDxElemwisePass);
}  // namespace fe
