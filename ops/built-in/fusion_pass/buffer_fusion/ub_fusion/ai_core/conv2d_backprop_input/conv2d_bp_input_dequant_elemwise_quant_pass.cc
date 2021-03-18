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
 * \file conv2d_bp_input_dequant_elemwise_quant_pass.cpp
 * \brief tbe conv2d_backprop_input dequant + elemwise + quant ops fusion pattern
 */
#include "conv2d_bp_input_dequant_elemwise_quant_pass.h"
#include <string>
#include "pattern_fusion_util.h"
#include "op_log.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "common/lxfusion_json_util.h"
#include "graph/utils/attr_utils.h"

namespace fe {

static const char PATTERN_DX[] = "conv2dbackpropinput";
static const char PATTERN_ELEM[] = "elemwise";
static const char PATTERN_DEQUANT[] = "dequant";
static const char PATTERN_QUANT[] = "quant";
static const char PATTERN_OTHER_INPUT[] = "InputData";
static const char PATTERN_OTHER_INPUT1[] = "InputData1";
static const char PATTERN_OTHER_OUTPUT[] = "OutputData";
static vector<string> typelist = {"LeakyRelu", "Prelu"};
/*
 * @brief:  define conv2dbackpropinput op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    conv2d_backprop_input --> dequant --> elemwise --> quant
 *                                                   --> output
 *    elemwise only supported leckyrelu/prelu
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern*> TbeDxDeqElemQuantPass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;
  string pass_name = "TbeDxDequantElemwiseQuantFusion";
  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pass_name);

  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", pass_name.c_str());
  pattern->AddOpDesc(PATTERN_DX, {OP_PATTERN_CONV_BACKPROP_INPUT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_DEQUANT, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEM, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_QUANT, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_DX})
      .SetOutputs(PATTERN_DX, {PATTERN_DEQUANT})
      .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_DEQUANT})
      .SetOutputs(PATTERN_DEQUANT, {PATTERN_ELEM})
      .SetOutputs(PATTERN_ELEM, {PATTERN_QUANT});
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", pass_name.c_str());

  string pass_name0 = "TbeDxDequantElemwiseQuantFusion0";
  BufferFusionPattern* pattern0 = new (std::nothrow) BufferFusionPattern(pass_name0);

  FUSION_PASS_CHECK((pattern0 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", pass_name0.c_str());
  pattern0->AddOpDesc(PATTERN_DX, {OP_PATTERN_CONV_BACKPROP_INPUT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_DEQUANT, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEM, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_QUANT, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_DX})
      .SetOutputs(PATTERN_DX, {PATTERN_DEQUANT})
      .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_DEQUANT})
      .SetOutputs(PATTERN_DEQUANT, {PATTERN_ELEM})
      .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_ELEM})
      .SetOutputs(PATTERN_ELEM, {PATTERN_QUANT});
  patterns.push_back(pattern0);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", pass_name0.c_str());

  string pass_name1 = "TbeDxDequantElemwiseQuantFusionDoubleOut";
  BufferFusionPattern* pattern1 = new (std::nothrow) BufferFusionPattern(pass_name1);

  FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", pass_name1.c_str());
  pattern1->AddOpDesc(PATTERN_DX, {OP_PATTERN_CONV_BACKPROP_INPUT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_DEQUANT, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEM, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_QUANT, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_OUTPUT, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_DX})
      .SetOutputs(PATTERN_DX, {PATTERN_DEQUANT})
      .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_DEQUANT})
      .SetOutputs(PATTERN_DEQUANT, {PATTERN_ELEM})
      .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_ELEM})
      .SetOutputs(PATTERN_ELEM, {PATTERN_QUANT, PATTERN_OTHER_OUTPUT}, TBE_OUTPUT_BRANCH_MULTI);
  patterns.push_back(pattern1);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", pass_name1.c_str());

  string pass_name2 = "TbeDxDequantElemwise";
  BufferFusionPattern* pattern2 = new (std::nothrow) BufferFusionPattern(pass_name2);

  FUSION_PASS_CHECK((pattern2 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", pass_name2.c_str());
  pattern2->AddOpDesc(PATTERN_DX, {OP_PATTERN_CONV_BACKPROP_INPUT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_DEQUANT, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEM, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_DX})
      .SetOutputs(PATTERN_DX, {PATTERN_DEQUANT})
      .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_DEQUANT})
      .SetOutputs(PATTERN_DEQUANT, {PATTERN_ELEM})
      .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_ELEM});
  patterns.push_back(pattern2);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", pass_name2.c_str());

  return patterns;
}

void TbeDxDeqElemQuantPass::DelSplitInfoByAxis(std::vector<AxisSplitMap> &split_maps, int axis) {
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

void TbeDxDeqElemQuantPass::SetSplitInfo(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes) {
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
  vector<ge::NodePtr> dequant_nodes = GetMatchedNodesByDescName(PATTERN_DEQUANT, mapping);
  if (dequant_nodes.empty()) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Dequant node not matched");
    return;
  }
  bool cut_cout_flag = false;
  vector<int64_t> deq_scale_shape;
  for (auto dequant_node : dequant_nodes) {
    deq_scale_shape = dequant_node->GetOpDesc()->GetInputDesc("deq_scale").GetOriginShape().GetDims();
    if(!(deq_scale_shape.size() == 1 && deq_scale_shape[0] == 1)) {
      cut_cout_flag = true;
    }
  }
  vector<ge::NodePtr> quant_nodes = GetMatchedNodesByDescName(PATTERN_QUANT, mapping);
  if (!quant_nodes.empty()) {
    cut_cout_flag = false;
    OP_LOGD(FUSED_OP_TYPE.c_str(), "the fusion with quant");
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

  int c_dim = 1;
  if (!cut_cout_flag) {
    DelSplitInfoByAxis(split_maps, c_dim);
  }

   // when deconv + dequant + prelu, add two input, else add one
  int deq_inpre = inpre + 1;
  int elem_inpre = inpre + 2;
  vector<int64_t> cout_dim = {1};
  vector<int64_t> split_flag = {0};
  for(auto it = split_maps.begin(); it != split_maps.end(); ++it) {
    auto input_split_infos = (*it).GetInputSplitInfoVec();
    auto output_split_infos = (*it).GetOutputSplitInfoVec();
    if (output_split_infos.empty()) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "output_split_infos is empty");
      return;
    }
    if (output_split_infos[0].GetAxis()[0] == 1) {
      if (cut_cout_flag) {
        InputSplitInfo input_split_info_deq;
        input_split_info_deq.SetIndex(deq_inpre);
        input_split_info_deq.SetAxis(cout_dim);
        input_split_info_deq.SetHeadOverLap(split_flag);
        input_split_info_deq.SetTailOverLap(split_flag);
        (*it).AddInputSplitInfo(input_split_info_deq);
        if (elemwise_nodes[0]->GetType() == "PRelu") {
          InputSplitInfo input_split_info_elem;
          input_split_info_elem.SetIndex(elem_inpre);
          input_split_info_elem.SetAxis(cout_dim);
          input_split_info_elem.SetHeadOverLap(split_flag);
          input_split_info_elem.SetTailOverLap(split_flag);
          (*it).AddInputSplitInfo(input_split_info_elem);
        }
      }
    }
  }
  op_calc_info.SetAxisSplitMaps(split_maps);
  SetFusionOpSliceInfoToJson(op_calc_info, op_slice_info_str);
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
Status TbeDxDeqElemQuantPass::GetFusionNodes(const BufferFusionMapping& mapping, vector<ge::NodePtr>& fusion_nodes) {
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

  vector<ge::NodePtr> elemNode = GetMatchedNodesByDescName(PATTERN_ELEM, mapping);
  FUSION_PASS_CHECK(elemNode.empty(),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "get elemNode failed."),
                    return FAILED);
  bool checkElemwise = (elemNode[0]->GetType() == "LeakyRelu" || elemNode[0]->GetType() == "PRelu");
  if (!checkElemwise) {
    fusion_nodes.clear();
    OP_LOGW(FUSED_OP_TYPE.c_str(), "only support LeakyRelu or Prelu");
    return SUCCESS;
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do conv2d_bp_input_dequant_elemwise_quant!");

  return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("TbeDxDeqElemQuantPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, TbeDxDeqElemQuantPass);
}  // namespace fe
