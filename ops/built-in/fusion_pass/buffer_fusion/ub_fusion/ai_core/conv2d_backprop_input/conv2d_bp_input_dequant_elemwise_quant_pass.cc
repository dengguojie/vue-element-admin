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
#include "common/op_slice_info.h"
#include "graph/utils/attr_utils.h"
#include "lx_fusion_func.h"
#include "anchor_util.h"

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

  bool cut_cout_flag = false;
  vector<ge::NodePtr> dequant_nodes = GetMatchedNodesByDescName(PATTERN_DEQUANT, mapping);
  if (dequant_nodes.empty()) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Dequant node not matched");
    return;
  } else {
    auto deq_scale = GetCurrNodeMutableInputDesc(dequant_nodes[0], "deq_scale");
    vector<int64_t> scalar = {1};
    cut_cout_flag = deq_scale != nullptr && deq_scale->GetOriginShape().GetDims() != scalar;
  }

  bool doubleout_flag = false;
  vector<ge::NodePtr> doubleout_nodes = GetMatchedNodesByDescName(PATTERN_OTHER_OUTPUT, mapping);
  if (!doubleout_nodes.empty()) {
    doubleout_flag = true;
    OP_LOGD(FUSED_OP_TYPE.c_str(), "the fusion with doubleout");
  }

  vector<ge::NodePtr> quant_nodes = GetMatchedNodesByDescName(PATTERN_QUANT, mapping);
  if (!quant_nodes.empty()) {
    cut_cout_flag = false;
    OP_LOGD(FUSED_OP_TYPE.c_str(), "the fusion with quant");
  }

  int inpre = deconv_nodes[0]->GetInDataNodes().size() - 1;
  vector<AxisSplitMap> split_maps;
  OpL1FusionType L1_fusion_type = L1FUSION_DISABLE;
  int64_t min_tbe_L1space = 0;
  if (!GetSplitMap(split_maps, deconv_nodes[0], FUSED_OP_TYPE, L1_fusion_type, min_tbe_L1space)) {
     return;
  }

  // the dequant is scala mode or with quant, can not split c_dim
  int c_dim = 1;
  if (!cut_cout_flag) {
    DelSplitInfoByOutputAxis(split_maps, c_dim);
  }

   // when deconv + dequant + prelu, add two input, else add one
  int deq_inpre = inpre + 1;
  int elem_inpre = inpre + 2;
  vector<int64_t> cout_dim = {c_dim};
  vector<int64_t> split_flag = {-1};
  for (auto it = split_maps.begin(); it != split_maps.end(); ++it) {
    auto input_split_infos = (*it).GetInputSplitInfoVec();
    auto output_split_infos = (*it).GetOutputSplitInfoVec();
    if (input_split_infos.empty() || output_split_infos.empty()) {
      continue;
    }
    if (doubleout_flag) {
      vector<int64_t> out_dim = output_split_infos[0].GetAxis();
      OutputSplitInfo double_output_split_info;
      double_output_split_info.Initialize();
      double_output_split_info.SetIndex(1);
      double_output_split_info.SetAxis(out_dim);
      (*it).AddOutputSplitInfo(double_output_split_info);
    }
    if (output_split_infos[0].GetAxis()[0] == 1) {
      if (cut_cout_flag) {
        InputSplitInfo input_split_info_deq;
        input_split_info_deq.Initialize();
        input_split_info_deq.SetIndex(deq_inpre);
        input_split_info_deq.SetAxis(cout_dim);
        input_split_info_deq.SetHeadOverLap(split_flag);
        input_split_info_deq.SetTailOverLap(split_flag);
        (*it).AddInputSplitInfo(input_split_info_deq);
        if (elemwise_nodes[0]->GetType() == "PRelu") {
          InputSplitInfo input_split_info_elem;
          input_split_info_elem.Initialize();
          input_split_info_elem.SetIndex(elem_inpre);
          input_split_info_elem.SetAxis(cout_dim);
          input_split_info_elem.SetHeadOverLap(split_flag);
          input_split_info_elem.SetTailOverLap(split_flag);
          (*it).AddInputSplitInfo(input_split_info_elem);
        }
      }
    }
  }

  SetSplitMap(split_maps, fusion_nodes, FUSED_OP_TYPE, L1_fusion_type, min_tbe_L1space);
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
    auto input0desc = GetCurrNodeInputDesc(dxNode, 0);
    auto input1desc = GetCurrNodeInputDesc(dxNode, 1);
    FUSION_PASS_CHECK(input0desc == nullptr,
                  CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input0desc is null"),
                  return FAILED);
    FUSION_PASS_CHECK(input1desc == nullptr,
                  CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input1desc is null"),
                  return FAILED);
    vector<int64_t> input0Dims = input0desc->GetOriginShape().GetDims();
    vector<int64_t> input1Dims = input1desc->GetOriginShape().GetDims();
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
        if (node_ptr != fusion_nodes.end()) {
          fusion_nodes.erase(node_ptr);
        }
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
  SetSplitInfo(mapping, fusion_nodes);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do conv2d_bp_input_dequant_elemwise_quant!");

  return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("TbeDxDeqElemQuantPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, TbeDxDeqElemQuantPass);
}  // namespace fe
