/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file conv3d_bp_input_elemwise_pass.cpp
 * \brief tbe conv3d_backprop_input + elemwise ops fusion pattern
 */
#include "conv3d_bp_input_elemwise_pass.h"
#include <string>
#include "pattern_fusion_util.h"
#include "op_log.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "common/lxfusion_json_util.h"
#include "graph/utils/attr_utils.h"
#include "lx_fusion_func.h"
#include "anchor_util.h"

namespace fe {
static const string PATTERN_DX = "conv3dbackpropinput";
static const string PATTERN_ELEM = "elemwise";
static const string PATTERN_ELEM1 = "elemwise1";
static const string PATTERN_OTHER_INPUT0 = "InputData0";
static const string PATTERN_OTHER_INPUT1 = "InputData1";
static const string PATTERN_OTHER_INPUT2 = "InputData2";
static const int32_t INPUTS_NUM_MAX = 3;
static const int32_t INPUTS_NUM_MIN = 2;
static vector<string> typelist = {"AddN", "LeakyReluGrad"};
/*
 * @brief:  define conv3dbackpropinput op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    conv3d_backprop_input --> elemwise
 *    now only conv3d_backprop_input --> AddN --> LeakyReluGrad
 *             conv3d_backprop_input --> LeakyReluGrad
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern*> TbeConv3dDxElemwisePass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;
  string pass_name = "TbeConv3dDxElemwisePass";
  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pass_name);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", pass_name.c_str());
  pattern->AddOpDesc(PATTERN_DX, {OP_PATTERN_CONV3D_BACKPROP_INPUT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
    .AddOpDesc(PATTERN_ELEM, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
    .AddOpDesc(PATTERN_ELEM1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
    .AddOpDesc(PATTERN_OTHER_INPUT0, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
    .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
    .AddOpDesc(PATTERN_OTHER_INPUT2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
    .SetHead({PATTERN_DX})
    .SetOutputs(PATTERN_DX, {PATTERN_ELEM})
    .SetOutputs(PATTERN_OTHER_INPUT0, {PATTERN_ELEM})
    .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_ELEM})
    .SetOutputs(PATTERN_ELEM, {PATTERN_ELEM1})
    .SetOutputs(PATTERN_OTHER_INPUT2, {PATTERN_ELEM1});
  patterns.push_back(pattern);

  string pass_name1 = "TbeConv3dDxElemwisePass1";
  BufferFusionPattern* pattern1 = new (std::nothrow) BufferFusionPattern(pass_name1);
  FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", pass_name1.c_str());
  pattern1->AddOpDesc(PATTERN_DX, {OP_PATTERN_CONV3D_BACKPROP_INPUT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
    .AddOpDesc(PATTERN_ELEM, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
    .AddOpDesc(PATTERN_ELEM1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
    .AddOpDesc(PATTERN_OTHER_INPUT0, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
    .AddOpDesc(PATTERN_OTHER_INPUT2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
    .SetHead({PATTERN_DX})
    .SetOutputs(PATTERN_DX, {PATTERN_ELEM})
    .SetOutputs(PATTERN_OTHER_INPUT0, {PATTERN_ELEM})
    .SetOutputs(PATTERN_ELEM, {PATTERN_ELEM1})
    .SetOutputs(PATTERN_OTHER_INPUT2, {PATTERN_ELEM1});
  patterns.push_back(pattern1);

  string pass_name2 = "TbeConv3dDxElemwisePass2";
  BufferFusionPattern* pattern2 = new (std::nothrow) BufferFusionPattern(pass_name2);
  FUSION_PASS_CHECK((pattern2 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", pass_name2.c_str());
  pattern2->AddOpDesc(PATTERN_DX, {OP_PATTERN_CONV3D_BACKPROP_INPUT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
    .AddOpDesc(PATTERN_ELEM1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
    .AddOpDesc(PATTERN_OTHER_INPUT2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
    .SetHead({PATTERN_DX})
    .SetOutputs(PATTERN_DX, {PATTERN_ELEM1})
    .SetOutputs(PATTERN_OTHER_INPUT2, {PATTERN_ELEM1});
  patterns.push_back(pattern2);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", pass_name.c_str());

  return patterns;
}

void TbeConv3dDxElemwisePass::SetSplitInfo(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes) {
  vector<ge::NodePtr> dx_nodes = GetMatchedNodesByDescName(PATTERN_DX, mapping);
  FUSION_PASS_CHECK(dx_nodes.empty(), OP_LOGW(FUSED_OP_TYPE.c_str(), "Conv3d_backprop node not matched"), return);

  vector<ge::NodePtr> elemwise_node = GetMatchedNodesByDescName(PATTERN_ELEM, mapping);

  vector<ge::NodePtr> elemwise1_node = GetMatchedNodesByDescName(PATTERN_ELEM1, mapping);
  if (elemwise_node.empty() && elemwise1_node.empty()) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "elemwise node not matched");
    return;
  }

  FUSION_PASS_CHECK(dx_nodes[0]->GetInDataNodes().size() <= 0,
    OP_LOGE(FUSED_OP_TYPE.c_str(), "conv3d_backprop_input's inputs can not <= 0."), return);
  int inpre = dx_nodes[0]->GetInDataNodes().size() - 1;

  vector<int64_t> split_flag = {-1};
  int fusion_inpre = 0;
  // when dx + leakyrelugrad, add 1 input. when dx + addn + leakyrelugrad, add 2 or 3
  if (elemwise_node.empty()) {
    fusion_inpre =  inpre + 1;
  } else {
    int addn_inpre = 0;
    FUSION_PASS_CHECK(elemwise_node[0]->GetInDataNodes().size() <= 0,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "elemwise_node's inputs can not <= 0."), return);
    addn_inpre = elemwise_node[0]->GetInDataNodes().size() - 1;
    fusion_inpre = inpre + addn_inpre + 1;
  }

  vector<AxisSplitMap> split_maps;
  OpL1FusionType L1_fusion_type = L1FUSION_DISABLE;
  int64_t min_tbe_L1space = 0;
  if (!GetSplitMap(split_maps, dx_nodes[0], FUSED_OP_TYPE, L1_fusion_type, min_tbe_L1space)) {
    return;
  }

  for (int i = inpre + 1; i <= fusion_inpre; i++) {
    for (auto it = split_maps.begin(); it != split_maps.end(); ++it) {
      auto output_split_infos = (*it).GetOutputSplitInfoVec();
      auto input_split_infos = (*it).GetInputSplitInfoVec();
      if (output_split_infos.empty() || input_split_infos.empty()) {
        continue;
      }

      vector<int64_t> axis = input_split_infos[0].GetAxis();
      InputSplitInfo input_split_info;
      input_split_info.Initialize();
      input_split_info.SetAxis(axis);
      input_split_info.SetIndex(i);
      input_split_info.SetHeadOverLap(split_flag);
      input_split_info.SetTailOverLap(split_flag);
      (*it).AddInputSplitInfo(input_split_info);
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
Status TbeConv3dDxElemwisePass::GetFusionNodes(const BufferFusionMapping& mapping,
                                               vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do conv3d_bp_input_elemwise!");
  // buffer fusion do not support dynamic shape now
  vector<ge::NodePtr> dxNodes = GetMatchedNodesByDescName(PATTERN_DX, mapping);
  for (const auto& dxNode : dxNodes){
    auto input0desc = GetCurrNodeInputDesc(dxNode, 0);
    auto input1desc = GetCurrNodeInputDesc(dxNode, 1);
    FUSION_PASS_CHECK(input0desc == nullptr,
                  CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputDesc0 is null"),
                  return FAILED);
    FUSION_PASS_CHECK(input1desc == nullptr,
                  CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputDesc1 is null"),
                  return FAILED);
    vector<int64_t> input0Dims = input0desc->GetOriginShape().GetDims();
    vector<int64_t> input1Dims = input1desc->GetOriginShape().GetDims();
    vector<int64_t> allDims;
    allDims.resize(input0Dims.size() + input1Dims.size());
    merge(input0Dims.begin(), input0Dims.end(), input1Dims.begin(), input1Dims.end(), allDims.begin());
    for (auto singleDim : allDims) {
      if (singleDim < 0) {
        OP_LOGW(FUSED_OP_TYPE.c_str(), "ub fusion not support dynamic shape");
        return SUCCESS;
      }
    }
  }

  vector<ge::NodePtr> elem2_node = GetMatchedNodesByDescName(PATTERN_ELEM1, mapping);
  for (auto& node : elem2_node) {
    if (node->GetType() != typelist[1]) {
      OP_LOGW(FUSED_OP_TYPE.c_str(),
              "the second elemwise node must be LeakyReluGrad, actually [%s].",
              node->GetType().c_str());
      return SUCCESS;
    }
  }

  vector<ge::NodePtr> elem_node = GetMatchedNodesByDescName(PATTERN_ELEM, mapping);
  for (auto& node : elem_node) {
    if (node->GetType() != typelist[0]) {
      OP_LOGW(FUSED_OP_TYPE.c_str(),
              "the first elemwise node must be AddN, actually [%s].",
              node->GetType().c_str());
      return SUCCESS;
    }
  }

  if (elem_node.empty() && elem2_node.empty()) {
    OP_LOGW(FUSED_OP_TYPE.c_str(),
            "the num of elemwise nodes is empty.");
    return SUCCESS;
  }

  fusion_nodes = GetMatchedNodes(mapping);

  SetSplitInfo(mapping, fusion_nodes);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do conv3d_bp_input_elemwise!");

  return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("TbeConv3dDxElemwisePass",
                            BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeConv3dDxElemwisePass);
}  // namespace fe
