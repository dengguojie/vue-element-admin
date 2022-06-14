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

/*!
 * \file conv3d_elemwise_pass.cpp
 * \brief tbe conv3d + elemwise ops fusion pattern
 */
#include "conv3d_elemwise_pass.h"
#include <string>
#include "pattern_fusion_util.h"
#include "op_log.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "common/lxfusion_json_util.h"
#include "graph/utils/attr_utils.h"
#include "lx_fusion_func.h"
#include "anchor_util.h"

namespace fe {
static const char PATTERN_CONV3D[] = "conv3d";
static const char PATTERN_ELEM[] = "elemwise";
static const char PATTERN_ELEM1[] = "elemwise1";
static const char kPatternOtherInput[] = "otherInput";
static const char kPatternDequant[] = "dequant";
static const char kPatternRequant[] = "requant";
static vector<string> elem_typelist = {"Add"};
static vector<string> elem_typelist1 = {"Relu"};
static const int DIMS_SIZE = 6;
static const int kIndex2 = 2;
static const int kIndex3 = 3;
static const int kIndex4 = 4;
static const int kIndex5 = 5;

/*
 * @brief:  define conv3d op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern*> TbeConv3dElemwisePass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;
  string pass_name1 = "TbeConv3dElemwisePass1";
  BufferFusionPattern* pattern1 = new (std::nothrow) BufferFusionPattern(pass_name1);
  /** conv3d   -->    add --> relu
  *     otherinput  ___/
  */
  FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  pattern1->AddOpDesc(PATTERN_CONV3D, {OP_PATTERN_CONV3D})
    .AddOpDesc(PATTERN_ELEM, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
    .AddOpDesc(PATTERN_ELEM1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
    .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
    .SetHead({PATTERN_CONV3D})
    .SetOutputs(PATTERN_CONV3D, {PATTERN_ELEM})
    .SetOutputs(kPatternOtherInput, {PATTERN_ELEM})
    .SetOutputs(PATTERN_ELEM, {PATTERN_ELEM1});
  patterns.push_back(pattern1);

  string pass_name2 = "TbeConv3dElemwisePass2";
  BufferFusionPattern* pattern2 = new (std::nothrow) BufferFusionPattern(pass_name2);
  /** conv3d   -->    elem
  *     otherinput  ___/
  */
  FUSION_PASS_CHECK((pattern2 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  pattern2->AddOpDesc(PATTERN_CONV3D, {OP_PATTERN_CONV3D})
    .AddOpDesc(PATTERN_ELEM, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
    .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
    .SetHead({PATTERN_CONV3D})
    .SetOutputs(PATTERN_CONV3D, {PATTERN_ELEM})
    .SetOutputs(kPatternOtherInput, {PATTERN_ELEM});
  patterns.push_back(pattern2);

  string pass_name3 = "TbeConv3dElemwisePass3";
  BufferFusionPattern* pattern3 = new (std::nothrow) BufferFusionPattern(pass_name3);
  // conv3d --> relu
  FUSION_PASS_CHECK((pattern3 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  pattern3->AddOpDesc(PATTERN_CONV3D, {OP_PATTERN_CONV3D})
    .AddOpDesc(PATTERN_ELEM, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
    .SetHead({PATTERN_CONV3D})
    .SetOutputs(PATTERN_CONV3D, {PATTERN_ELEM});
  patterns.push_back(pattern3);

  string pass_name4 = "TbeConv3dDequantFusionPass";
  BufferFusionPattern *pattern4 = new (std::nothrow) BufferFusionPattern(pass_name4);
  // conv3d --> dequant
  FUSION_PASS_CHECK((pattern4 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  pattern4-> AddOpDesc(kPatternDequant, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                       TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
    .AddOpDesc(PATTERN_CONV3D, {OP_PATTERN_CONV3D}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
               TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
    .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
               TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
    .SetHead({PATTERN_CONV3D})
    .SetOutputs(PATTERN_CONV3D, {kPatternDequant})
    .SetOutputs(kPatternOtherInput, {kPatternDequant});

  patterns.push_back(pattern4);

  string pass_name5 = "TbeConv3dRequantFusionPass";
  BufferFusionPattern *pattern5 = new (std::nothrow) BufferFusionPattern(pass_name5);
  // conv3d --> requant
  FUSION_PASS_CHECK((pattern5 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  // define pattern rules
  pattern5-> AddOpDesc(PATTERN_CONV3D, {OP_PATTERN_CONV3D}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
               TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
    .AddOpDesc(kPatternRequant, {OP_PATTERN_REQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
               TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
    .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
               TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
    .SetHead({PATTERN_CONV3D})
    .SetOutputs(PATTERN_CONV3D, {kPatternRequant}, TBE_OUTPUT_BRANCH_SINGLE, true)
    .SetOutputs(kPatternOtherInput, {kPatternRequant});

  patterns.push_back(pattern5);

  return patterns;
}

void TbeConv3dElemwisePass::SetSplitInfo(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes) {
  vector<ge::NodePtr> conv3d_nodes = GetMatchedNodesByDescName(PATTERN_CONV3D, mapping);
  FUSION_PASS_CHECK(conv3d_nodes.empty(), OP_LOGW(FUSED_OP_TYPE.c_str(), "Conv3d node not matched"), return);

  vector<ge::NodePtr> elemwise_node = GetMatchedNodesByDescName(PATTERN_ELEM, mapping);
  vector<ge::NodePtr> dequant_node = GetMatchedNodesByDescName(kPatternDequant, mapping);
  vector<ge::NodePtr> requant_node = GetMatchedNodesByDescName(kPatternRequant, mapping);
  FUSION_PASS_CHECK((elemwise_node.empty() && dequant_node.empty() && requant_node.empty()),
                     OP_LOGW(FUSED_OP_TYPE.c_str(), "elemwise node and dequant,requant node not matched"), return);

  vector<int64_t> split_flag = {-1};
  FUSION_PASS_CHECK(conv3d_nodes[0]->GetInDataNodes().size() <= 0,
    OP_LOGE(FUSED_OP_TYPE.c_str(), "conv3d_nodes's input can not <= 0."), return);
  int inpre = conv3d_nodes[0]->GetInDataNodes().size() - 1;
  int fusion_inpre = inpre + 1;

  vector<AxisSplitMap> split_maps;
  OpL1FusionType L1_fusion_type = L1FUSION_DISABLE;
  int64_t min_tbe_L1space = 0;
  if (!GetSplitMap(split_maps, conv3d_nodes[0], FUSED_OP_TYPE, L1_fusion_type, min_tbe_L1space)) {
    return;
  }

  for (auto it = split_maps.begin(); it != split_maps.end(); ++it) {
    auto output_split_infos = (*it).GetOutputSplitInfoVec();
    auto input_split_infos = (*it).GetInputSplitInfoVec();
    if (output_split_infos.empty() || input_split_infos.empty()) {
      continue;
    }
    vector<int64_t> axis = input_split_infos[0].GetAxis();
    InputSplitInfo input_split_info;
    input_split_info.Initialize();
    input_split_info.SetIndex(fusion_inpre);
    input_split_info.SetAxis(axis);
    input_split_info.SetHeadOverLap(split_flag);
    input_split_info.SetTailOverLap(split_flag);
    (*it).AddInputSplitInfo(input_split_info);
  }

  SetSplitMap(split_maps, fusion_nodes, FUSED_OP_TYPE, L1_fusion_type, min_tbe_L1space);
}

/*
 * @brief: parse nodes matched in mapping and dynamic
 * @param [in] mapping: nodes matched by pattern
 * @return uint32_t: fusion status or not.
 */
Status TbeConv3dElemwisePass::CheckDynamicShape(const BufferFusionMapping& mapping) const {
  // buffer fusion do not support dynamic shape now
  vector<ge::NodePtr> conv3dNodes = GetMatchedNodesByDescName(PATTERN_CONV3D, mapping);
  for (const auto& conv3dNode : conv3dNodes) {
    auto conv3dinput0desc = GetCurrNodeInputDesc(conv3dNode, 0);
    auto conv3dinput1desc = GetCurrNodeInputDesc(conv3dNode, 1);
    FUSION_PASS_CHECK(conv3dinput0desc == nullptr,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input0desc is null"), return FAILED);
    FUSION_PASS_CHECK(conv3dinput1desc == nullptr,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input1desc is null"), return FAILED);
    vector<int64_t> input0Dims = conv3dinput0desc->GetOriginShape().GetDims();
    vector<int64_t> input1Dims = conv3dinput1desc->GetOriginShape().GetDims();
    vector<int64_t> allDims;
    allDims.resize(input0Dims.size() + input1Dims.size());
    merge(input0Dims.begin(), input0Dims.end(), input1Dims.begin(), input1Dims.end(), allDims.begin());
    for (auto singleDim : allDims) {
      FUSION_PASS_CHECK(singleDim < 0,
        OP_LOGW(FUSED_OP_TYPE.c_str(), "ub fusion not support dynamic shape"), return NOT_CHANGED);
    }
  }
  return SUCCESS;
}

/*
 * @brief: parse nodes matched in mapping and Elem
 * @param [in] mapping: nodes matched by pattern
 * @return uint32_t: fusion status or not.
 */
Status TbeConv3dElemwisePass::CheckElemInput(const BufferFusionMapping& mapping) const {
  vector<ge::NodePtr> elemNode = GetMatchedNodesByDescName(PATTERN_ELEM, mapping);
  FUSION_PASS_CHECK(elemNode.empty(),
    OP_LOGW(FUSED_OP_TYPE.c_str(), "ElemWise node not match!"), return NOT_CHANGED);
  FUSION_PASS_CHECK(elemNode[0]->GetOpDesc() == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(),
    "ElemWise node not match!"), return NOT_CHANGED);
  auto inputs = elemNode[0]->GetOpDesc()->GetAllInputsDesc();
  FUSION_PASS_CHECK(inputs.size() != 2,
    OP_LOGW(FUSED_OP_TYPE.c_str(), "ElemWise node not match!"), return NOT_CHANGED);
  auto input0desc = GetCurrNodeInputDesc(elemNode[0], 0);
  auto input1desc = GetCurrNodeInputDesc(elemNode[0], 1);
  FUSION_PASS_CHECK(input0desc == nullptr,
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input0desc is null"), return FAILED);
  FUSION_PASS_CHECK(input1desc == nullptr,
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input1desc is null"), return FAILED);
  auto dims0 = input0desc->GetShape().GetDims();
  auto dims1 = input1desc->GetShape().GetDims();
  FUSION_PASS_CHECK(dims0.size() != dims1.size(),
    OP_LOGW(FUSED_OP_TYPE.c_str(), "the dim sizes of two inputs not equal!"),
    return NOT_CHANGED);
  FUSION_PASS_CHECK(dims0.size() != DIMS_SIZE,
    OP_LOGW(FUSED_OP_TYPE.c_str(), "the dim sizes is not 6!"),
    return NOT_CHANGED);
  auto fusionShape0 = std::vector<int64_t>{dims0[0] * dims0[1], dims0[kIndex2],
    dims0[kIndex3] * dims0[kIndex4], dims0[kIndex5]};
  auto fusionShape1 = std::vector<int64_t>{dims1[0] * dims1[1], dims1[kIndex2],
    dims1[kIndex3] * dims1[kIndex4], dims1[kIndex5]};
  for (size_t i = 0; i < fusionShape0.size(); ++i) {
    FUSION_PASS_CHECK(fusionShape0[i] != fusionShape1[i] &&
      fusionShape0[i] != 1 && fusionShape1[i] != 1,
      OP_LOGW(FUSED_OP_TYPE.c_str(), "the shape can not support to fuse!"),
      return NOT_CHANGED);
  }
  return SUCCESS;
}

/*
 * @brief: parse nodes matched in mapping and check pattern limit
 * @param [in] mapping: nodes matched by pattern
 * @return uint32_t: fusion status or not.
 */
Status TbeConv3dElemwisePass::CheckElemType(const BufferFusionMapping &mapping, const string &elem,
                                            const vector<string> &elem_types) const {
  vector<ge::NodePtr> elemNode = GetMatchedNodesByDescName(elem, mapping);
  FUSION_PASS_CHECK(elemNode.empty(), OP_LOGW(FUSED_OP_TYPE.c_str(), "ElemWise node not match!"), return NOT_CHANGED);
  auto ret = find(elem_types.begin(), elem_types.end(), elemNode[0]->GetType());
  if (ret == elem_types.end()) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "only supported add and Relu in first elemwise");
    return NOT_CHANGED;
  }
  return SUCCESS;
}

/*
 * @brief: parse nodes matched in mapping and check pattern limit
 * @param [in] mapping: nodes matched by pattern
 * @return uint32_t: fusion status or not.
 */
Status TbeConv3dElemwisePass::CheckPattern1(const BufferFusionMapping& mapping) const {
  FUSION_PASS_CHECK(CheckElemInput(mapping) != SUCCESS,
    OP_LOGW(FUSED_OP_TYPE.c_str(), "ElemWise node not match!"), return NOT_CHANGED);

  FUSION_PASS_CHECK(CheckElemType(mapping, PATTERN_ELEM, elem_typelist) != SUCCESS,
    OP_LOGW(FUSED_OP_TYPE.c_str(), "ElemWise Type not match!"), return NOT_CHANGED);

  FUSION_PASS_CHECK(CheckElemType(mapping, PATTERN_ELEM1, elem_typelist1) != SUCCESS,
    OP_LOGW(FUSED_OP_TYPE.c_str(), "ElemWise1 Type not match!"), return NOT_CHANGED);

  return SUCCESS;
}

Status TbeConv3dElemwisePass::CheckPattern2(const BufferFusionMapping& mapping) const {
  FUSION_PASS_CHECK(CheckElemInput(mapping) != SUCCESS,
    OP_LOGW(FUSED_OP_TYPE.c_str(), "ElemWise node not match!"), return NOT_CHANGED);

  return SUCCESS;
}

/*
 * @brief: parse nodes matched in mapping and check pattern limit
 * @param [in] mapping: nodes matched by pattern
 * @return uint32_t: fusion status or not.
 */
Status TbeConv3dElemwisePass::CheckPattern3(const BufferFusionMapping& mapping) const {
  vector<ge::NodePtr> elemNode = GetMatchedNodesByDescName(PATTERN_ELEM, mapping);
  FUSION_PASS_CHECK(elemNode.empty(),
    OP_LOGW(FUSED_OP_TYPE.c_str(), "ElemWise node not match!"), return NOT_CHANGED);
  FUSION_PASS_CHECK(elemNode[0]->GetOpDesc() == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(),
    "elemdesc is null!"), return FAILED);
  auto inputs = elemNode[0]->GetOpDesc()->GetAllInputsDesc();
  FUSION_PASS_CHECK(inputs.size() != 1,
    OP_LOGW(FUSED_OP_TYPE.c_str(), "ElemWise node input num not match!"), return NOT_CHANGED);
  auto input0desc = GetCurrNodeInputDesc(elemNode[0], 0);
  FUSION_PASS_CHECK(input0desc == nullptr,
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input0desc is null"), return FAILED);
  auto ret = find(elem_typelist1.begin(), elem_typelist1.end(), elemNode[0]->GetType());
  if (ret == elem_typelist1.end()) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "only supported add and Relu in first elemwise");
    return NOT_CHANGED;
  }

  return SUCCESS;
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] mapping: nodes matched by pattern
 * @param [out] fusion_nodes: the nodes of fusion
 * @return uint32_t: fusion status or not.
 */
Status TbeConv3dElemwisePass::GetFusionNodes(const BufferFusionMapping& mapping,
                                             vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do conv3d_elemwise!");
  if (SUCCESS != CheckDynamicShape(mapping)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "CheckPattern failed!");
    return SUCCESS;
  }
  vector<ge::NodePtr> elemNode = GetMatchedNodesByDescName(PATTERN_ELEM, mapping);
  if (!elemNode.empty() && SUCCESS != CheckPattern1(mapping) &&
      SUCCESS != CheckPattern2(mapping) && SUCCESS != CheckPattern3(mapping)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "CheckPattern failed!");
    return SUCCESS;
  }
  fusion_nodes = GetMatchedNodes(mapping);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do conv3d_elemwise!");
  SetSplitInfo(mapping, fusion_nodes);
  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("TbeConv3dElemwisePass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, TbeConv3dElemwisePass);
}  // namespace fe
