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
 * \file batch_matmul_elementwise_ub_fusion.cpp
 * \brief tbe batchmatmul and all elementwise ops fusion pattern
 */
#include "batch_matmul_elementwise_ub_fusion.h"

#include <string>
#include <vector>

#include "anchor_util.h"
#include "common/lxfusion_json_util.h"
#include "graph/utils/attr_utils.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "lx_fusion_func.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace fe {
namespace {
static const char PATTERN_BATCH_MATMUL[] = "batchmatmul";
static const char PATTERN_ELEM[] = "elemwise";
static const char PATTERN_ELEM_1[] = "elemwise1";
static const char PATTERN_ELEM_2[] = "elemwise2";
static vector<string> elem_typelist = {"FusedMulAdd", "Add", "Div", "Relu", "ReluGrad"};
static vector<string> elem1_typelist = {"Add", "Relu", "FusedMulAdd"};
static const char PATTERN_OTHER_INPUT[] = "InputData";
static const char PATTERN_OTHER_INPUT1[] = "InputData1";
static const char PATTERN_OTHER_OUTPUT[] = "OutputData";
} // namespace

/*
 * @brief: define Matmul and element-wise op fusion pattern
 *
 * Matmul + Elemwise + (Elemwise1)
 *
 * fusion node: Matmul, Elemwise, (Elemwise1)
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> TbeBatchMatmulElementWiseFusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;

  string passName = "TbeBatchMatmulELEMPASS";
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(passName);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName.c_str());
  // define pattern rules
  pattern->AddOpDesc(PATTERN_BATCH_MATMUL, {OP_PATTERN_BATCH_MATMUL}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEM, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEM_1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_BATCH_MATMUL})
      .SetOutputs(PATTERN_BATCH_MATMUL, {PATTERN_ELEM}, TBE_OUTPUT_BRANCH_SINGLE, true)
      .SetOutputs(PATTERN_ELEM, {PATTERN_ELEM_1}, TBE_OUTPUT_BRANCH_SINGLE, true)
      .SetOutputs(PATTERN_ELEM_1, {}, TBE_OUTPUT_BRANCH_SINGLE, true);
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName.c_str());

  string passName1 = "TbeBatchMatmulELEMPASS";
  BufferFusionPattern *pattern1 = new (std::nothrow) BufferFusionPattern(passName1);
  FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName1.c_str());
  // define pattern rules
  pattern1->AddOpDesc(PATTERN_BATCH_MATMUL, {OP_PATTERN_BATCH_MATMUL}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEM, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_BATCH_MATMUL})
      .SetOutputs(PATTERN_BATCH_MATMUL, {PATTERN_ELEM}, TBE_OUTPUT_BRANCH_SINGLE, true)
      .SetOutputs(PATTERN_ELEM, {}, TBE_OUTPUT_BRANCH_SINGLE, true);
  patterns.push_back(pattern1);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName1.c_str());

  /*
   * BatchmatmulV2 --> Mul --> Sigmoid --> Mul --> Output
   *       \_______________________________/
   */
  string passName2 = "TbeBatchMatmulELEMPASS2";
  BufferFusionPattern *pattern2 = new (std::nothrow) BufferFusionPattern(passName2);
  FUSION_PASS_CHECK((pattern2 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName2.c_str());
  // define pattern rules
  pattern2->AddOpDesc(PATTERN_BATCH_MATMUL, {OP_PATTERN_BATCH_MATMUL}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEM, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEM_1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEM_2, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_OUTPUT, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_BATCH_MATMUL})
      .SetOutputs(PATTERN_BATCH_MATMUL, {PATTERN_ELEM_1, PATTERN_OTHER_OUTPUT}, TBE_OUTPUT_BRANCH_MULTI)
      .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_ELEM_1})
      .SetOutputs(PATTERN_ELEM_1, {PATTERN_ELEM_2}, TBE_OUTPUT_BRANCH_SINGLE)
      .SetOutputs(PATTERN_ELEM_2, {PATTERN_ELEM}, TBE_OUTPUT_BRANCH_SINGLE)
      .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_ELEM});
  patterns.push_back(pattern2);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName2.c_str());

  return patterns;
}

void TbeBatchMatmulElementWiseFusionPass::SetSplitInfo(const BufferFusionMapping &mapping,
                                                       std::vector<ge::NodePtr> &fusion_nodes) {
  vector<ge::NodePtr> matmulNodes = GetMatchedNodesByDescName(PATTERN_BATCH_MATMUL, mapping);
  vector<ge::NodePtr> elemWiseNodes = GetMatchedNodesByDescName(PATTERN_ELEM, mapping);
  vector<ge::NodePtr> elemWiseNodes1 = GetMatchedNodesByDescName(PATTERN_ELEM_1, mapping);
  vector<ge::NodePtr> elemWiseNodes2 = GetMatchedNodesByDescName(PATTERN_ELEM_2, mapping);
  if (matmulNodes.empty()) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Matmul node not matched");
    return;
  }
  if (elemWiseNodes.empty()) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Elemwise node not matched");
    return;
  }
  FUSION_PASS_CHECK(matmulNodes[0]->GetInDataNodes().size() <= 0,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "matmulNodes's input can not <= 0."), return);
  int pre = matmulNodes[0]->GetInDataNodes().size() - 1;
  vector<AxisSplitMap> split_maps;
  OpL1FusionType fusion_type = L1FUSION_DISABLE;
  int64_t min_tbe_l1space = 0;
  if (!GetSplitMap(split_maps, matmulNodes[0], FUSED_OP_TYPE, fusion_type, min_tbe_l1space)) {
    return;
  }
  AddElemwiseSplitMap(split_maps, elemWiseNodes[0], pre);
  if (!elemWiseNodes1.empty()) {
    AddElemwiseSplitMap(split_maps, elemWiseNodes1[0], pre);
  }
  if (!elemWiseNodes2.empty()) {
    AddElemwiseSplitMap(split_maps, elemWiseNodes2[0], pre);
  }
  SetSplitMap(split_maps, fusion_nodes, FUSED_OP_TYPE, fusion_type, min_tbe_l1space);
}

Status TbeBatchMatmulElementWiseFusionPass::CheckPattern1(const BufferFusionMapping &mapping) {
  vector<ge::NodePtr> elemNode = GetMatchedNodesByDescName(PATTERN_ELEM, mapping);
  vector<ge::NodePtr> elemNode1 = GetMatchedNodesByDescName(PATTERN_ELEM_1, mapping);

  FUSION_PASS_CHECK(elemNode.empty(), OP_LOGW(FUSED_OP_TYPE.c_str(), "ElemWise node not match!"), return SUCCESS);
  auto ret = find(elem_typelist.begin(), elem_typelist.end(), elemNode[0]->GetType());
  if (ret == elem_typelist.end()) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "only supported add, div, muladd and Relu in first elemwise");
    return SUCCESS;
  }

  if (!elemNode1.empty()) {
    ret = find(elem1_typelist.begin(), elem1_typelist.end(), elemNode1[0]->GetType());
    if (ret == elem1_typelist.end()) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "only supported add, relu and muladd in second elemwise");
      return SUCCESS;
    }
  }

  return SUCCESS;
}

Status TbeBatchMatmulElementWiseFusionPass::CheckPattern2(const BufferFusionMapping &mapping) {
  vector<ge::NodePtr> elemNode = GetMatchedNodesByDescName(PATTERN_ELEM, mapping);
  vector<ge::NodePtr> elemNode1 = GetMatchedNodesByDescName(PATTERN_ELEM_1, mapping);
  vector<ge::NodePtr> elemNode2 = GetMatchedNodesByDescName(PATTERN_ELEM_2, mapping);
  vector<ge::NodePtr> matmulNodes = GetMatchedNodesByDescName(PATTERN_BATCH_MATMUL, mapping);
  FUSION_PASS_CHECK(elemNode.empty() || elemNode1.empty() || elemNode2.empty() || matmulNodes.empty(),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "ElemWise node not match!"), return SUCCESS);

  bool is_matched = false;
  if (elemNode[0]->GetType() != "Mul" || elemNode1[0]->GetType() != "Mul" || elemNode2[0]->GetType() != "Sigmoid") {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "ElemWise node not match, node name [%s], [%s], [%s].",
            elemNode[0]->GetType().c_str(), elemNode1[0]->GetType().c_str(), elemNode2[0]->GetType().c_str());
    return SUCCESS;
  }

  auto elem_name = elemNode[0]->GetName();
  auto out_nodes = matmulNodes[0]->GetOutDataNodes();
  if (out_nodes.size() != 2) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "matmulNode output size not match!");
    return SUCCESS;
  }

  for (auto &node : out_nodes) {
    is_matched = is_matched || (node->GetName() == elem_name);
  }
  FUSION_PASS_CHECK(!is_matched, OP_LOGW(FUSED_OP_TYPE.c_str(), "ElemWise node name not match!"), return SUCCESS);

  return SUCCESS;
}

Status TbeBatchMatmulElementWiseFusionPass::GetFusionNodes(const BufferFusionMapping &mapping,
                                                           vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do TbeBatchMatmulElementWiseFusionPass!");

  if (SUCCESS != CheckPattern1(mapping) && SUCCESS != CheckPattern2(mapping)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "CheckPattern failed!");
    return SUCCESS;
  }

  fusion_nodes = GetMatchedNodes(mapping);

  // buffer fusion do not support dynamic shape now
  vector<ge::NodePtr> matmulNodes = GetMatchedNodesByDescName(PATTERN_BATCH_MATMUL, mapping);
  for (const auto &matmulNode : matmulNodes) {
    auto input0desc = GetCurrNodeInputDesc(matmulNode, 0);
    auto input1desc = GetCurrNodeInputDesc(matmulNode, 1);
    FUSION_PASS_CHECK(input0desc == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputDesc0 is null"),
                      return SUCCESS);
    FUSION_PASS_CHECK(input1desc == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputDesc1 is null"),
                      return SUCCESS);
    vector<int64_t> input0Dims = input0desc->GetOriginShape().GetDims();
    vector<int64_t> input1Dims = input1desc->GetOriginShape().GetDims();
    vector<int64_t> allDim;
    allDim.resize(input0Dims.size() + input1Dims.size());
    merge(input0Dims.begin(), input0Dims.end(), input1Dims.begin(), input1Dims.end(), allDim.begin());
    for (auto singleDim : allDim) {
      if (singleDim < 0) {
        fusion_nodes.clear();
        OP_LOGW(FUSED_OP_TYPE.c_str(), "ub fusion not support dynamic shape");
        return SUCCESS;
      }
    }
  }

  SetSplitInfo(mapping, fusion_nodes);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do TbeBatchMatmulElementWiseFusionPass!");

  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("TbeBatchMatmulElementWiseFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeBatchMatmulElementWiseFusionPass);
} // namespace fe
