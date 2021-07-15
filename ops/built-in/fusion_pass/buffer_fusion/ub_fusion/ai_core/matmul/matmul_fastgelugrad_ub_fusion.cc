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
 * \file matmul_fastgelugrad_ub_fusion.cpp
 * \brief tbe matmul + elemwise ops fusion pattern
 */
#include "matmul_fastgelugrad_ub_fusion.h"

#include <string>
#include "pattern_fusion_util.h"
#include "op_log.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "common/lxfusion_json_util.h"
#include "graph/utils/attr_utils.h"
#include "lx_fusion_func.h"
#include "anchor_util.h"

namespace fe {

static const char PATTERN_MATMUL[] = "matmul";
static const char PATTERN_ELTWISE[] = "eltwise";
static const char PATTERN_OTHER_INPUT1[] = "otherInput1";
static const char PATTERN_OTHER_INPUT2[] = "otherInput2";
static const char PATTERN_OUTPUT1[] = "OUTPUT1";
static const char PATTERN_OUTPUT2[] = "OUTPUT2";
/*
 * @brief:  define matmul op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    matmul --> elemwise
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern*> MatmulFastGelugradUbFusion::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;
  string passName = "MatmulFastGelugradUbFusion";
  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(passName);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName.c_str());
  pattern->AddOpDesc(PATTERN_MATMUL, {OP_PATTERN_MATMUL}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELTWISE, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OUTPUT1, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OUTPUT2, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_MATMUL})
      .SetOutputs(PATTERN_MATMUL, {PATTERN_ELTWISE, PATTERN_OUTPUT1, PATTERN_OUTPUT2}, TBE_OUTPUT_BRANCH_MULTI)
      .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_ELTWISE});
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName.c_str());
  return patterns;
}

void MatmulFastGelugradUbFusion::SetSplitInfo(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes) {
  vector<ge::NodePtr> matmulNodes = GetMatchedNodesByDescName(PATTERN_MATMUL, mapping);
  vector<ge::NodePtr> elemWiseNodes = GetMatchedNodesByDescName(PATTERN_ELTWISE, mapping);
  if (matmulNodes.empty()) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Matmul node not matched");
    return;
  }
  if (elemWiseNodes.empty()) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Elemwise node not matched");
    return;
  }

  int pre = matmulNodes[0]->GetInDataNodes().size() - 1;
  vector<AxisSplitMap> split_maps;
  if (!GetSplitMap(split_maps, matmulNodes[0], FUSED_OP_TYPE)) {
    return;
  }
  AddElemwiseSplitMap(split_maps, elemWiseNodes[0], pre);
  SetSplitMap(split_maps, fusion_nodes, FUSED_OP_TYPE);
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status MatmulFastGelugradUbFusion::GetFusionNodes(const BufferFusionMapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do MatmulFastGelugradUbFusion!");
  fusionNodes = GetMatchedNodes(mapping);

  // buffer fusion do not support dynamic shape now
  vector<ge::NodePtr> matmulNodes = GetMatchedNodesByDescName(PATTERN_MATMUL, mapping);
  for (const auto& matmulNode : matmulNodes){
    auto input0desc = GetCurrNodeInputDesc(matmulNode, 0);
    auto input1desc = GetCurrNodeInputDesc(matmulNode, 1);
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
        fusionNodes.clear();
        OP_LOGW(FUSED_OP_TYPE.c_str(), "ub fusion not support dynamic shape");
        return SUCCESS;
      }
    }
  }

  // multi input node can not be fused except head node
  for (auto& item : mapping) {
    auto opdesc = find(item.first->types.begin(), item.first->types.end(), TBE_PATTERN_OUTPUT_NODE);
    if (opdesc != item.first->types.end()) {
      for (auto& node : item.second) {
        auto nodePtr = find(fusionNodes.begin(), fusionNodes.end(), node);
        if (nodePtr != fusionNodes.end()) {
          fusionNodes.erase(nodePtr);
        }
      }
    }
  }
  SetSplitInfo(mapping, fusionNodes);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do MatmulFastGelugradUbFusion!");
  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("MatmulFastGelugradUbFusion", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, MatmulFastGelugradUbFusion);
}  // namespace fe
